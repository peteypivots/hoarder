import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { spawn } from "child_process";
import { execa } from "execa";
import { workerStatsCounter } from "metrics";

import { db } from "@karakeep/db";
import { assets, AssetTypes } from "@karakeep/db/schema";
import {
  TranscriptionQueue,
  ZTranscriptionRequest,
  zTranscriptionRequestSchema,
  QuotaService,
} from "@karakeep/shared-server";
import { readAsset, saveAsset, newAssetId, ASSET_TYPES } from "@karakeep/shared/assetdb";
import serverConfig from "@karakeep/shared/config";
import logger from "@karakeep/shared/logger";
import { DequeuedJob, getQueueClient } from "@karakeep/shared/queueing";
import { eq } from "drizzle-orm";

const TMP_FOLDER = path.join(os.tmpdir(), "transcription");

export class TranscriptionWorker {
  static async build() {
    logger.info("Starting transcription worker ...");

    return (await getQueueClient())!.createRunner<ZTranscriptionRequest>(
      TranscriptionQueue,
      {
        run: runWorker,
        onComplete: async (job) => {
          workerStatsCounter.labels("transcription", "completed").inc();
          const jobId = job.id;
          logger.info(
            `[Transcription][${jobId}] Transcription completed successfully`,
          );
          return Promise.resolve();
        },
        onError: async (job) => {
          workerStatsCounter.labels("transcription", "failed").inc();
          if (job.numRetriesLeft == 0) {
            workerStatsCounter.labels("transcription", "failed_permanent").inc();
          }
          const jobId = job.id;
          logger.error(
            `[Transcription][${jobId}] Transcription job failed: ${job.error}`,
          );
          return Promise.resolve();
        },
      },
      {
        pollIntervalMs: 5000,
        timeoutSecs: 600, // 10 minutes for transcription
        concurrency: 1, // Serialize to prevent CUDA OOM
        validator: zTranscriptionRequestSchema,
      },
    );
  }
}

async function extractAudio(videoPath: string, jobId: string): Promise<string> {
  await fs.promises.mkdir(TMP_FOLDER, { recursive: true });
  const wavPath = path.join(
    TMP_FOLDER,
    `${path.basename(videoPath, path.extname(videoPath))}-${Date.now()}.wav`
  );

  logger.info(`[Transcription][${jobId}] Extracting audio to ${wavPath}`);

  await execa("ffmpeg", [
    "-y",
    "-i",
    videoPath,
    "-ac",
    "1",
    "-ar",
    "16000",
    "-f",
    "wav",
    wavPath,
  ]);

  return wavPath;
}

async function transcribeAudio(wavPath: string, jobId: string): Promise<string> {
  const transcriberUrl = process.env.TRANSCRIBER_SERVICE_URL || "http://transcriber:8081";
  
  logger.info(`[Transcription][${jobId}] Sending audio to transcriber service at ${transcriberUrl}`);

  // Read the WAV file as binary buffer
  const audioBuffer = await fs.promises.readFile(wavPath);

  // Send raw audio data
  const response = await fetch(`${transcriberUrl}/transcribe`, {
    method: 'POST',
    body: audioBuffer,
    headers: {
      'Content-Type': 'audio/wav',
      'Content-Length': audioBuffer.length.toString(),
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Transcriber service returned ${response.status}: ${errorText}`);
  }

  const result = await response.json();
  
  if (!result.transcript) {
    throw new Error("Transcriber service did not return a transcript");
  }

  logger.info(`[Transcription][${jobId}] Received transcript (${result.transcript.length} chars)`);
  
  return result.transcript;
}

function generateTranscriptHTML(transcript: string): string {
  const modelName = process.env.USE_PYTHON_TRANSCRIPTION === "true" ? "GLM-ASR-Nano-2512" : "Ollama";
  const timestamp = new Date().toISOString();

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Audio Transcription</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }
    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    .metadata { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }
    .metadata strong { color: #2c3e50; }
    .transcript { white-space: pre-wrap; background: #fff; padding: 20px; border-left: 4px solid #3498db; }
  </style>
</head>
<body>
  <h1>🎙️ Audio Transcription</h1>
  <div class="metadata">
    <p><strong>Model:</strong> ${modelName}</p>
    <p><strong>Generated:</strong> ${timestamp}</p>
  </div>
  <div class="transcript">${transcript.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>
</body>
</html>`;
}

async function uploadTranscript(
  bookmarkId: string,
  userId: string,
  htmlContent: string,
  jobId: string
): Promise<string> {
  const assetId = newAssetId();
  const transcriptBuffer = Buffer.from(htmlContent, 'utf-8');

  logger.info(`[Transcription][${jobId}] Saving transcript as asset ${assetId}`);

  // Check storage quota before saving
  const quotaApproved = await QuotaService.checkStorageQuota(
    db,
    userId,
    transcriptBuffer.byteLength,
  );

  // Save the transcript as an asset
  await saveAsset({
    userId,
    assetId,
    asset: transcriptBuffer,
    metadata: {
      contentType: ASSET_TYPES.TEXT_HTML,
      fileName: "transcript.html",
    },
    quotaApproved,
  });

  logger.info(`[Transcription][${jobId}] Saved transcript as asset ${assetId}`);

  // Insert asset metadata into database
  await db.insert(assets).values({
    id: assetId,
    assetType: AssetTypes.USER_UPLOADED,
    bookmarkId,
    userId,
    size: transcriptBuffer.byteLength,
    contentType: ASSET_TYPES.TEXT_HTML,
    fileName: "transcript.html",
  });

  logger.info(`[Transcription][${jobId}] Attached transcript to bookmark ${bookmarkId}`);

  return assetId;
}

async function runWorker(job: DequeuedJob<ZTranscriptionRequest>) {
  const jobId = job.id;
  const { bookmarkId, assetId, userId } = job.data;

  logger.info(`[Transcription][${jobId}] Processing bookmark ${bookmarkId}, asset ${assetId}`);

  // Read the video asset
  const videoAssetData = await readAsset({
    userId,
    assetId,
  });

  if (!videoAssetData) {
    throw new Error(`Asset ${assetId} not found for user ${userId}`);
  }

  // Save to temp file for processing
  const tmpVideoPath = path.join(TMP_FOLDER, `video-${assetId}-${Date.now()}.bin`);
  await fs.promises.mkdir(TMP_FOLDER, { recursive: true });
  await fs.promises.writeFile(tmpVideoPath, videoAssetData.asset);

  let wavPath: string | null = null;

  try {
    // Extract audio
    wavPath = await extractAudio(tmpVideoPath, jobId);

    // Transcribe
    const transcript = await transcribeAudio(wavPath, jobId);

    if (!transcript || transcript.trim().length === 0) {
      throw new Error("Transcription returned empty result");
    }

    logger.info(`[Transcription][${jobId}] Transcript length: ${transcript.length} chars`);

    // Generate HTML
    const htmlContent = generateTranscriptHTML(transcript);

    // Upload and attach
    await uploadTranscript(bookmarkId, userId, htmlContent, jobId);

    logger.info(`[Transcription][${jobId}] Successfully attached transcript to bookmark`);
  } finally {
    // Cleanup temp files
    try {
      await fs.promises.rm(tmpVideoPath, { force: true });
      if (wavPath) {
        await fs.promises.rm(wavPath, { force: true });
      }
    } catch (e) {
      logger.warn(`[Transcription][${jobId}] Failed to cleanup temp files: ${e}`);
    }
  }
}
