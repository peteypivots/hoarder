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
  // Use file-based polling queue instead of HTTP to avoid timeouts
  const jobsDir = "/data/transcription_jobs";
  if (!fs.existsSync(jobsDir)) {
    await fs.promises.mkdir(jobsDir, { recursive: true });
  }
  
  const uniqueId = `${jobId}-${Date.now()}`;
  const jobWavPath = path.join(jobsDir, `job_${uniqueId}.wav`);
  const jobJsonPath = path.join(jobsDir, `job_${uniqueId}.json`);
  const jobDonePath = path.join(jobsDir, `job_${uniqueId}.done`);
  const jobErrorPath = path.join(jobsDir, `job_${uniqueId}.error`);
  
  logger.info(`[Transcription][${jobId}] Using file-based queue: ${uniqueId}`);
  
  // 1. Copy audio to shared volume
  await fs.promises.copyFile(wavPath, jobWavPath);
  
  // 2. Create job file to trigger transcriber
  await fs.promises.writeFile(jobJsonPath, JSON.stringify({
    id: uniqueId,
    created: Date.now(),
    originalJobId: jobId
  }));
  
  logger.info(`[Transcription][${jobId}] Job enqueued, waiting for results...`);
  
  // 3. Poll for result (max 2 hours)
  const startTime = Date.now();
  const MAX_WAIT = 2 * 60 * 60 * 1000; 
  
  try {
    while (Date.now() - startTime < MAX_WAIT) {
      await new Promise(resolve => setTimeout(resolve, 2000)); // Poll every 2s
      
      // Check for success
      if (fs.existsSync(jobDonePath)) {
        const resultData = JSON.parse(await fs.promises.readFile(jobDonePath, 'utf8'));
        logger.info(`[Transcription][${jobId}] Job completed successfully`);
        
        // Cleanup
        await fs.promises.unlink(jobWavPath).catch(() => {});
        await fs.promises.unlink(jobJsonPath).catch(() => {});
        await fs.promises.unlink(jobDonePath).catch(() => {});

        return resultData.transcript;
      }
      
      // Check for error
      if (fs.existsSync(jobErrorPath)) {
        const errorData = JSON.parse(await fs.promises.readFile(jobErrorPath, 'utf8'));
        
        // Cleanup
        await fs.promises.unlink(jobWavPath).catch(() => {});
        await fs.promises.unlink(jobJsonPath).catch(() => {});
        await fs.promises.unlink(jobErrorPath).catch(() => {});

        throw new Error(errorData.error || "Unknown error from transcriber");
      }
    }
    
    throw new Error("Transcription timed out after 2 hours");
    
  } finally {
    // Cleanup files
    const files = [jobWavPath, jobJsonPath, jobDonePath, jobErrorPath, path.join(jobsDir, `job_${uniqueId}.lock`)];
    for (const file of files) {
      if (fs.existsSync(file)) {
        await fs.promises.unlink(file).catch(err => logger.warn(`Failed to cleanup ${file}: ${err}`));
      }
    }
  }
}

function inferModelName(transcript: string): string {
  // Prefer inferring from the transcript itself, since the workers service may not
  // share the same env/config as the transcriber container.
  const t = transcript || "";
  const looksDiarized = /\bSPEAKER_\d+\b/.test(t) || /\bUnknown:\b/.test(t);
  if (looksDiarized) {
    return "Whisper Large-v3 Turbo + Diarization";
  }

  if (process.env.USE_WHISPER_FOR_LONG_AUDIO === "true") {
    return "Whisper Large-v3 (auto-select)";
  }

  if (process.env.USE_PYTHON_TRANSCRIPTION === "true") {
    return "GLM-ASR-Nano-2512";
  }

  return "Ollama";
}

function generateTranscriptHTML(transcript: string): string {
  const modelName = inferModelName(transcript);
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
