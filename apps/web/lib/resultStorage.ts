import type { AnalyzeResult } from "./api";

export const RESULT_STORAGE_KEY = "causasent.lastResult";
const PENDING_JOB_KEY = "causasent.pendingJob";

export function persistResult(r: AnalyzeResult): void {
  try {
    sessionStorage.setItem(RESULT_STORAGE_KEY, JSON.stringify(r));
  } catch {
    /* sessionStorage may be unavailable; dashboard handles null */
  }
}

export function loadResult(): AnalyzeResult | null {
  try {
    const raw = sessionStorage.getItem(RESULT_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as AnalyzeResult) : null;
  } catch {
    return null;
  }
}

export interface PendingJob {
  reviews: string[];
  useLlm: boolean;
  source: "paste" | "csv" | "sample";
  csvFilename?: string;
  startedAt: number;
}

export function persistPendingJob(job: PendingJob): void {
  try {
    sessionStorage.setItem(PENDING_JOB_KEY, JSON.stringify(job));
  } catch {
    /* noop */
  }
}

export function takePendingJob(): PendingJob | null {
  try {
    const raw = sessionStorage.getItem(PENDING_JOB_KEY);
    if (!raw) return null;
    sessionStorage.removeItem(PENDING_JOB_KEY); // consume once
    return JSON.parse(raw) as PendingJob;
  } catch {
    return null;
  }
}
