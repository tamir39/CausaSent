// Typed client for the CausaSent FastAPI backend.

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

// -----------------------------------------------------------------------------
// Wire-format types — mirror src/inference/aggregate.py and action_llm.py.
// -----------------------------------------------------------------------------

export type Sentiment = "positive" | "negative";

export type AspectCategory =
  | "delivery"
  | "packaging"
  | "product_quality"
  | "price"
  | "customer_service"
  | "usability"
  | "appearance";

export interface PredictedTuple {
  aspect_category: AspectCategory;
  aspect_term: string;
  aspect_term_span: [number, number];
  sentiment: Sentiment;
  confidence: number;
}

export interface AspectCell {
  aspect_category: AspectCategory;
  sentiment: Sentiment;
  count: number;
  top_terms: [string, number][];
}

export interface AspectSummary {
  aspect_category: AspectCategory;
  total_mentions: number;
  n_reviews: number;
  positive: number;
  negative: number;
  negative_ratio: number;
  cells: Partial<Record<Sentiment, AspectCell>>;
}

export interface CorpusSummary {
  n_reviews: number;
  n_reviews_with_tuples: number;
  n_tuples: number;
  sentiment_overall: Partial<Record<Sentiment, number>>;
  aspects: Partial<Record<AspectCategory, AspectSummary>>;
}

export interface ActionRecommendation {
  aspect_category: AspectCategory;
  sentiment: Sentiment;
  priority: "high" | "medium" | "low";
  action: string;
  evidence_terms: string[];
}

export interface PerReview {
  id: string;
  review: string;
  tuples: PredictedTuple[];
}

export interface AnalyzeResult {
  n_reviews: number;
  summary: CorpusSummary;
  actions: ActionRecommendation[];
  per_review: PerReview[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  ckpt_exists: boolean;
  gemini_enabled: boolean;
}

export interface LabelsResponse {
  aspects: AspectCategory[];
  aspect_labels_vi: Record<AspectCategory, string>;
  sentiments: Sentiment[];
}

// -----------------------------------------------------------------------------
// Client methods.
// -----------------------------------------------------------------------------

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail || JSON.stringify(j);
    } catch {
      // ignore
    }
    throw new Error(`[${res.status}] ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return jsonOrThrow<HealthResponse>(
    await fetch(`${API_BASE}/health`, { cache: "no-store" })
  );
}

export async function getLabels(): Promise<LabelsResponse> {
  return jsonOrThrow<LabelsResponse>(
    await fetch(`${API_BASE}/labels`, { cache: "no-store" })
  );
}

export async function analyze(
  reviews: string[],
  opts: { useLlm?: boolean; topK?: number; minConfidence?: number } = {}
): Promise<AnalyzeResult> {
  return jsonOrThrow<AnalyzeResult>(
    await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reviews,
        use_llm: opts.useLlm ?? true,
        top_k: opts.topK ?? 5,
        min_confidence: opts.minConfidence ?? 0.0,
      }),
    })
  );
}

export async function analyzeCsv(
  file: File,
  opts: {
    reviewCol?: string;
    useLlm?: boolean;
    topK?: number;
    minConfidence?: number;
  } = {}
): Promise<AnalyzeResult> {
  const fd = new FormData();
  fd.append("file", file);
  if (opts.reviewCol) fd.append("review_col", opts.reviewCol);
  fd.append("use_llm", String(opts.useLlm ?? true));
  fd.append("top_k", String(opts.topK ?? 5));
  fd.append("min_confidence", String(opts.minConfidence ?? 0.0));
  return jsonOrThrow<AnalyzeResult>(
    await fetch(`${API_BASE}/analyze-csv`, { method: "POST", body: fd })
  );
}

// -----------------------------------------------------------------------------
// Streaming (Server-Sent Events) — for progressive dashboard updates.
// -----------------------------------------------------------------------------

export type StreamEventType =
  | "start"
  | "review"
  | "summary"
  | "actions_pending"
  | "actions"
  | "done"
  | "error";

export interface StreamReviewEvent {
  idx: number;
  id: string;
  review: string;
  tuples: PredictedTuple[];
}

export interface StreamSummaryEvent {
  summary: CorpusSummary;
}

export interface StreamActionsEvent {
  actions: ActionRecommendation[];
}

export interface StreamStartEvent {
  n_reviews: number;
  use_llm: boolean;
}

export interface StreamDoneEvent {
  n_reviews: number;
  n_tuples: number;
  n_actions: number;
  elapsed_ms: number;
}

export type StreamHandler = (
  type: StreamEventType,
  data:
    | StreamReviewEvent
    | StreamSummaryEvent
    | StreamActionsEvent
    | StreamStartEvent
    | StreamDoneEvent
    | { use_llm?: boolean }
    | Record<string, never>
) => void;

export async function analyzeStream(
  reviews: string[],
  opts: { useLlm?: boolean; topK?: number; minConfidence?: number } = {},
  onEvent: StreamHandler,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${API_BASE}/analyze-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      reviews,
      use_llm: opts.useLlm ?? true,
      top_k: opts.topK ?? 5,
      min_confidence: opts.minConfidence ?? 0.0,
    }),
    signal,
  });
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      detail = JSON.stringify(await res.json());
    } catch {
      /* ignore */
    }
    throw new Error(`[${res.status}] ${detail}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line ("\n\n").
    let sep = buffer.indexOf("\n\n");
    while (sep !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      sep = buffer.indexOf("\n\n");
      let event: StreamEventType = "error";
      let dataStr = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7).trim() as StreamEventType;
        else if (line.startsWith("data: ")) dataStr = line.slice(6);
      }
      if (!dataStr) continue;
      try {
        onEvent(event, JSON.parse(dataStr));
      } catch (e) {
        // bad frame — surface but keep reading
        onEvent("error", { message: (e as Error).message } as never);
      }
    }
  }
}
