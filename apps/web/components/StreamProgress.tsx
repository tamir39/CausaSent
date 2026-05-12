"use client";

import { useEffect, useState } from "react";

import { ASPECT_LABEL_VI } from "@/lib/labels";
import type { PredictedTuple } from "@/lib/api";

export type Stage =
  | "starting"
  | "extracting"
  | "aggregating"
  | "generating-actions"
  | "done"
  | "error";

const STAGE_LABEL: Record<Stage, string> = {
  starting: "Khởi tạo job…",
  extracting: "Trích xuất aspect",
  aggregating: "Đang gộp summary",
  "generating-actions": "Sinh action bằng Gemini",
  done: "Hoàn tất",
  error: "Lỗi",
};

export function StreamProgress({
  stage,
  done,
  total,
  recent,
  elapsedMs,
  source,
  csvFilename,
}: {
  stage: Stage;
  done: number;
  total: number;
  recent: { review: string; tuples: PredictedTuple[] } | null;
  elapsedMs: number | null;
  source?: "paste" | "csv" | "sample";
  csvFilename?: string;
}) {
  // Soft elapsed counter while running.
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (stage === "done" || stage === "error") return;
    const id = setInterval(() => setTick((t) => t + 1), 200);
    return () => clearInterval(id);
  }, [stage]);
  void tick;

  const pct = total > 0 ? (100 * done) / total : 0;
  const isLive = stage !== "done" && stage !== "error";
  const tooltipTotal = `${done}/${total} reviews`;

  return (
    <div className="panel relative overflow-hidden p-4">
      {/* shimmer accent line while live */}
      {isLive && (
        <span
          aria-hidden
          className="absolute left-0 top-0 h-[2px] bg-gradient-to-r from-brand-500 via-pink-500 to-brand-500 animate-pulse"
          style={{ width: `${Math.max(pct, 4)}%` }}
        />
      )}
      <div className="flex flex-wrap items-center gap-2 mb-2.5">
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.16em] border ${
            stage === "error"
              ? "border-neg/30 bg-neg-soft text-neg"
              : stage === "done"
              ? "border-pos/30 bg-pos-soft text-pos"
              : "border-brand-500/30 bg-brand-50 text-brand-700"
          }`}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              stage === "error"
                ? "bg-neg"
                : stage === "done"
                ? "bg-pos"
                : "bg-brand-500 animate-pulse"
            }`}
          />
          {STAGE_LABEL[stage]}
        </span>
        <span className="text-[11px] text-fg-muted tabular-nums" title={tooltipTotal}>
          {done} / {total} reviews
        </span>
        {source && (
          <span className="text-[10.5px] text-fg-faint">
            source: {source === "csv" ? csvFilename ?? "csv" : source}
          </span>
        )}
        <span className="text-[10.5px] text-fg-faint tabular-nums ml-auto">
          {elapsedMs != null
            ? `${(elapsedMs / 1000).toFixed(1)}s`
            : `${((performance.now() % 1e8) / 1000).toFixed(0)}s …`}
        </span>
      </div>

      <div className="h-1.5 rounded-full bg-bg-subtle overflow-hidden">
        <div
          className={`h-full transition-all duration-200 ${
            stage === "error"
              ? "bg-neg"
              : "bg-gradient-to-r from-brand-500 to-pink-500"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {recent && (
        <div className="mt-3 text-[12px] flex items-start gap-2">
          <span className="shrink-0 mt-0.5 inline-flex w-4 h-4 items-center justify-center rounded-full bg-pos-soft text-pos text-[10px] font-bold">
            ✓
          </span>
          <div className="min-w-0 flex-1">
            <div className="text-fg-muted truncate" title={recent.review}>
              {recent.review || "(empty)"}
            </div>
            {recent.tuples.length === 0 ? (
              <div className="text-fg-dim text-[10.5px] italic">
                không trích xuất được aspect nào
              </div>
            ) : (
              <div className="flex flex-wrap gap-1 mt-1">
                {recent.tuples.slice(0, 6).map((t, i) => (
                  <span
                    key={i}
                    className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0 text-[10.5px] border ${
                      t.sentiment === "negative"
                        ? "border-neg/30 bg-neg-soft/60 text-neg"
                        : "border-pos/30 bg-pos-soft/60 text-pos"
                    }`}
                  >
                    {ASPECT_LABEL_VI[t.aspect_category]} · {t.aspect_term}
                  </span>
                ))}
                {recent.tuples.length > 6 && (
                  <span className="text-[10.5px] text-fg-faint self-center">
                    +{recent.tuples.length - 6}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
