"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";

import { ActionCards } from "@/components/ActionCards";
import { AspectBreakdown } from "@/components/AspectBreakdown";
import { AspectRadar } from "@/components/AspectRadar";
import { StreamProgress, type Stage } from "@/components/StreamProgress";
import { TopPriorityHero } from "@/components/TopPriorityHero";
import {
  analyzeStream,
  type ActionRecommendation,
  type AnalyzeResult,
  type CorpusSummary,
  type PredictedTuple,
} from "@/lib/api";
import { PRIORITY_RANK } from "@/lib/labels";
import {
  loadResult,
  persistResult,
  takePendingJob,
  type PendingJob,
} from "@/lib/resultStorage";

const EMPTY_SUMMARY: CorpusSummary = {
  n_reviews: 0,
  n_reviews_with_tuples: 0,
  n_tuples: 0,
  sentiment_overall: {},
  aspects: {},
};

export default function DashboardPage() {
  // Result + streaming state.
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [missing, setMissing] = useState(false);

  // Streaming progress state.
  const [stage, setStage] = useState<Stage | null>(null);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [partialSummary, setPartialSummary] = useState<CorpusSummary | null>(null);
  const [actions, setActions] = useState<ActionRecommendation[]>([]);
  const [recent, setRecent] = useState<{
    review: string;
    tuples: PredictedTuple[];
  } | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [streamErr, setStreamErr] = useState<string | null>(null);
  const [job, setJob] = useState<PendingJob | null>(null);
  const startedRef = useRef(false);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    const pending = takePendingJob();
    if (pending) {
      setJob(pending);
      setStage("starting");
      setProgress({ done: 0, total: pending.reviews.length });
      runStream(pending);
      return;
    }
    const stored = loadResult();
    if (stored) setResult(stored);
    else setMissing(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function runStream(pending: PendingJob) {
    const perReview: { id: string; review: string; tuples: PredictedTuple[] }[] = [];
    let finalSummary: CorpusSummary = EMPTY_SUMMARY;
    let finalActions: ActionRecommendation[] = [];
    try {
      await analyzeStream(
        pending.reviews,
        { useLlm: pending.useLlm },
        (evt, data) => {
          switch (evt) {
            case "start":
              setStage("extracting");
              break;
            case "review": {
              const d = data as {
                idx: number;
                id: string;
                review: string;
                tuples: PredictedTuple[];
              };
              perReview.push({ id: d.id, review: d.review, tuples: d.tuples });
              setProgress((p) => ({ ...p, done: d.idx + 1 }));
              setRecent({ review: d.review, tuples: d.tuples });
              break;
            }
            case "summary": {
              const d = data as { summary: CorpusSummary };
              setPartialSummary(d.summary);
              finalSummary = d.summary;
              break;
            }
            case "actions_pending":
              setStage(
                pending.useLlm ? "generating-actions" : "aggregating"
              );
              break;
            case "actions": {
              const d = data as { actions: ActionRecommendation[] };
              setActions(d.actions);
              finalActions = d.actions;
              break;
            }
            case "done": {
              const d = data as { elapsed_ms: number };
              setElapsedMs(d.elapsed_ms);
              setStage("done");
              const final: AnalyzeResult = {
                n_reviews: pending.reviews.length,
                summary: finalSummary,
                actions: finalActions,
                per_review: perReview,
              };
              persistResult(final);
              setResult(final);
              break;
            }
            case "error": {
              const d = data as { message?: string };
              setStreamErr(d.message || "stream error");
              setStage("error");
              break;
            }
          }
        }
      );
    } catch (e) {
      const msg = (e as Error).message || "";
      setStreamErr(
        /NetworkError|Failed to fetch|fetch failed/i.test(msg)
          ? "Mất kết nối tới backend (http://localhost:8000)."
          : msg
      );
      setStage("error");
    }
  }

  if (missing) {
    return (
      <div className="panel-pad max-w-md">
        <h2 className="text-base font-semibold mb-1">Chưa có kết quả</h2>
        <p className="text-[12.5px] text-fg-muted mb-3">
          Bạn cần phân tích ít nhất 1 batch trước khi xem dashboard.
        </p>
        <Link href="/" className="btn-primary inline-flex">
          ← Quay lại phân tích
        </Link>
      </div>
    );
  }

  // Pick the best available summary (live partial > final result).
  const summary: CorpusSummary =
    partialSummary ?? result?.summary ?? EMPTY_SUMMARY;
  const live = stage != null && stage !== "done" && stage !== "error";

  const negCount = summary.sentiment_overall.negative ?? 0;
  const posCount = summary.sentiment_overall.positive ?? 0;
  const total = negCount + posCount;
  const negPct = total ? (100 * negCount) / total : 0;

  const sortedActions: ActionRecommendation[] = [...actions].sort(
    (a, b) =>
      (PRIORITY_RANK[a.priority] ?? 99) - (PRIORITY_RANK[b.priority] ?? 99)
  );
  // fallback to result.actions when not streaming
  const displayActions =
    sortedActions.length > 0
      ? sortedActions
      : result
      ? [...result.actions].sort(
          (a, b) =>
            (PRIORITY_RANK[a.priority] ?? 99) -
            (PRIORITY_RANK[b.priority] ?? 99)
        )
      : [];
  const topAction = displayActions[0] ?? null;
  const topNegAction =
    displayActions.find((a) => a.sentiment === "negative") ?? null;
  const heroAction = topNegAction ?? topAction;

  return (
    <div className="space-y-6">
      {/* ============ TITLE BAR ============ */}
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.18em] text-fg-faint font-semibold mb-1.5">
            Analysis report{live ? " · live" : ""}
          </div>
          <div className="flex items-baseline gap-3">
            <h1 className="text-[24px] font-bold tracking-tight">Dashboard</h1>
            <span className="text-[11px] text-fg-faint tabular-nums">
              {progress.total
                ? `${progress.done} / ${progress.total} reviews`
                : `${result?.n_reviews ?? 0} reviews`}{" "}
              · {summary.n_tuples} tuples
            </span>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <Link href="/" className="btn-ghost">
            ← Batch mới
          </Link>
          <button
            onClick={() => window.print()}
            className="btn-outline"
            disabled={live}
            title={live ? "đợi stream kết thúc" : "in / export PDF"}
          >
            Export
          </button>
        </div>
      </header>

      {/* ============ STREAM PROGRESS (only while streaming or just-finished) ============ */}
      {stage !== null && (
        <StreamProgress
          stage={streamErr ? "error" : stage}
          done={progress.done}
          total={progress.total}
          recent={recent}
          elapsedMs={elapsedMs}
          source={job?.source}
          csvFilename={job?.csvFilename}
        />
      )}
      {streamErr && (
        <div className="rounded-md border border-neg/30 bg-neg-soft p-3 text-[12px] text-neg">
          <span className="font-semibold">Lỗi stream:</span> {streamErr}
        </div>
      )}

      {/* ============ HERO PRIORITY ============ */}
      {heroAction ? (
        <section>
          <div className="flex items-baseline justify-between mb-3">
            <h2 className="text-[12px] font-semibold tracking-[0.04em] text-fg uppercase">
              {live ? "Action sẽ xuất hiện ở đây" : "Việc cần làm trước nhất"}
            </h2>
            <span className="text-[10.5px] text-fg-faint">
              ưu tiên theo negative_ratio × count
            </span>
          </div>
          <TopPriorityHero action={heroAction} />
        </section>
      ) : live ? (
        <section>
          <div className="flex items-baseline justify-between mb-3">
            <h2 className="text-[12px] font-semibold tracking-[0.04em] text-fg uppercase">
              Hero action
            </h2>
            <span className="text-[10.5px] text-fg-faint">
              {stage === "generating-actions"
                ? "Gemini đang viết action…"
                : "chưa có action — chờ extract xong"}
            </span>
          </div>
          <div className="panel p-5 text-[13px] text-fg-faint italic flex items-center gap-2">
            <span className="inline-block w-3 h-3 border-2 border-brand-500/70 border-t-transparent rounded-full animate-spin" />
            Đang xử lý {progress.done} / {progress.total} reviews. Hero action
            sẽ ghim lên đây ngay khi Gemini trả về.
          </div>
        </section>
      ) : null}

      {/* ============ KPI STRIP (live tick) ============ */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="Reviews phân tích" value={progress.done || (result?.n_reviews ?? 0)} pulsing={live} />
        <Stat
          label="Có aspect"
          value={summary.n_reviews_with_tuples}
          sub={
            progress.done
              ? `${Math.round(
                  (100 * summary.n_reviews_with_tuples) /
                    Math.max(1, progress.done)
                )}% coverage`
              : undefined
          }
          pulsing={live}
        />
        <Stat label="Aspect tuples" value={summary.n_tuples} pulsing={live} />
        <Stat
          label="Tỷ lệ tiêu cực"
          value={total ? `${negPct.toFixed(0)}%` : "—"}
          tone={negPct > 50 ? "neg" : negPct > 30 ? "warn" : "pos"}
          sub={total ? `${negCount} / ${total} tuples` : undefined}
          pulsing={live}
        />
      </section>

      {/* ============ MAIN GRID ============ */}
      <section className="grid lg:grid-cols-[1.55fr_1fr] gap-6 items-start">
        <div className="space-y-6 min-w-0">
          <div className="panel-pad">
            <div className="flex items-baseline justify-between mb-3">
              <h3 className="text-[12px] font-semibold tracking-[0.04em] text-fg uppercase">
                Aspect coverage
              </h3>
              <span className="text-[10.5px] text-fg-faint">
                volume × sentiment per aspect{live ? " · cập nhật real-time" : ""}
              </span>
            </div>
            <AspectRadar summary={summary} />
          </div>

          <div>
            <div className="flex items-baseline justify-between mb-3">
              <h3 className="text-[12px] font-semibold tracking-[0.04em] text-fg uppercase">
                Aspect breakdown
              </h3>
              <span className="text-[10.5px] text-fg-faint">
                sắp xếp theo priority · ring đỏ = hotspot ({">"}50% neg)
              </span>
            </div>
            <AspectBreakdown summary={summary} />
          </div>
        </div>

        <aside className="lg:sticky lg:top-[60px] space-y-3 min-w-0">
          <div className="flex items-baseline justify-between">
            <h3 className="text-[12px] font-semibold tracking-[0.04em] text-fg uppercase">
              Action queue
            </h3>
            <span className="text-[10.5px] text-fg-faint">
              {live && displayActions.length === 0
                ? stage === "generating-actions"
                  ? "đang gen…"
                  : "chờ extract xong"
                : `${displayActions.length} items`}
            </span>
          </div>
          <div className="max-h-[calc(100vh-7rem)] overflow-y-auto pr-1 -mr-1">
            {displayActions.length === 0 && live ? (
              <div className="panel-pad text-[12px] text-fg-faint italic flex items-center gap-2">
                <span className="inline-block w-3 h-3 border-2 border-brand-500/70 border-t-transparent rounded-full animate-spin" />
                {stage === "generating-actions"
                  ? "Gemini đang sinh action…"
                  : "Action sẽ xuất hiện sau khi extract xong."}
              </div>
            ) : (
              <ActionCards actions={displayActions} />
            )}
          </div>
        </aside>
      </section>
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
  tone,
  pulsing,
}: {
  label: string;
  value: string | number;
  sub?: string;
  tone?: "neg" | "warn" | "pos";
  pulsing?: boolean;
}) {
  const valueColor =
    tone === "neg"
      ? "text-neg"
      : tone === "warn"
      ? "text-warn"
      : tone === "pos"
      ? "text-pos"
      : "text-fg";
  return (
    <div className={`panel p-3.5 ${pulsing ? "ring-1 ring-brand-500/15" : ""}`}>
      <div className="stat-label flex items-center gap-1.5">
        {label}
        {pulsing && (
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-brand-500 animate-pulse" />
        )}
      </div>
      <div
        className={`mt-1.5 text-[24px] font-bold tracking-tight tabular-nums leading-none ${valueColor}`}
      >
        {value}
      </div>
      {sub && (
        <div className="mt-1 text-[10.5px] text-fg-faint tabular-nums">{sub}</div>
      )}
    </div>
  );
}
