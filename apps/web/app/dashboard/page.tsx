"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";

import { ActionCards } from "@/components/ActionCards";
import { AspectBreakdown } from "@/components/AspectBreakdown";
import { AspectRadar } from "@/components/AspectRadar";
import { KpiStrip } from "@/components/KpiStrip";
import { PerAspectBars } from "@/components/PerAspectBars";
import { PriorityDistribution } from "@/components/PriorityDistribution";
import { ReviewsTable, type ReviewRow } from "@/components/ReviewsTable";
import { SentimentDonut } from "@/components/SentimentDonut";
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

type TabId = "overview" | "actions" | "aspects" | "reviews";

const TABS: Array<{ id: TabId; label: string; sub: string }> = [
  { id: "overview", label: "Tổng quan",  sub: "headline + KPI" },
  { id: "actions",  label: "Hành động",  sub: "khắc phục · khuếch trương" },
  { id: "aspects",  label: "Aspects",    sub: "radar · breakdown · charts" },
  { id: "reviews",  label: "Reviews",    sub: "search + filter" },
];

export default function DashboardPage() {
  // Result + streaming state.
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [missing, setMissing] = useState(false);

  // Streaming progress state.
  const [stage, setStage] = useState<Stage | null>(null);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [partialSummary, setPartialSummary] = useState<CorpusSummary | null>(null);
  const [actions, setActions] = useState<ActionRecommendation[]>([]);
  const [reviewRows, setReviewRows] = useState<ReviewRow[]>([]);
  const [recent, setRecent] = useState<{
    review: string;
    tuples: PredictedTuple[];
  } | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [streamErr, setStreamErr] = useState<string | null>(null);
  const [job, setJob] = useState<PendingJob | null>(null);
  const startedRef = useRef(false);

  // Active tab
  const [tab, setTab] = useState<TabId>("overview");

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    const pending = takePendingJob();
    if (pending) {
      setJob(pending);
      setStage("starting");
      setProgress({ done: 0, total: pending.reviews.length });
      setReviewRows([]);
      runStream(pending);
      return;
    }
    const stored = loadResult();
    if (stored) {
      setResult(stored);
      setReviewRows(
        stored.per_review.map((r) => ({
          id: r.id,
          review: r.review,
          tuples: r.tuples,
        }))
      );
    } else {
      setMissing(true);
    }
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
              setReviewRows((prev) => [
                ...prev,
                { id: d.id, review: d.review, tuples: d.tuples },
              ]);
              break;
            }
            case "summary": {
              const d = data as { summary: CorpusSummary };
              setPartialSummary(d.summary);
              finalSummary = d.summary;
              break;
            }
            case "actions_pending":
              setStage(pending.useLlm ? "generating-actions" : "aggregating");
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

  // ── HOOKS MUST RUN BEFORE ANY EARLY RETURN ──
  // displayActions wraps a sort, so it stays as useMemo. Anything that uses
  // state/props but not a hook can stay below.
  const displayActions = useMemo<ActionRecommendation[]>(() => {
    const src = actions.length > 0 ? actions : result?.actions ?? [];
    return [...src].sort(
      (a, b) => (PRIORITY_RANK[a.priority] ?? 99) - (PRIORITY_RANK[b.priority] ?? 99)
    );
  }, [actions, result]);

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

  const topAction = displayActions[0] ?? null;
  const topNegAction =
    displayActions.find((a) => a.sentiment === "negative") ?? null;
  const heroAction = topNegAction ?? topAction;

  // Tab badges (counts)
  const tabBadge: Record<TabId, string | number | null> = {
    overview: null,
    actions: displayActions.length || null,
    aspects: Object.keys(summary.aspects).length || null,
    reviews: progress.total
      ? `${reviewRows.length}/${progress.total}`
      : reviewRows.length || null,
  };

  return (
    <div className="space-y-5">
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
              {elapsedMs != null && !live && (
                <> · {(elapsedMs / 1000).toFixed(1)}s</>
              )}
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

      {/* ============ STREAM PROGRESS — always above tabs ============ */}
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

      {/* ============ TAB BAR ============ */}
      <nav className="border-b border-line">
        <ul className="flex flex-wrap gap-0">
          {TABS.map((t) => {
            const active = tab === t.id;
            const badge = tabBadge[t.id];
            return (
              <li key={t.id}>
                <button
                  type="button"
                  onClick={() => setTab(t.id)}
                  className={`relative px-4 py-2.5 text-[13px] font-medium transition border-b-2 ${
                    active
                      ? "border-brand-500 text-fg"
                      : "border-transparent text-fg-muted hover:text-fg hover:bg-bg-subtle"
                  }`}
                >
                  <span>{t.label}</span>
                  {badge != null && (
                    <span
                      className={`ml-1.5 inline-flex items-center justify-center rounded-full px-1.5 text-[10px] tabular-nums font-semibold ${
                        active
                          ? "bg-brand-500 text-white"
                          : "bg-bg-subtle text-fg-muted"
                      }`}
                    >
                      {badge}
                    </span>
                  )}
                  {active && (
                    <span className="block text-[9.5px] uppercase tracking-[0.14em] text-fg-faint mt-0.5">
                      {t.sub}
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* ============ TAB CONTENT ============ */}
      {tab === "overview" && (
        <section className="space-y-5">
          {heroAction ? (
            <div>
              <div className="flex items-baseline justify-between mb-2">
                <h2 className="text-[12px] font-semibold tracking-[0.04em] text-fg uppercase">
                  {live ? "Action sẽ xuất hiện ở đây" : "Việc cần làm trước nhất"}
                </h2>
                <span className="text-[10.5px] text-fg-faint">
                  ưu tiên theo negative_ratio × count
                </span>
              </div>
              <TopPriorityHero action={heroAction} />
            </div>
          ) : live ? (
            <div className="panel p-5 text-[13px] text-fg-faint italic flex items-center gap-2">
              <span className="inline-block w-3 h-3 border-2 border-brand-500/70 border-t-transparent rounded-full animate-spin" />
              Đang xử lý {progress.done} / {progress.total} reviews. Hero action sẽ
              ghim lên đây khi Gemini trả về.
            </div>
          ) : null}

          <KpiStrip
            summary={summary}
            reviewsProcessed={progress.done || (result?.n_reviews ?? 0)}
            reviewsTotal={progress.total || (result?.n_reviews ?? 0)}
            live={live}
          />

          <div className="grid lg:grid-cols-2 gap-5">
            <SentimentDonut summary={summary} />
            <div className="panel-pad">
              <div className="flex items-baseline justify-between mb-2">
                <h3 className="text-[12px] font-semibold tracking-[0.04em] uppercase text-fg">
                  Snapshot
                </h3>
                <span className="text-[10.5px] text-fg-faint">
                  điểm nóng nhất
                </span>
              </div>
              <Snapshot
                summary={summary}
                topNegAction={topNegAction}
                topPosAction={
                  displayActions.find((a) => a.sentiment === "positive") ?? null
                }
                onJump={(id) => setTab(id)}
              />
            </div>
          </div>
        </section>
      )}

      {tab === "actions" && (
        <section className="space-y-5">
          <PriorityDistribution actions={displayActions} />
          {displayActions.length === 0 && live ? (
            <div className="panel-pad text-[12px] text-fg-faint italic flex items-center gap-2">
              <span className="inline-block w-3 h-3 border-2 border-brand-500/70 border-t-transparent rounded-full animate-spin" />
              {stage === "generating-actions"
                ? "Gemini đang sinh action…"
                : "Action sẽ xuất hiện sau khi extract xong."}
            </div>
          ) : (
            <ActionCards actions={displayActions} splitBySentiment />
          )}
        </section>
      )}

      {tab === "aspects" && (
        <section className="space-y-5">
          <div className="grid lg:grid-cols-2 gap-5">
            <div className="panel-pad">
              <div className="flex items-baseline justify-between mb-2">
                <h3 className="text-[12px] font-semibold tracking-[0.04em] uppercase text-fg">
                  Aspect coverage
                </h3>
                <span className="text-[10.5px] text-fg-faint">
                  volume × sentiment{live ? " · real-time" : ""}
                </span>
              </div>
              <AspectRadar summary={summary} />
            </div>
            <PerAspectBars summary={summary} />
          </div>

          <div>
            <div className="flex items-baseline justify-between mb-2">
              <h3 className="text-[12px] font-semibold tracking-[0.04em] text-fg uppercase">
                Aspect breakdown
              </h3>
              <span className="text-[10.5px] text-fg-faint">
                ring đỏ = hotspot ({">"}50% neg) · click card để xem reviews tham chiếu
              </span>
            </div>
            <AspectBreakdown summary={summary} reviews={reviewRows} />
          </div>
        </section>
      )}

      {tab === "reviews" && (
        <section>
          {reviewRows.length === 0 && live ? (
            <div className="panel-pad text-[12px] text-fg-faint italic flex items-center gap-2">
              <span className="inline-block w-3 h-3 border-2 border-brand-500/70 border-t-transparent rounded-full animate-spin" />
              Streaming review… ({progress.done}/{progress.total})
            </div>
          ) : (
            <ReviewsTable rows={reviewRows} expectedTotal={progress.total} />
          )}
        </section>
      )}
    </div>
  );
}

/** Tiny callout card under SentimentDonut in Overview tab. */
function Snapshot({
  summary,
  topNegAction,
  topPosAction,
  onJump,
}: {
  summary: CorpusSummary;
  topNegAction: ActionRecommendation | null;
  topPosAction: ActionRecommendation | null;
  onJump: (id: TabId) => void;
}) {
  const aspects = Object.values(summary.aspects);
  const hotspot = aspects
    .filter((a) => !!a)
    .sort((a, b) => b!.negative * b!.negative_ratio - a!.negative * a!.negative_ratio)[0];

  return (
    <div className="space-y-3 text-[12.5px]">
      {hotspot ? (
        <div>
          <div className="text-[10px] uppercase tracking-[0.14em] text-fg-faint mb-0.5">
            Hotspot
          </div>
          <div className="text-fg">
            <span className="font-semibold text-neg">
              {hotspot.aspect_category}
            </span>{" "}
            — {hotspot.negative} complaint
            {hotspot.negative === 1 ? "" : "s"} /{" "}
            {(hotspot.negative_ratio * 100).toFixed(0)}% negative
          </div>
        </div>
      ) : (
        <div className="text-fg-faint italic">Chưa có hotspot.</div>
      )}

      {topNegAction && (
        <div>
          <div className="text-[10px] uppercase tracking-[0.14em] text-fg-faint mb-0.5">
            Cần khắc phục
          </div>
          <div className="text-fg line-clamp-2">{topNegAction.action}</div>
        </div>
      )}
      {topPosAction && (
        <div>
          <div className="text-[10px] uppercase tracking-[0.14em] text-fg-faint mb-0.5">
            Nên duy trì
          </div>
          <div className="text-fg line-clamp-2">{topPosAction.action}</div>
        </div>
      )}

      <button
        type="button"
        onClick={() => onJump("actions")}
        className="text-[11px] text-brand-600 hover:underline font-medium"
      >
        Xem toàn bộ action queue →
      </button>
    </div>
  );
}
