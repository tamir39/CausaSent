"use client";

import type { CorpusSummary } from "@/lib/api";

/**
 * Dense horizontal stat strip — replaces 4 stacked p-3.5 cards.
 * Each cell shows: caption, value, and a contextual mini-bar / sub line.
 * Color emphasis (text-pos / text-neg / text-warn) carries the "highlight"
 * instead of larger type or padding.
 */
export function KpiStrip({
  summary,
  reviewsProcessed,
  reviewsTotal,
  live,
}: {
  summary: CorpusSummary;
  reviewsProcessed: number;
  reviewsTotal: number;
  live: boolean;
}) {
  const pos = summary.sentiment_overall.positive ?? 0;
  const neg = summary.sentiment_overall.negative ?? 0;
  const total = pos + neg;
  const negPct = total ? (100 * neg) / total : 0;
  const coverageBase = reviewsTotal || reviewsProcessed || 1;
  const coveragePct = (100 * summary.n_reviews_with_tuples) / coverageBase;
  const progPct = reviewsTotal ? (100 * reviewsProcessed) / reviewsTotal : 100;

  const tone =
    negPct > 50 ? "neg" : negPct > 30 ? "warn" : negPct > 0 ? "pos" : "neutral";

  return (
    <div className="panel flex flex-col sm:flex-row divide-y sm:divide-y-0 sm:divide-x divide-line/80 overflow-hidden">
      <Cell
        caption="Reviews"
        value={
          reviewsTotal
            ? `${reviewsProcessed}/${reviewsTotal}`
            : `${reviewsProcessed}`
        }
        bar={progPct}
        barColor="bg-brand-500"
        pulsing={live && reviewsProcessed < reviewsTotal}
      />
      <Cell
        caption="Có aspect"
        value={`${summary.n_reviews_with_tuples}`}
        sub={`${coveragePct.toFixed(0)}% coverage`}
        bar={coveragePct}
        barColor="bg-emerald-500"
        pulsing={live}
      />
      <Cell
        caption="Aspect tuples"
        value={`${summary.n_tuples}`}
        sub={
          summary.n_tuples > 0 && reviewsProcessed > 0
            ? `~${(summary.n_tuples / reviewsProcessed).toFixed(1)} / review`
            : undefined
        }
        pulsing={live}
      />
      <Cell
        caption="Tỷ lệ tiêu cực"
        value={total ? `${negPct.toFixed(0)}%` : "—"}
        sub={total ? `${neg} neg · ${pos} pos` : "chưa có tuple"}
        bar={negPct}
        barColor={
          tone === "neg"
            ? "bg-neg"
            : tone === "warn"
            ? "bg-warn"
            : tone === "pos"
            ? "bg-pos"
            : "bg-fg-dim"
        }
        valueClass={
          tone === "neg"
            ? "text-neg"
            : tone === "warn"
            ? "text-warn"
            : tone === "pos"
            ? "text-pos"
            : ""
        }
        pulsing={live}
      />
    </div>
  );
}

function Cell({
  caption,
  value,
  sub,
  bar,
  barColor = "bg-brand-500",
  valueClass = "",
  pulsing,
}: {
  caption: string;
  value: string | number;
  sub?: string;
  bar?: number;
  barColor?: string;
  valueClass?: string;
  pulsing?: boolean;
}) {
  return (
    <div className="flex-1 px-4 py-3 min-w-0">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[10px] uppercase tracking-[0.16em] text-fg-faint font-semibold flex items-center gap-1.5">
          {caption}
          {pulsing && (
            <span className="inline-block w-1 h-1 rounded-full bg-brand-500 animate-pulse" />
          )}
        </span>
        <span
          className={`text-[18px] font-bold tabular-nums leading-none ${valueClass || "text-fg"}`}
        >
          {value}
        </span>
      </div>
      <div className="mt-2 flex items-center gap-2">
        {bar !== undefined && (
          <div className="flex-1 h-[3px] rounded-full bg-bg-subtle overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${barColor}`}
              style={{ width: `${Math.min(100, Math.max(0, bar))}%` }}
            />
          </div>
        )}
        {sub && (
          <span className="text-[10px] text-fg-faint tabular-nums whitespace-nowrap">
            {sub}
          </span>
        )}
      </div>
    </div>
  );
}
