"use client";

import type { ActionRecommendation } from "@/lib/api";

const PRIORITY_COLOR = {
  high: "#e11d48",   // neg
  medium: "#d97706", // warn
  low: "#a1a1aa",    // muted
} as const;

const PRIORITY_LABEL_VI = {
  high: "Cao",
  medium: "Trung bình",
  low: "Thấp",
} as const;

/**
 * Stacked horizontal bar of priority composition per sentiment.
 * One row per sentiment (neg / pos), each split into high/medium/low segments.
 */
export function PriorityDistribution({ actions }: { actions: ActionRecommendation[] }) {
  if (actions.length === 0) {
    return (
      <div className="panel-pad text-[12px] text-fg-faint italic">
        Chưa có action để phân bố priority.
      </div>
    );
  }
  const buckets = {
    negative: { high: 0, medium: 0, low: 0 },
    positive: { high: 0, medium: 0, low: 0 },
  };
  for (const a of actions) {
    buckets[a.sentiment][a.priority] += 1;
  }
  const totalNeg =
    buckets.negative.high + buckets.negative.medium + buckets.negative.low;
  const totalPos =
    buckets.positive.high + buckets.positive.medium + buckets.positive.low;

  return (
    <div className="panel-pad">
      <div className="flex items-baseline justify-between mb-2">
        <h3 className="text-[12px] font-semibold tracking-[0.04em] uppercase text-fg">
          Phân bố priority
        </h3>
        <div className="flex items-center gap-3 text-[10px]">
          {(["high", "medium", "low"] as const).map((p) => (
            <span key={p} className="inline-flex items-center gap-1">
              <span
                className="w-2 h-2 rounded-sm"
                style={{ background: PRIORITY_COLOR[p] }}
              />
              <span className="text-fg-muted">{PRIORITY_LABEL_VI[p]}</span>
            </span>
          ))}
        </div>
      </div>

      <div className="space-y-3">
        <Row
          label="Khắc phục"
          labelColor="text-neg"
          total={totalNeg}
          counts={buckets.negative}
        />
        <Row
          label="Khuếch trương"
          labelColor="text-pos"
          total={totalPos}
          counts={buckets.positive}
        />
      </div>
    </div>
  );
}

function Row({
  label,
  labelColor,
  total,
  counts,
}: {
  label: string;
  labelColor: string;
  total: number;
  counts: { high: number; medium: number; low: number };
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <span className={`text-[11px] font-semibold uppercase tracking-[0.04em] ${labelColor}`}>
          {label}
        </span>
        <span className="text-[10.5px] tabular-nums text-fg-faint">
          {total} action{total === 1 ? "" : "s"}
        </span>
      </div>
      {total === 0 ? (
        <div className="h-3 rounded-full bg-bg-subtle" />
      ) : (
        <div className="flex h-3 rounded-full overflow-hidden bg-bg-subtle">
          {(["high", "medium", "low"] as const).map((p) => {
            const n = counts[p];
            if (n === 0) return null;
            const pct = (100 * n) / total;
            return (
              <div
                key={p}
                className="h-full relative group"
                style={{ width: `${pct}%`, background: PRIORITY_COLOR[p] }}
                title={`${PRIORITY_LABEL_VI[p]}: ${n} (${pct.toFixed(0)}%)`}
              >
                {pct >= 12 && (
                  <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-white tabular-nums">
                    {n}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
