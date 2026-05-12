"use client";

import { ASPECT_LABEL_VI, PRIORITY_RANK } from "@/lib/labels";
import type { ActionRecommendation } from "@/lib/api";

const STYLE = {
  high: {
    accent: "bg-neg",
    text: "text-neg",
    ring: "ring-neg/20",
    bg: "bg-neg-soft/40",
  },
  medium: {
    accent: "bg-warn",
    text: "text-warn",
    ring: "ring-warn/20",
    bg: "bg-warn-soft/40",
  },
  low: {
    accent: "bg-fg-dim",
    text: "text-fg-muted",
    ring: "ring-line",
    bg: "bg-white",
  },
} as const;

export function ActionCards({
  actions,
  splitBySentiment = false,
}: {
  actions: ActionRecommendation[];
  /** When true, render two columns: complaints (left) and praises (right). */
  splitBySentiment?: boolean;
}) {
  if (actions.length === 0) {
    return (
      <div className="panel-pad text-[12px] text-fg-faint italic">
        Không có hành động đề xuất cho batch này.
      </div>
    );
  }

  if (!splitBySentiment) {
    return (
      <ul className="space-y-1.5">
        {sortByPriority(actions).map((a, i) => (
          <Card key={`${a.aspect_category}-${a.sentiment}-${i}`} action={a} />
        ))}
      </ul>
    );
  }

  const neg = sortByPriority(actions.filter((a) => a.sentiment === "negative"));
  const pos = sortByPriority(actions.filter((a) => a.sentiment === "positive"));

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      <Column
        title="Khắc phục"
        subtitle="negative cells"
        accent="text-neg"
        count={neg.length}
        items={neg}
      />
      <Column
        title="Khuếch trương"
        subtitle="positive cells"
        accent="text-pos"
        count={pos.length}
        items={pos}
      />
    </div>
  );
}

function sortByPriority(arr: ActionRecommendation[]): ActionRecommendation[] {
  return [...arr].sort(
    (a, b) => (PRIORITY_RANK[a.priority] ?? 99) - (PRIORITY_RANK[b.priority] ?? 99)
  );
}

function Column({
  title,
  subtitle,
  accent,
  count,
  items,
}: {
  title: string;
  subtitle: string;
  accent: string;
  count: number;
  items: ActionRecommendation[];
}) {
  return (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between mb-1.5 px-0.5">
        <div className={`text-[11px] font-semibold tracking-[0.04em] uppercase ${accent}`}>
          {title}
        </div>
        <div className="text-[9.5px] text-fg-faint tabular-nums uppercase tracking-wider">
          {subtitle} · {count}
        </div>
      </div>
      {items.length === 0 ? (
        <div className="panel p-2.5 text-[11px] text-fg-faint italic">
          (không có)
        </div>
      ) : (
        <ul className="space-y-1.5">
          {items.map((a, i) => (
            <Card
              key={`${a.aspect_category}-${a.sentiment}-${i}`}
              action={a}
              hideSentimentPill
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function Card({
  action: a,
  hideSentimentPill = false,
}: {
  action: ActionRecommendation;
  hideSentimentPill?: boolean;
}) {
  const s = STYLE[a.priority] ?? STYLE.low;
  const isNeg = a.sentiment === "negative";
  return (
    <li
      className={`panel ring-1 ${s.ring} ${s.bg} relative pl-3 pr-3 py-2.5`}
    >
      <span
        className={`absolute left-0 top-2 bottom-2 w-[2px] rounded-r ${s.accent}`}
        aria-hidden
      />
      <div className="flex items-center gap-1.5 mb-1">
        <span
          className={`text-[9.5px] font-bold uppercase tracking-[0.14em] ${s.text}`}
        >
          {a.priority}
        </span>
        <span className="text-fg-dim">·</span>
        <span className="text-[10.5px] text-fg-muted">
          {ASPECT_LABEL_VI[a.aspect_category]}
        </span>
        {!hideSentimentPill && (
          <span className={isNeg ? "pill-neg ml-auto" : "pill-pos ml-auto"}>
            {isNeg ? "neg" : "pos"}
          </span>
        )}
      </div>
      <p className="text-[13px] leading-snug text-fg font-medium">{a.action}</p>
      {a.evidence_terms.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {a.evidence_terms.slice(0, 3).map((t) => (
            <span key={t} className="chip">
              {t}
            </span>
          ))}
        </div>
      )}
    </li>
  );
}
