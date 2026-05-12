"use client";

import type { ActionRecommendation } from "@/lib/api";
import { ASPECT_LABEL_VI } from "@/lib/labels";

const PRIORITY_STYLE = {
  high: {
    accent: "bg-neg",
    glow: "from-neg-soft via-neg-soft/30 to-transparent",
    text: "text-neg",
    ring: "ring-neg/20",
    badge: "bg-neg-soft text-neg border-neg/30",
    label: "Ưu tiên cao",
  },
  medium: {
    accent: "bg-warn",
    glow: "from-warn-soft via-warn-soft/30 to-transparent",
    text: "text-warn",
    ring: "ring-warn/20",
    badge: "bg-warn-soft text-warn border-warn/30",
    label: "Ưu tiên trung bình",
  },
  low: {
    accent: "bg-fg-dim",
    glow: "from-bg-subtle via-bg-subtle/60 to-transparent",
    text: "text-fg-muted",
    ring: "ring-line",
    badge: "bg-bg-subtle text-fg-muted border-line",
    label: "Ưu tiên thấp",
  },
} as const;

export function TopPriorityHero({ action }: { action: ActionRecommendation }) {
  const style = PRIORITY_STYLE[action.priority];
  const isNeg = action.sentiment === "negative";

  return (
    <div
      className={`panel relative overflow-hidden p-6 lg:p-7 bg-gradient-to-br ${style.glow} ring-1 ${style.ring}`}
    >
      <span className={`absolute left-0 top-0 bottom-0 w-1 ${style.accent}`} aria-hidden />

      <div className="grid lg:grid-cols-[1fr_auto] gap-5 lg:gap-8 items-start">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-1.5 mb-3">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-[0.16em] border ${style.badge}`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${style.accent}`} />
              {style.label}
            </span>
            <span className={isNeg ? "pill-neg" : "pill-pos"}>
              {isNeg ? "Tiêu cực" : "Tích cực"}
            </span>
            <span className="text-[11.5px] text-fg-muted ml-1">
              {ASPECT_LABEL_VI[action.aspect_category]}
              <span className="text-fg-dim ml-1.5">({action.aspect_category})</span>
            </span>
          </div>

          <div className="text-[10px] uppercase tracking-[0.18em] text-fg-faint font-semibold mb-2">
            Hành động đề xuất hàng đầu
          </div>
          <p className="text-[22px] lg:text-[26px] font-semibold leading-snug text-fg max-w-3xl text-balance">
            {action.action}
          </p>
        </div>

        {action.evidence_terms.length > 0 && (
          <div className="lg:border-l lg:border-line lg:pl-7 lg:min-w-[200px]">
            <div className="text-[9.5px] uppercase tracking-[0.16em] text-fg-faint font-semibold mb-2">
              Khách hay nhắc
            </div>
            <div className="flex flex-wrap gap-1.5">
              {action.evidence_terms.map((t) => (
                <span
                  key={t}
                  className="inline-flex items-center rounded-md border border-line bg-white px-2 py-0.5 text-[11.5px] text-fg shadow-soft"
                >
                  {t}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
