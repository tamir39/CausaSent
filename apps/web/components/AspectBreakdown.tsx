"use client";

import { ASPECT_LABEL_VI } from "@/lib/labels";
import type { AspectSummary, CorpusSummary, Sentiment } from "@/lib/api";

function priorityScore(a: AspectSummary): number {
  return a.negative * a.negative_ratio;
}

export function AspectBreakdown({ summary }: { summary: CorpusSummary }) {
  const aspects = Object.values(summary.aspects).filter(
    (a): a is AspectSummary => !!a
  );
  aspects.sort((a, b) => priorityScore(b) - priorityScore(a));

  if (aspects.length === 0) {
    return (
      <div className="panel-pad text-[12px] text-fg-faint italic">
        Chưa trích xuất được aspect nào từ batch này.
      </div>
    );
  }

  const maxMentions = Math.max(...aspects.map((a) => a.total_mentions));

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-2">
      {aspects.map((a) => {
        const total = a.positive + a.negative;
        const posPct = total ? (100 * a.positive) / total : 0;
        const negPct = 100 - posPct;
        const volumePct = (a.total_mentions / maxMentions) * 100;
        const negRatioPct = a.negative_ratio * 100;
        const isHotspot = negRatioPct > 50 && a.negative >= 3;
        return (
          <article
            key={a.aspect_category}
            className={`panel-pad panel-hover space-y-2 ${
              isHotspot ? "ring-1 ring-neg/25" : ""
            }`}
          >
            <header className="flex items-start justify-between gap-1.5">
              <div className="min-w-0">
                <div className="font-semibold text-[13px] text-fg tracking-tight truncate">
                  {ASPECT_LABEL_VI[a.aspect_category]}
                </div>
                <div className="text-[9.5px] text-fg-dim uppercase tracking-wider mt-0.5">
                  {a.aspect_category}
                </div>
              </div>
              <div className="text-right shrink-0">
                <div className="text-[18px] font-bold tabular-nums text-fg leading-none">
                  {a.total_mentions}
                </div>
                <div className="text-[9.5px] text-fg-faint uppercase tracking-wider mt-0.5">
                  mentions
                </div>
              </div>
            </header>

            <div>
              <div className="h-[3px] rounded-full bg-bg-subtle overflow-hidden">
                <div
                  className="h-full bg-brand-500"
                  style={{ width: `${volumePct}%` }}
                />
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex items-center justify-between text-[9.5px] uppercase tracking-wider">
                <span className="text-fg-faint">{a.n_reviews} reviews</span>
                <span
                  className={
                    isHotspot
                      ? "text-neg font-semibold tabular-nums"
                      : "text-fg-faint tabular-nums"
                  }
                >
                  {negRatioPct.toFixed(0)}% neg
                </span>
              </div>
              <div className="flex h-[6px] rounded-full overflow-hidden bg-bg-subtle">
                <div className="h-full bg-pos" style={{ width: `${posPct}%` }} />
                <div className="h-full bg-neg" style={{ width: `${negPct}%` }} />
              </div>
              <div className="flex justify-between text-[9.5px] tabular-nums">
                <span className="text-pos">+{a.positive}</span>
                <span className="text-neg">−{a.negative}</span>
              </div>
            </div>

            {(["negative", "positive"] as Sentiment[]).map((sent) => {
              const cell = a.cells[sent];
              if (!cell || cell.top_terms.length === 0) return null;
              return (
                <div key={sent}>
                  <div
                    className={`text-[9.5px] uppercase tracking-wider font-semibold mb-1 ${
                      sent === "negative" ? "text-neg" : "text-pos"
                    }`}
                  >
                    {sent === "negative" ? "Complaints" : "Praises"}
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {cell.top_terms.slice(0, 3).map(([term, n]) => (
                      <span
                        key={term}
                        className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10.5px] border ${
                          sent === "negative"
                            ? "border-neg/25 bg-neg-soft/60 text-neg"
                            : "border-pos/25 bg-pos-soft/60 text-pos"
                        }`}
                      >
                        {term}
                        <span className="opacity-60 tabular-nums">·{n}</span>
                      </span>
                    ))}
                  </div>
                </div>
              );
            })}
          </article>
        );
      })}
    </div>
  );
}
