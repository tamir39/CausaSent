"use client";

import { useMemo, useState } from "react";

import { ASPECT_LABEL_VI } from "@/lib/labels";
import type {
  AspectCategory,
  AspectSummary,
  CorpusSummary,
  PredictedTuple,
  Sentiment,
} from "@/lib/api";
import type { ReviewRow } from "./ReviewsTable";

function priorityScore(a: AspectSummary): number {
  return a.negative * a.negative_ratio;
}

interface Props {
  summary: CorpusSummary;
  /**
   * Per-review attribution. When provided, each aspect card becomes
   * expandable to show *which* reviews mention this aspect, with the
   * matching aspect_term span highlighted.
   */
  reviews?: ReviewRow[];
}

export function AspectBreakdown({ summary, reviews }: Props) {
  const aspects = Object.values(summary.aspects).filter(
    (a): a is AspectSummary => !!a
  );
  aspects.sort((a, b) => priorityScore(b) - priorityScore(a));

  // Build a lookup: aspect_category → reviews mentioning it, grouped by sentiment.
  // Computed once per render of the parent (cheap — N reviews × few tuples each).
  const referenceMap = useMemo(() => {
    const m: Record<
      string,
      { positive: Array<{ row: ReviewRow; tuple: PredictedTuple }>; negative: Array<{ row: ReviewRow; tuple: PredictedTuple }> }
    > = {};
    if (!reviews) return m;
    for (const row of reviews) {
      for (const t of row.tuples) {
        const bucket =
          (m[t.aspect_category] ||= { positive: [], negative: [] });
        const arr = t.sentiment === "negative" ? bucket.negative : bucket.positive;
        arr.push({ row, tuple: t });
      }
    }
    return m;
  }, [reviews]);

  const [expanded, setExpanded] = useState<AspectCategory | null>(null);
  // Optional inline filter by term (when user clicks a chip).
  const [termFilter, setTermFilter] = useState<{
    aspect: AspectCategory;
    sentiment: Sentiment;
    term: string;
  } | null>(null);

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
        const isExp = expanded === a.aspect_category;
        const refs = referenceMap[a.aspect_category];
        const refCount = refs ? refs.positive.length + refs.negative.length : 0;
        const expandable = reviews != null && refCount > 0;
        return (
          <article
            key={a.aspect_category}
            className={`panel-pad space-y-2 ${
              isExp ? "ring-1 ring-brand-500/40 md:col-span-2 xl:col-span-3" : "panel-hover"
            } ${isHotspot && !isExp ? "ring-1 ring-neg/25" : ""}`}
          >
            <header
              className={`flex items-start justify-between gap-1.5 ${
                expandable ? "cursor-pointer select-none" : ""
              }`}
              onClick={() => {
                if (!expandable) return;
                if (isExp) {
                  setExpanded(null);
                  setTermFilter(null);
                } else {
                  setExpanded(a.aspect_category);
                  setTermFilter(null);
                }
              }}
            >
              <div className="min-w-0">
                <div className="font-semibold text-[13px] text-fg tracking-tight truncate flex items-center gap-1.5">
                  {ASPECT_LABEL_VI[a.aspect_category]}
                  {expandable && (
                    <span
                      className="text-fg-dim text-[10px]"
                      aria-hidden
                      title={isExp ? "thu gọn" : "click để xem reviews"}
                    >
                      {isExp ? "▾" : "▸"}
                    </span>
                  )}
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
                    {cell.top_terms.slice(0, 3).map(([term, n]) => {
                      const active =
                        termFilter?.aspect === a.aspect_category &&
                        termFilter.sentiment === sent &&
                        termFilter.term === term;
                      return (
                        <button
                          key={term}
                          type="button"
                          disabled={!expandable}
                          onClick={(e) => {
                            e.stopPropagation();
                            if (!expandable) return;
                            setExpanded(a.aspect_category);
                            setTermFilter(
                              active
                                ? null
                                : { aspect: a.aspect_category, sentiment: sent, term }
                            );
                          }}
                          className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10.5px] border transition ${
                            sent === "negative"
                              ? active
                                ? "border-neg bg-neg text-white"
                                : "border-neg/25 bg-neg-soft/60 text-neg hover:bg-neg-soft"
                              : active
                              ? "border-pos bg-pos text-white"
                              : "border-pos/25 bg-pos-soft/60 text-pos hover:bg-pos-soft"
                          } ${expandable ? "cursor-pointer" : "cursor-default"}`}
                          title={
                            expandable
                              ? active
                                ? "bỏ filter — xem tất cả review"
                                : `lọc review chứa "${term}"`
                              : undefined
                          }
                        >
                          {term}
                          <span className="opacity-60 tabular-nums">·{n}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}

            {/* ============ EXPANDED REFERENCES PANEL ============ */}
            {isExp && refs && (
              <div className="mt-3 pt-3 border-t border-line space-y-3">
                <div className="flex items-center justify-between">
                  <div className="text-[10px] uppercase tracking-[0.14em] text-fg-faint font-semibold">
                    Reviews tham chiếu
                    {termFilter && (
                      <>
                        {" "}
                        · lọc theo{" "}
                        <span
                          className={
                            termFilter.sentiment === "negative"
                              ? "text-neg"
                              : "text-pos"
                          }
                        >
                          “{termFilter.term}”
                        </span>
                      </>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      setExpanded(null);
                      setTermFilter(null);
                    }}
                    className="text-[10.5px] text-fg-faint hover:text-fg underline"
                  >
                    đóng
                  </button>
                </div>
                <div className="grid md:grid-cols-2 gap-3">
                  <RefList
                    sentiment="negative"
                    items={refs.negative}
                    termFilter={termFilter}
                    aspect={a.aspect_category}
                  />
                  <RefList
                    sentiment="positive"
                    items={refs.positive}
                    termFilter={termFilter}
                    aspect={a.aspect_category}
                  />
                </div>
              </div>
            )}
          </article>
        );
      })}
    </div>
  );
}

function RefList({
  sentiment,
  items,
  termFilter,
  aspect,
}: {
  sentiment: Sentiment;
  items: Array<{ row: ReviewRow; tuple: PredictedTuple }>;
  termFilter: { aspect: AspectCategory; sentiment: Sentiment; term: string } | null;
  aspect: AspectCategory;
}) {
  const filtered =
    termFilter && termFilter.aspect === aspect && termFilter.sentiment === sentiment
      ? items.filter((it) =>
          it.tuple.aspect_term
            .normalize("NFC")
            .toLowerCase()
            .includes(termFilter.term.toLowerCase())
        )
      : termFilter
      ? [] // term filter active but for a different sentiment → hide
      : items;

  const color = sentiment === "negative" ? "text-neg" : "text-pos";
  const label = sentiment === "negative" ? "Tiêu cực" : "Tích cực";

  return (
    <div>
      <div className={`text-[10.5px] uppercase tracking-wider font-semibold mb-1.5 ${color}`}>
        {label} · {filtered.length}
      </div>
      {filtered.length === 0 ? (
        <div className="text-[11px] text-fg-dim italic">không có review nào.</div>
      ) : (
        <ul className="space-y-1.5 max-h-80 overflow-y-auto pr-1">
          {filtered.map((it, i) => (
            <ReviewItem key={`${it.row.id}-${i}`} row={it.row} tuple={it.tuple} />
          ))}
        </ul>
      )}
    </div>
  );
}

function ReviewItem({ row, tuple }: { row: ReviewRow; tuple: PredictedTuple }) {
  const [s, e] = tuple.aspect_term_span;
  const review = row.review;
  const before = review.slice(0, s);
  const span = review.slice(s, e);
  const after = review.slice(e);
  const isNeg = tuple.sentiment === "negative";
  return (
    <li
      className={`rounded-md border px-2 py-1.5 text-[12px] leading-snug ${
        isNeg ? "border-neg/20 bg-neg-soft/30" : "border-pos/20 bg-pos-soft/30"
      }`}
    >
      <div className="flex items-baseline gap-1.5 mb-0.5">
        <span className="text-[9.5px] text-fg-dim uppercase tracking-wider tabular-nums">
          #{row.id}
        </span>
        <span className="text-[10px] text-fg-faint">
          conf {(tuple.confidence * 100).toFixed(0)}%
        </span>
      </div>
      <div className="text-fg break-words">
        <span className="text-fg-muted">{before}</span>
        <mark
          className={`px-0.5 rounded-sm font-semibold ${
            isNeg
              ? "bg-neg/15 text-neg"
              : "bg-pos/15 text-pos"
          }`}
        >
          {span}
        </mark>
        <span className="text-fg-muted">{after}</span>
      </div>
    </li>
  );
}
