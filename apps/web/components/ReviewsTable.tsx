"use client";

import { useMemo, useState } from "react";

import { ASPECT_LABEL_VI } from "@/lib/labels";
import type { PredictedTuple } from "@/lib/api";

export interface ReviewRow {
  id: string;
  review: string;
  tuples: PredictedTuple[];
}

type FilterMode = "all" | "with-aspect" | "no-aspect" | "negative";

/**
 * Live-updating table of reviews + their extracted tuples.
 * Rows append as `review` events stream in; search filters client-side.
 */
export function ReviewsTable({
  rows,
  expectedTotal,
}: {
  rows: ReviewRow[];
  expectedTotal: number;
}) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<FilterMode>("all");
  const [expanded, setExpanded] = useState<string | null>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return rows.filter((r) => {
      if (mode === "with-aspect" && r.tuples.length === 0) return false;
      if (mode === "no-aspect" && r.tuples.length > 0) return false;
      if (
        mode === "negative" &&
        !r.tuples.some((t) => t.sentiment === "negative")
      )
        return false;
      if (!q) return true;
      if (r.review.toLowerCase().includes(q)) return true;
      if (
        r.tuples.some(
          (t) =>
            t.aspect_term.toLowerCase().includes(q) ||
            t.aspect_category.toLowerCase().includes(q) ||
            (ASPECT_LABEL_VI[t.aspect_category] || "")
              .toLowerCase()
              .includes(q)
        )
      )
        return true;
      return false;
    });
  }, [rows, query, mode]);

  const counts = useMemo(
    () => ({
      all: rows.length,
      with: rows.filter((r) => r.tuples.length > 0).length,
      without: rows.filter((r) => r.tuples.length === 0).length,
      neg: rows.filter((r) => r.tuples.some((t) => t.sentiment === "negative"))
        .length,
    }),
    [rows]
  );

  return (
    <div className="panel">
      <div className="flex flex-wrap items-center gap-2 px-3 py-2 border-b border-line">
        <div className="flex items-center gap-1.5">
          <FilterChip
            active={mode === "all"}
            onClick={() => setMode("all")}
            count={counts.all}
          >
            Tất cả
          </FilterChip>
          <FilterChip
            active={mode === "with-aspect"}
            onClick={() => setMode("with-aspect")}
            count={counts.with}
            tone="pos"
          >
            Có aspect
          </FilterChip>
          <FilterChip
            active={mode === "negative"}
            onClick={() => setMode("negative")}
            count={counts.neg}
            tone="neg"
          >
            Có complaint
          </FilterChip>
          <FilterChip
            active={mode === "no-aspect"}
            onClick={() => setMode("no-aspect")}
            count={counts.without}
            tone="muted"
          >
            Không có aspect
          </FilterChip>
        </div>
        <div className="flex-1" />
        <div className="relative">
          <input
            type="search"
            placeholder="Tìm theo text / aspect / category…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-56 rounded-md text-[12px] px-2.5 py-1 border"
          />
        </div>
        <span className="text-[10.5px] text-fg-faint tabular-nums">
          {filtered.length}/{rows.length}
          {rows.length < expectedTotal ? ` · streaming ${rows.length}/${expectedTotal}` : ""}
        </span>
      </div>

      <div className="max-h-[420px] overflow-auto">
        {filtered.length === 0 ? (
          <div className="px-4 py-8 text-center text-[12px] text-fg-faint">
            {rows.length === 0
              ? "Chưa có review nào — chờ stream bắt đầu…"
              : "Không có review nào khớp filter."}
          </div>
        ) : (
          <table className="w-full text-[12.5px]">
            <thead className="sticky top-0 bg-white/95 backdrop-blur z-10 border-b border-line">
              <tr className="text-left text-[10px] uppercase tracking-[0.14em] text-fg-faint">
                <th className="px-3 py-2 w-10 text-right">#</th>
                <th className="px-3 py-2">Review</th>
                <th className="px-3 py-2 w-[42%]">Aspects</th>
                <th className="px-3 py-2 w-12 text-right">Tuples</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line/60">
              {filtered.map((r, i) => {
                const isExp = expanded === r.id;
                const hasNeg = r.tuples.some((t) => t.sentiment === "negative");
                return (
                    <tr
                      key={r.id}
                      className={`group cursor-pointer transition ${
                        isExp
                          ? "bg-bg-subtle"
                          : "hover:bg-bg-subtle/70"
                      }`}
                      onClick={() => setExpanded(isExp ? null : r.id)}
                    >
                      <td className="px-3 py-2 text-right text-[10.5px] tabular-nums text-fg-dim align-top pt-2.5">
                        {i + 1}
                      </td>
                      <td className="px-3 py-2 align-top">
                        <div
                          className={`leading-snug ${
                            isExp ? "" : "line-clamp-2"
                          } ${
                            r.tuples.length === 0 ? "text-fg-faint italic" : "text-fg"
                          }`}
                          title={r.review}
                        >
                          {r.review || "(empty review)"}
                        </div>
                      </td>
                      <td className="px-3 py-2 align-top">
                        {r.tuples.length === 0 ? (
                          <span className="text-[10.5px] text-fg-dim italic">
                            không có aspect
                          </span>
                        ) : (
                          <div className="flex flex-wrap gap-1">
                            {r.tuples.slice(0, isExp ? r.tuples.length : 4).map((t, idx) => (
                              <span
                                key={idx}
                                className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0 text-[10.5px] border ${
                                  t.sentiment === "negative"
                                    ? "border-neg/30 bg-neg-soft/60 text-neg"
                                    : "border-pos/30 bg-pos-soft/60 text-pos"
                                }`}
                                title={`${t.aspect_category} · ${t.sentiment} · conf ${(t.confidence * 100).toFixed(0)}%`}
                              >
                                {ASPECT_LABEL_VI[t.aspect_category]} · {t.aspect_term}
                              </span>
                            ))}
                            {!isExp && r.tuples.length > 4 && (
                              <span className="text-[10.5px] text-fg-faint self-center">
                                +{r.tuples.length - 4}
                              </span>
                            )}
                          </div>
                        )}
                      </td>
                      <td className="px-3 py-2 align-top text-right">
                        <span
                          className={`text-[12px] font-semibold tabular-nums ${
                            hasNeg ? "text-neg" : r.tuples.length > 0 ? "text-pos" : "text-fg-dim"
                          }`}
                        >
                          {r.tuples.length}
                        </span>
                      </td>
                    </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function FilterChip({
  active,
  onClick,
  count,
  tone = "default",
  children,
}: {
  active: boolean;
  onClick: () => void;
  count: number;
  tone?: "default" | "pos" | "neg" | "muted";
  children: React.ReactNode;
}) {
  const activeRing =
    tone === "neg"
      ? "border-neg/50 bg-neg-soft text-neg"
      : tone === "pos"
      ? "border-pos/50 bg-pos-soft text-pos"
      : tone === "muted"
      ? "border-line-soft bg-bg-subtle text-fg-muted"
      : "border-brand-500/40 bg-brand-50 text-brand-700";
  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[11px] font-medium transition ${
        active
          ? activeRing
          : "border-line text-fg-muted hover:bg-bg-subtle"
      }`}
    >
      {children}
      <span className="text-[10px] tabular-nums opacity-70">{count}</span>
    </button>
  );
}
