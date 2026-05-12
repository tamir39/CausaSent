"use client";

import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

import type { CorpusSummary } from "@/lib/api";

const POS = "#059669";
const NEG = "#e11d48";

export function SentimentDonut({ summary }: { summary: CorpusSummary }) {
  const pos = summary.sentiment_overall.positive ?? 0;
  const neg = summary.sentiment_overall.negative ?? 0;
  const total = pos + neg;
  const negPct = total ? (100 * neg) / total : 0;
  const posPct = total ? 100 - negPct : 0;
  const data = total
    ? [
        { name: "Tích cực", value: pos, color: POS },
        { name: "Tiêu cực", value: neg, color: NEG },
      ]
    : [{ name: "Chưa có dữ liệu", value: 1, color: "#e4e4e7" }];

  const tone =
    negPct >= 60
      ? { label: "Cảnh báo", color: "text-neg" }
      : negPct >= 30
      ? { label: "Cần chú ý", color: "text-warn" }
      : total
      ? { label: "Tốt", color: "text-pos" }
      : { label: "—", color: "text-fg-faint" };

  return (
    <div className="panel-pad">
      <div className="flex items-baseline justify-between mb-2">
        <h3 className="text-[12px] font-semibold tracking-[0.04em] uppercase text-fg">
          Sentiment tổng quan
        </h3>
        <span className={`text-[10.5px] font-semibold ${tone.color}`}>
          {tone.label}
        </span>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative w-[140px] h-[140px] shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                dataKey="value"
                innerRadius={45}
                outerRadius={65}
                strokeWidth={0}
                startAngle={90}
                endAngle={-270}
              >
                {data.map((d, i) => (
                  <Cell key={i} fill={d.color} />
                ))}
              </Pie>
              {total > 0 && (
                <Tooltip
                  contentStyle={{
                    background: "#ffffff",
                    border: "1px solid #e4e4e7",
                    borderRadius: 6,
                    fontSize: 11,
                    padding: "4px 8px",
                    boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
                  }}
                  formatter={(value: number, name: string) => [
                    `${value} (${((100 * value) / total).toFixed(1)}%)`,
                    name,
                  ]}
                />
              )}
            </PieChart>
          </ResponsiveContainer>
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
            <div
              className={`text-[22px] font-bold tabular-nums leading-none ${
                total ? "text-fg" : "text-fg-faint"
              }`}
            >
              {total ? `${negPct.toFixed(0)}%` : "—"}
            </div>
            <div className="text-[9.5px] uppercase tracking-wider text-fg-faint mt-0.5">
              tiêu cực
            </div>
          </div>
        </div>

        <div className="flex-1 space-y-2 min-w-0">
          <LegendRow color={POS} label="Tích cực" value={pos} pct={posPct} />
          <LegendRow color={NEG} label="Tiêu cực" value={neg} pct={negPct} />
          <div className="pt-2 border-t border-line text-[10.5px] text-fg-faint tabular-nums">
            Tổng {total} tuples từ {summary.n_reviews_with_tuples}/
            {summary.n_reviews} reviews
          </div>
        </div>
      </div>
    </div>
  );
}

function LegendRow({
  color,
  label,
  value,
  pct,
}: {
  color: string;
  label: string;
  value: number;
  pct: number;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ background: color }} />
      <span className="text-[12px] text-fg-muted">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-bg-subtle overflow-hidden">
        <div className="h-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-[12px] font-semibold tabular-nums text-fg w-9 text-right">
        {value}
      </span>
    </div>
  );
}
