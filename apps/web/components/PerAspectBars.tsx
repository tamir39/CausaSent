"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ASPECT_LABEL_VI } from "@/lib/labels";
import type { CorpusSummary } from "@/lib/api";

const POS = "#059669";
const NEG = "#e11d48";

/**
 * Horizontal grouped/stacked bar chart for per-aspect breakdown.
 * Easier to read exact counts than the radar.
 */
export function PerAspectBars({ summary }: { summary: CorpusSummary }) {
  const data = Object.keys(ASPECT_LABEL_VI).map((cat) => {
    const s = summary.aspects[cat as keyof typeof ASPECT_LABEL_VI];
    return {
      aspect: ASPECT_LABEL_VI[cat as keyof typeof ASPECT_LABEL_VI],
      cat,
      positive: s?.positive ?? 0,
      negative: s?.negative ?? 0,
    };
  });
  // Sort by total mentions descending — most-talked-about on top.
  data.sort((a, b) => b.positive + b.negative - (a.positive + a.negative));

  const hasData = data.some((d) => d.positive + d.negative > 0);

  return (
    <div className="panel-pad">
      <div className="flex items-baseline justify-between mb-2">
        <h3 className="text-[12px] font-semibold tracking-[0.04em] uppercase text-fg">
          Số lần nhắc theo aspect
        </h3>
        <span className="text-[10.5px] text-fg-faint">
          stacked · sort theo total mentions
        </span>
      </div>
      {!hasData ? (
        <div className="text-[12px] text-fg-faint italic py-6 text-center">
          Chưa có dữ liệu.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 5, right: 16, bottom: 5, left: 16 }}
          >
            <CartesianGrid horizontal={false} stroke="#f4f4f5" />
            <XAxis
              type="number"
              tick={{ fill: "#71717a", fontSize: 10 }}
              axisLine={{ stroke: "#e4e4e7" }}
              tickLine={{ stroke: "#e4e4e7" }}
            />
            <YAxis
              type="category"
              dataKey="aspect"
              tick={{ fill: "#3f3f46", fontSize: 11, fontWeight: 500 }}
              axisLine={{ stroke: "#e4e4e7" }}
              tickLine={{ stroke: "#e4e4e7" }}
              width={100}
            />
            <Tooltip
              cursor={{ fill: "rgba(0,0,0,0.03)" }}
              contentStyle={{
                background: "#ffffff",
                border: "1px solid #e4e4e7",
                borderRadius: 6,
                fontSize: 11,
                padding: "4px 8px",
                boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
              }}
              labelStyle={{ color: "#09090b", fontWeight: 600, marginBottom: 2 }}
            />
            <Legend
              wrapperStyle={{ fontSize: 11, paddingTop: 4 }}
              iconType="square"
              iconSize={8}
            />
            <Bar dataKey="positive" name="Tích cực" stackId="s" fill={POS} radius={[0, 0, 0, 0]} />
            <Bar dataKey="negative" name="Tiêu cực" stackId="s" fill={NEG} radius={[0, 2, 2, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
