"use client";

import {
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

import { ASPECT_LABEL_VI } from "@/lib/labels";
import type { CorpusSummary } from "@/lib/api";

const POS = "#059669"; // emerald-600
const NEG = "#e11d48"; // rose-600

export function AspectRadar({ summary }: { summary: CorpusSummary }) {
  const aspects = Object.keys(ASPECT_LABEL_VI) as (keyof typeof ASPECT_LABEL_VI)[];
  const data = aspects.map((a) => {
    const s = summary.aspects[a];
    return {
      aspect: ASPECT_LABEL_VI[a],
      positive: s?.positive ?? 0,
      negative: s?.negative ?? 0,
    };
  });

  return (
    <div className="flex flex-col sm:flex-row items-center gap-2">
      <div className="flex-1 w-full min-w-0">
        <ResponsiveContainer width="100%" height={210}>
          <RadarChart data={data} margin={{ top: 6, right: 24, bottom: 6, left: 24 }}>
            <PolarGrid stroke="#e4e4e7" />
            <PolarAngleAxis
              dataKey="aspect"
              tick={{ fill: "#52525b", fontSize: 10.5, fontWeight: 500 }}
            />
            <Radar
              name="Tiêu cực"
              dataKey="negative"
              stroke={NEG}
              fill={NEG}
              fillOpacity={0.28}
              strokeWidth={1.5}
            />
            <Radar
              name="Tích cực"
              dataKey="positive"
              stroke={POS}
              fill={POS}
              fillOpacity={0.18}
              strokeWidth={1.5}
            />
            <Tooltip
              cursor={{ stroke: "#d4d4d8" }}
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
          </RadarChart>
        </ResponsiveContainer>
      </div>
      <ul className="text-[10.5px] space-y-1.5 sm:min-w-[110px]">
        <li className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm bg-pos" />
          <span className="text-fg-muted">Tích cực</span>
        </li>
        <li className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm bg-neg" />
          <span className="text-fg-muted">Tiêu cực</span>
        </li>
      </ul>
    </div>
  );
}
