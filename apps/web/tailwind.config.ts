import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // "Paper" theme — clean light, Linear/Vercel-style.
        bg: {
          DEFAULT: "#ffffff",
          soft:    "#fafafa",   // page canvas
          subtle:  "#f4f4f5",   // hover surfaces
          panel:   "#ffffff",
        },
        line: {
          DEFAULT: "#e4e4e7",   // zinc-200 — main border
          soft:    "#d4d4d8",   // zinc-300 — emphasised border
          faint:   "#f4f4f5",   // zinc-100 — barely-there border
        },
        fg: {
          DEFAULT: "#09090b",   // zinc-950 — body text
          muted:   "#3f3f46",   // zinc-700
          faint:   "#71717a",   // zinc-500 — labels
          dim:     "#a1a1aa",   // zinc-400 — captions
        },
        brand: {
          50:  "#eef2ff",
          100: "#e0e7ff",
          300: "#a5b4fc",
          400: "#818cf8",
          500: "#6366f1",  // indigo-500 primary
          600: "#4f46e5",
          700: "#4338ca",
        },
        pos: {
          DEFAULT: "#059669",   // emerald-600 — strong on white
          soft:    "#d1fae5",   // emerald-100
        },
        neg: {
          DEFAULT: "#e11d48",   // rose-600
          soft:    "#ffe4e6",   // rose-100
        },
        warn: {
          DEFAULT: "#d97706",   // amber-600
          soft:    "#fef3c7",   // amber-100
        },
      },
      boxShadow: {
        soft: "0 1px 2px 0 rgba(0,0,0,0.04), 0 1px 3px 0 rgba(0,0,0,0.05)",
        glow: "0 1px 2px 0 rgba(99,102,241,0.10), 0 8px 24px -10px rgba(99,102,241,0.35)",
        panel: "0 1px 0 0 rgba(0,0,0,0.02), 0 1px 2px 0 rgba(0,0,0,0.04)",
      },
    },
  },
  plugins: [],
};

export default config;
