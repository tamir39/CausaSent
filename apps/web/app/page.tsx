"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { getHealth, type AnalyzeResult, type HealthResponse } from "@/lib/api";
import { persistPendingJob, persistResult } from "@/lib/resultStorage";
import { SAMPLE_RESULT } from "@/lib/sampleResult";

const SAMPLE = `Ship lâu nhưng đóng gói đẹp, sản phẩm dùng tạm được, giá cũng ổn.
Áo chất vải mịn, mặc thoáng, shop tư vấn nhiệt tình, giao nhanh.
Hàng giao chậm 5 ngày, hộp móp méo, nhân viên trả lời cộc lốc, thất vọng.
Giá hơi cao so với chất lượng, nhưng màu sắc và kiểu dáng thì ưng.
Sản phẩm chất lượng tốt, giao hàng nhanh, giá hợp lý.
Đóng gói cẩn thận, sản phẩm đẹp như hình, sẽ ủng hộ shop tiếp.`;

const ASPECTS = [
  "delivery",
  "packaging",
  "product_quality",
  "price",
  "customer_service",
  "usability",
  "appearance",
];

const PIPELINE = [
  {
    n: 1,
    title: "Aspect-Term Extraction",
    sub: "PhoBERT-large · 2 heads · contrastive",
    body: "Token-level BIO trên 7 categories + binary sentiment ở B-token.",
  },
  {
    n: 2,
    title: "Aggregation",
    sub: "frequency rank · NFC normalize",
    body: "Gộp N reviews → ranking aspect_terms theo (aspect, sentiment) cell.",
  },
  {
    n: 3,
    title: "Action Recommendation",
    sub: "Gemini 2.5 Flash · single call",
    body: "Một LLM call duy nhất, trả JSON actions có evidence terms.",
  },
];

export default function HomePage() {
  const router = useRouter();
  const [text, setText] = useState(SAMPLE);
  const [file, setFile] = useState<File | null>(null);
  const [useLlm, setUseLlm] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reviewLines = text.split("\n").filter((l) => l.trim()).length;

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthErr, setHealthErr] = useState(false);
  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((h) => {
        if (!cancelled) setHealth(h);
      })
      .catch(() => {
        if (!cancelled) setHealthErr(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);
  const backendReady = !!health;
  const ckptMissing = health && !health.ckpt_exists;

  async function handleAnalyze() {
    setError(null);
    setBusy(true);
    try {
      let reviews: string[];
      let source: "paste" | "csv" = "paste";
      let csvFilename: string | undefined;
      if (file) {
        // Parse CSV in-browser so the dashboard can stream review-by-review
        // without an extra round-trip to /analyze-csv.
        const text = await file.text();
        reviews = await parseCsvReviews(text);
        source = "csv";
        csvFilename = file.name;
        if (reviews.length === 0)
          throw new Error("CSV không có review nào (auto-detect cột thất bại).");
      } else {
        reviews = text.split("\n").map((r) => r.trim()).filter(Boolean);
        if (reviews.length === 0) throw new Error("Cần ít nhất 1 review.");
      }

      if (!backendReady) {
        throw new Error(
          "Backend chưa sẵn sàng ở http://localhost:8000. Chờ badge chuyển xanh 'API live' rồi thử lại — hoặc chạy `.\\run.bat` nếu chưa start."
        );
      }

      persistPendingJob({
        reviews,
        useLlm,
        source,
        csvFilename,
        startedAt: Date.now(),
      });
      router.push("/dashboard");
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }

  function viewSample() {
    persistResult(SAMPLE_RESULT);
    router.push("/dashboard");
  }

  // Naive CSV parser: handles quoted fields, commas, line breaks within quotes.
  // Picks the first column matching common review-column names; falls back to
  // the longest-text column.
  async function parseCsvReviews(text: string): Promise<string[]> {
    const rows: string[][] = [];
    let field = "";
    let row: string[] = [];
    let inQ = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (inQ) {
        if (ch === '"' && text[i + 1] === '"') {
          field += '"';
          i++;
        } else if (ch === '"') {
          inQ = false;
        } else {
          field += ch;
        }
      } else {
        if (ch === '"') inQ = true;
        else if (ch === ",") {
          row.push(field);
          field = "";
        } else if (ch === "\r") {
          /* skip */
        } else if (ch === "\n") {
          row.push(field);
          field = "";
          rows.push(row);
          row = [];
        } else {
          field += ch;
        }
      }
    }
    if (field.length > 0 || row.length > 0) {
      row.push(field);
      rows.push(row);
    }
    if (rows.length < 2) return [];
    // Strip BOM from first header
    const header = rows[0].map((h) => h.replace(/^﻿/, "").trim().toLowerCase());
    const candidates = ["review", "content", "comment", "text", "review_text"];
    let col = -1;
    for (const c of candidates) {
      const idx = header.indexOf(c);
      if (idx !== -1) {
        col = idx;
        break;
      }
    }
    if (col === -1) {
      // fallback: column with the longest average text length
      const len: number[] = header.map(() => 0);
      for (let r = 1; r < rows.length; r++) {
        rows[r].forEach((v, c) => {
          len[c] += (v || "").length;
        });
      }
      col = len.indexOf(Math.max(...len));
    }
    return rows
      .slice(1)
      .map((r) => (r[col] || "").trim())
      .filter(Boolean);
  }

  async function loadRealSample() {
    setError(null);
    try {
      const res = await fetch("/sample-real-30.json", { cache: "no-store" });
      if (!res.ok) throw new Error(`fetch ${res.status}`);
      const data = (await res.json()) as AnalyzeResult;
      persistResult(data);
      router.push("/dashboard");
    } catch (e) {
      setError(
        `Không load được 30-review result: ${
          (e as Error).message
        }. Đảm bảo file public/sample-real-30.json tồn tại.`
      );
    }
  }

  return (
    <div className="space-y-10">
      {/* ============ HERO: text left, form right ============ */}
      <div className="grid lg:grid-cols-[1.05fr_1fr] gap-8 lg:gap-12 items-start">
        <section className="space-y-6 pt-2">
          <div className="inline-flex items-center gap-2 chip">
            <span className="w-1.5 h-1.5 rounded-full bg-pos animate-pulse" />
            <span className="text-fg">ATE v2 · PhoBERT-large + Gemini</span>
          </div>

          <h1 className="text-[30px] lg:text-[36px] font-bold tracking-tight leading-[1.08] text-balance">
            Từ review TMĐT tiếng Việt tới{" "}
            <span className="bg-gradient-to-r from-brand-600 to-pink-600 bg-clip-text text-transparent">
              hành động cụ thể
            </span>{" "}
            cho người bán.
          </h1>

          <p className="text-fg-muted text-[14px] leading-relaxed max-w-xl">
            Trích xuất{" "}
            <span className="text-fg font-medium">aspect · category · sentiment</span>{" "}
            cho từng review, gộp N reviews thành dashboard và recommended actions
            ưu tiên theo mức độ tiêu cực.
          </p>

          <div className="flex flex-wrap gap-1.5">
            {ASPECTS.map((a) => (
              <span key={a} className="chip">
                {a}
              </span>
            ))}
          </div>

          <div className="flex flex-wrap gap-2 pt-1">
            <button onClick={viewSample} className="btn-outline">
              → Xem dashboard mẫu
            </button>
            <button onClick={loadRealSample} className="btn-outline">
              → Load 30 reviews thật (model output)
            </button>
            <a
              href="https://huggingface.co/datasets/Tamir39/causasent-ate-v2"
              target="_blank"
              rel="noreferrer"
              className="btn-ghost"
            >
              Tải dataset trên HF ↗
            </a>
          </div>
        </section>

        <section className="panel p-5 space-y-4 lg:sticky lg:top-[60px]">
          <div className="flex items-center justify-between">
            <h2 className="text-[11px] font-semibold tracking-[0.14em] uppercase text-fg-muted">
              Phân tích batch
            </h2>
            <div className="flex items-center gap-2">
              <BackendBadge
                ready={backendReady}
                ckptMissing={!!ckptMissing}
                erred={healthErr}
              />
              <span className="text-[10.5px] text-fg-faint tabular-nums">
                {file
                  ? `1 file · ${Math.round(file.size / 1024)} KB`
                  : `${reviewLines} reviews`}
              </span>
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-[10.5px] text-fg-faint font-medium flex items-center justify-between">
              <span>Dán review (1 dòng = 1 review)</span>
              {file && (
                <button
                  type="button"
                  onClick={() => setFile(null)}
                  className="text-brand-600 hover:underline text-[10.5px]"
                >
                  ✕ bỏ file
                </button>
              )}
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={7}
              disabled={!!file}
              className="w-full rounded-md px-2.5 py-2 text-[12.5px] font-mono leading-relaxed resize-none disabled:opacity-40"
              placeholder="Review 1…&#10;Review 2…"
            />
          </div>

          <div className="relative">
            <div className="absolute inset-y-1/2 left-0 right-0 border-t border-line" />
            <div className="relative flex justify-center">
              <span className="bg-white px-2 text-[9px] uppercase tracking-widest text-fg-dim">
                hoặc
              </span>
            </div>
          </div>

          <label className="block space-y-1.5">
            <span className="text-[10.5px] text-fg-faint font-medium flex items-center justify-between">
              <span>Upload CSV (auto-detect cột review)</span>
              <a
                href="/reviews_demo.csv"
                download
                className="text-brand-600 hover:underline text-[10.5px]"
              >
                ↓ tải mẫu (30 reviews)
              </a>
            </span>
            <input
              type="file"
              accept=".csv,.tsv"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="block w-full text-[11px] text-fg-muted
                file:mr-2 file:py-1.5 file:px-2.5 file:rounded-md
                file:border file:border-line file:bg-bg-subtle file:text-fg file:font-medium
                file:cursor-pointer hover:file:bg-line/50"
            />
          </label>

          <label className="flex items-center gap-2 text-[11px] text-fg-muted cursor-pointer select-none">
            <input
              type="checkbox"
              checked={useLlm}
              onChange={(e) => setUseLlm(e.target.checked)}
              className="accent-brand-500"
            />
            Sinh action bằng{" "}
            <span className="text-fg font-medium">Gemini 2.5 Flash</span>
            <span className="text-fg-dim">· bỏ chọn = template</span>
          </label>

          {error && (
            <div className="rounded-md border border-neg/30 bg-neg-soft p-2.5 text-[11px] text-neg leading-relaxed">
              <span className="font-semibold">Lỗi:</span> {error}
            </div>
          )}

          <button onClick={handleAnalyze} disabled={busy || !backendReady} className="btn-primary w-full">
            {busy ? (
              <>
                <span className="inline-block w-3 h-3 border-2 border-white/70 border-t-transparent rounded-full animate-spin" />
                Mở dashboard…
              </>
            ) : (
              <>
                Phân tích →{" "}
                <span className="opacity-70">
                  {file ? "(CSV)" : `${reviewLines || "?"} reviews`}
                </span>
              </>
            )}
          </button>
          {!backendReady && !healthErr && (
            <p className="text-[10.5px] text-fg-faint text-center">
              Đợi backend khởi động…
            </p>
          )}
        </section>
      </div>

      {/* skip — moved before pipeline below */}
      {/* ============ PIPELINE: full-width row of cards, no horizontal divider ============ */}
      <section>
        <div className="flex items-baseline justify-between mb-4">
          <h2 className="text-[12px] font-semibold tracking-[0.04em] text-fg uppercase">
            Pipeline
          </h2>
          <span className="text-[10.5px] text-fg-faint">
            3 tầng tách biệt · từ raw review tới action
          </span>
        </div>
        <div className="grid md:grid-cols-3 gap-3">
          {PIPELINE.map((p) => (
            <div key={p.n} className="panel-pad panel-hover space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[9.5px] tracking-[0.16em] text-brand-600 font-bold">
                  STEP 0{p.n}
                </span>
                <span className="text-[9.5px] text-fg-dim uppercase tracking-wider">
                  {p.sub}
                </span>
              </div>
              <h3 className="text-[14px] font-semibold text-fg tracking-tight">
                {p.title}
              </h3>
              <p className="text-[12px] text-fg-muted leading-relaxed">{p.body}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function BackendBadge({
  ready,
  ckptMissing,
  erred,
}: {
  ready: boolean;
  ckptMissing: boolean;
  erred: boolean;
}) {
  if (ready && ckptMissing) {
    return (
      <span
        className="inline-flex items-center gap-1 rounded-full border border-warn/30 bg-warn-soft px-2 py-0.5 text-[10px] font-semibold text-warn"
        title="Backend chạy nhưng chưa có checkpoints/phobert/best.pt"
      >
        <span className="w-1.5 h-1.5 rounded-full bg-warn" /> API · no ckpt
      </span>
    );
  }
  if (ready) {
    return (
      <span
        className="inline-flex items-center gap-1 rounded-full border border-pos/30 bg-pos-soft px-2 py-0.5 text-[10px] font-semibold text-pos"
        title="FastAPI backend đang chạy ở localhost:8000"
      >
        <span className="w-1.5 h-1.5 rounded-full bg-pos animate-pulse" /> API live
      </span>
    );
  }
  if (erred) {
    return (
      <span
        className="inline-flex items-center gap-1 rounded-full border border-neg/30 bg-neg-soft px-2 py-0.5 text-[10px] font-semibold text-neg"
        title="Không kết nối được tới http://localhost:8000 — chạy `.\run.bat api`"
      >
        <span className="w-1.5 h-1.5 rounded-full bg-neg" /> API offline
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-line bg-bg-subtle px-2 py-0.5 text-[10px] font-semibold text-fg-faint">
      <span className="w-1.5 h-1.5 rounded-full bg-fg-dim animate-pulse" /> checking…
    </span>
  );
}
