import type { Metadata, Viewport } from "next";
import Link from "next/link";
import "./globals.css";
import { ServiceWorkerRegistration } from "@/components/ServiceWorkerRegistration";

export const metadata: Metadata = {
  title: "CausaSent — Vietnamese Review Analyzer",
  description:
    "Phân tích aspect-based sentiment + đề xuất hành động cho review TMĐT tiếng Việt.",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "CausaSent",
  },
};

export const viewport: Viewport = {
  themeColor: "#6366f1",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi">
      <body className="antialiased selection:bg-brand-500/20 selection:text-brand-700">
        <header className="border-b border-line bg-white/85 backdrop-blur-md sticky top-0 z-20 h-[46px] flex items-center">
          <div className="px-4 lg:px-6 w-full flex items-center gap-4">
            <Link href="/" className="flex items-center gap-2 group">
              <span
                className="w-6 h-6 rounded-md bg-gradient-to-br from-brand-500 to-pink-500 grid place-items-center text-[11px] font-black text-white shadow-glow"
                aria-hidden
              >
                C
              </span>
              <div className="leading-none">
                <div className="font-semibold text-[13px] tracking-tight text-fg">
                  CausaSent
                </div>
                <div className="text-[9px] uppercase tracking-[0.18em] text-fg-faint mt-0.5">
                  Vietnamese ABSA · v2
                </div>
              </div>
            </Link>
            <nav className="hidden md:flex items-center gap-0.5 text-[13px] ml-2">
              <Link href="/" className="btn-ghost">
                Phân tích
              </Link>
              <Link href="/dashboard" className="btn-ghost">
                Dashboard
              </Link>
            </nav>
            <div className="flex-1" />
            <a
              href="https://huggingface.co/datasets/Tamir39/causasent-ate-v2"
              target="_blank"
              rel="noreferrer"
              className="hidden sm:inline-flex btn-outline"
            >
              HF Dataset <span aria-hidden>↗</span>
            </a>
          </div>
        </header>
        <main className="px-4 lg:px-6 py-4 max-w-[1600px] mx-auto">{children}</main>
        <footer className="border-t border-line/80 mt-8 py-3 px-4 lg:px-6 max-w-[1600px] mx-auto text-[10.5px] text-fg-faint flex flex-wrap justify-between gap-2">
          <span>CausaSent v2 · PhoBERT-large + Gemini 2.5 Flash</span>
          <span className="text-fg-dim">
            7,066 reviews · 10,307 annotations · 91% IAA
          </span>
        </footer>
        <ServiceWorkerRegistration />
      </body>
    </html>
  );
}
