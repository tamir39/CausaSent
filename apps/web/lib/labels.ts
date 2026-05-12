import type { AspectCategory } from "./api";

export const ASPECT_LABEL_VI: Record<AspectCategory, string> = {
  delivery: "Giao hàng",
  packaging: "Đóng gói",
  product_quality: "Chất lượng SP",
  price: "Giá",
  customer_service: "Hỗ trợ KH",
  usability: "Trải nghiệm",
  appearance: "Hình thức",
};

export const SENTIMENT_COLOR = {
  positive: "#22c55e",
  negative: "#ef4444",
} as const;

export const PRIORITY_COLOR = {
  high: "#ef4444",
  medium: "#f59e0b",
  low: "#64748b",
} as const;

export const PRIORITY_RANK = { high: 0, medium: 1, low: 2 } as const;
