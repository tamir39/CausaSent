// Pre-baked AnalyzeResult so the dashboard can be inspected without a backend.
// Kept in sync (by hand) with the shape produced by src/inference/aggregate.py
// and src/inference/action_llm.py — see lib/api.ts for the type contract.

import type { AnalyzeResult } from "./api";

export const SAMPLE_RESULT: AnalyzeResult = {
  n_reviews: 6,
  summary: {
    n_reviews: 6,
    n_reviews_with_tuples: 6,
    n_tuples: 14,
    sentiment_overall: { positive: 8, negative: 6 },
    aspects: {
      delivery: {
        aspect_category: "delivery",
        total_mentions: 4,
        n_reviews: 4,
        positive: 2,
        negative: 2,
        negative_ratio: 0.5,
        cells: {
          positive: {
            aspect_category: "delivery",
            sentiment: "positive",
            count: 2,
            top_terms: [["giao nhanh", 1], ["giao hàng", 1]],
          },
          negative: {
            aspect_category: "delivery",
            sentiment: "negative",
            count: 2,
            top_terms: [["ship", 1], ["giao chậm", 1]],
          },
        },
      },
      packaging: {
        aspect_category: "packaging",
        total_mentions: 3,
        n_reviews: 3,
        positive: 2,
        negative: 1,
        negative_ratio: 0.333,
        cells: {
          positive: {
            aspect_category: "packaging",
            sentiment: "positive",
            count: 2,
            top_terms: [["đóng gói", 2]],
          },
          negative: {
            aspect_category: "packaging",
            sentiment: "negative",
            count: 1,
            top_terms: [["hộp móp", 1]],
          },
        },
      },
      product_quality: {
        aspect_category: "product_quality",
        total_mentions: 3,
        n_reviews: 3,
        positive: 2,
        negative: 1,
        negative_ratio: 0.333,
        cells: {
          positive: {
            aspect_category: "product_quality",
            sentiment: "positive",
            count: 2,
            top_terms: [["chất lượng tốt", 1], ["sản phẩm", 1]],
          },
          negative: {
            aspect_category: "product_quality",
            sentiment: "negative",
            count: 1,
            top_terms: [["sản phẩm", 1]],
          },
        },
      },
      price: {
        aspect_category: "price",
        total_mentions: 2,
        n_reviews: 2,
        positive: 1,
        negative: 1,
        negative_ratio: 0.5,
        cells: {
          positive: {
            aspect_category: "price",
            sentiment: "positive",
            count: 1,
            top_terms: [["giá hợp lý", 1]],
          },
          negative: {
            aspect_category: "price",
            sentiment: "negative",
            count: 1,
            top_terms: [["giá hơi cao", 1]],
          },
        },
      },
      customer_service: {
        aspect_category: "customer_service",
        total_mentions: 2,
        n_reviews: 2,
        positive: 1,
        negative: 1,
        negative_ratio: 0.5,
        cells: {
          positive: {
            aspect_category: "customer_service",
            sentiment: "positive",
            count: 1,
            top_terms: [["shop tư vấn", 1]],
          },
          negative: {
            aspect_category: "customer_service",
            sentiment: "negative",
            count: 1,
            top_terms: [["nhân viên", 1]],
          },
        },
      },
    },
  },
  actions: [
    {
      aspect_category: "delivery",
      sentiment: "negative",
      priority: "medium",
      action: "Cải thiện tốc độ giao hàng cho đơn ship chậm.",
      evidence_terms: ["ship", "giao chậm"],
    },
    {
      aspect_category: "packaging",
      sentiment: "negative",
      priority: "low",
      action: "Khắc phục lỗi đóng gói khiến hộp móp méo khi giao.",
      evidence_terms: ["hộp móp"],
    },
    {
      aspect_category: "customer_service",
      sentiment: "negative",
      priority: "low",
      action: "Tập huấn nhân viên giao tiếp lịch sự, nhẹ nhàng.",
      evidence_terms: ["nhân viên"],
    },
    {
      aspect_category: "price",
      sentiment: "negative",
      priority: "low",
      action: "Xem xét lại định giá để bớt cảm giác chát so với chất lượng.",
      evidence_terms: ["giá hơi cao"],
    },
    {
      aspect_category: "product_quality",
      sentiment: "positive",
      priority: "medium",
      action: "Tiếp tục đầu tư vào chất lượng vải / hoàn thiện sản phẩm.",
      evidence_terms: ["chất lượng tốt", "sản phẩm"],
    },
    {
      aspect_category: "packaging",
      sentiment: "positive",
      priority: "medium",
      action: "Giữ phong cách đóng gói hiện tại — khách khen nhiều.",
      evidence_terms: ["đóng gói"],
    },
    {
      aspect_category: "delivery",
      sentiment: "positive",
      priority: "low",
      action: "Duy trì tốc độ giao hàng nhanh ở các đơn nội thành.",
      evidence_terms: ["giao nhanh", "giao hàng"],
    },
    {
      aspect_category: "customer_service",
      sentiment: "positive",
      priority: "low",
      action: "Khen thưởng đội tư vấn — khách phản hồi tốt.",
      evidence_terms: ["shop tư vấn"],
    },
    {
      aspect_category: "price",
      sentiment: "positive",
      priority: "low",
      action: "Giữ mức giá hiện tại — khách đánh giá hợp lý.",
      evidence_terms: ["giá hợp lý"],
    },
  ],
  per_review: [],
};
