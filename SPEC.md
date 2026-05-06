# **Causal & Actionable Sentiment Analysis cho Review Tiếng Việt**

---

## 🎯 0. Mục tiêu

Xây dựng hệ thống nhận **review tiếng Việt** và sinh ra các bộ:

> **(aspect, sentiment, cause_span, action)**

Trong đó:

* **aspect**: khía cạnh (delivery, packaging, …)
* **sentiment**: {positive, negative, neutral}
* **cause_span**: đoạn text trong review gây ra sentiment
* **action**: hành động đề xuất (ngắn gọn, có thể thực thi)

---

## 📌 1. Dữ liệu (bắt buộc)

### 1.1 Domain & nguồn

* Chọn **1 domain** (khuyến nghị: e-commerce)
* Crawl từ:

  * Shopee / TikTok Shop / Google Maps / App Store
* Ngôn ngữ: **Tiếng Việt**

---

### 1.2 Schema dữ liệu chuẩn (lưu JSON)

```json
{
  "id": "string",
  "review": "string",
  "annotations": [
    {
      "aspect": "string",
      "sentiment": "positive | negative | neutral",
      "cause_text": "string",
      "cause_span": [start_char, end_char],
      "action": "string"
    }
  ]
}
```

---

### 1.3 Aspect taxonomy (cố định)

```text
delivery
packaging
product_quality
price
customer_service
usability
appearance
```

* Không được tạo aspect ngoài danh sách
* Mỗi annotation = 1 aspect duy nhất

---

### 1.4 Quy mô tối thiểu

* Tổng: ≥ **1000 reviews**
* Trung bình: ≥ **2 annotation/review**
* Sau khi expand → ~2000–3000 samples

---

### 1.5 Chia dữ liệu

```text
Train / Val / Test = 80 / 10 / 10
```

* Test set phải được **review thủ công kỹ**

---

## 🔥 2. Pipeline tạo label (bắt buộc – trọng tâm)

### 2.1 Crawl raw data

* Lưu:

  * review_text
  * metadata (optional)

---

### 2.2 Weak labeling bằng LLM

#### Prompt chuẩn:

```text
Extract aspects, sentiment, cause (exact span from text), and action from the Vietnamese review.

Rules:
- Aspect must be from predefined list
- Cause must be exact substring from review
- Action must be short and actionable

Return JSON array:
[
  {aspect, sentiment, cause, action}
]
```

---

### 2.3 Convert sang training format

* Map cause → **char span**
* Validate:

  * cause phải nằm trong review
  * sentiment hợp lệ

---

### 2.4 Human refinement (bắt buộc)

* Chọn ≥ **1000 samples**
* Sửa:

  * sai aspect
  * sai span
  * action vô nghĩa

👉 Đây là **gold dataset**

---

### 2.5 Dataset cuối

* train = pseudo + gold
* test = **chỉ gold**

---

## ⚙️ 3. Mô hình (KHÔNG benchmark – chọn tốt nhất)

---

# 🧩 3.1 Task 1 + 2 (gộp)

## 🎯 Bài toán:

Joint:

* Aspect detection
* Sentiment classification
* Cause extraction

---

## Formulation:

### Sequence tagging (BIO mở rộng)

Ví dụ:

```text
Ship  B-DEL-NEG B-CAUSE
chậm  I-DEL-NEG I-CAUSE
,     O         O
hộp   B-PACK-NEG B-CAUSE
móp   I-PACK-NEG I-CAUSE
```

---

## Model

```text
PhoBERT-large
 → Dropout
 → Linear (token classification)
```

---

## Label space

```text
B-DEL-NEG, I-DEL-NEG
B-PACK-NEG, I-PACK-NEG
...
B-CAUSE, I-CAUSE
O
```

---

## Loss

* CrossEntropy (token-level)

---

## Output

* aspect + sentiment (từ tag)
* cause_span (từ BIO CAUSE)

---

# 🧩 3.2 Task 3 — Action Generation

---

## 🎯 Bài toán:

Sinh action từ structured input

---

## Input format:

```text
aspect: delivery
sentiment: negative
cause: ship chậm
review: Ship chậm, hộp móp...
```

---

## Output:

```text
Cải thiện thời gian giao hàng
```

---

## Model

```text
mT5-base (fine-tune)
```

---

## Loss

* CrossEntropy (seq2seq)

---

# ⚙️ 4. Training

---

## 4.1 PhoBERT

```text
lr: 2e-5
batch_size: 16
epoch: 5
max_len: 128
optimizer: AdamW
```

---

## 4.2 mT5

```text
lr: 3e-5
batch_size: 8
epoch: 5
max_input_len: 128
max_output_len: 32
```

---

## 4.3 Training strategy

* Train Task 1+2 trước
* Sau đó generate input cho Task 3
* Fine-tune mT5

---

# 📊 5. Evaluation

---

## 5.1 Task 1+2

* Token-level F1
* Entity-level F1 (aspect + sentiment)

---

## 5.2 Task 3

* ROUGE-L
* Human evaluation:

  * hợp lý
  * actionable

---

# 🧠 6. Giải thích dùng Deep Learning (bắt buộc)

Phải nêu rõ:

---

## 6.1 Context understanding

* Review có nhiều cách diễn đạt khác nhau
* DL (Transformer) học semantic context tốt hơn rule-based

---

## 6.2 Sequence tagging

* Cause extraction là bài toán span-level
* Cần token-level modeling → chỉ DL làm hiệu quả

---

## 6.3 Text generation

* Action không thể rule-based
* Seq2Seq model sinh ngôn ngữ tự nhiên

---

👉 Kết luận viết trong báo cáo:

> Deep Learning enables unified modeling of contextual understanding, structured extraction, and conditional generation.

---

# 🎮 7. Demo (khuyến khích mạnh)

---

## Input:

```text
"Ship lâu nhưng đóng gói đẹp"
```

---

## Output:

```text
Delivery → Negative
Cause → Ship lâu
Action → Cải thiện vận chuyển

Packaging → Positive
Cause → đóng gói đẹp
Action → Giữ nguyên đóng gói
```

---

# 📦 8. Sản phẩm nộp

* GitHub:

  * data pipeline
  * training code
  * inference
* Dataset (JSON)
* Model checkpoint (HuggingFace)
* Slide + video demo

---

# ⚠️ 9. Yêu cầu quan trọng

* Không benchmark nhiều model
* Phải có:

  * fine-tune model
  * dataset tự xây dựng
* Focus:

  > **data + modeling + pipeline**

---

# 🎯 10. Mục tiêu cuối cùng

Hệ thống phải:

> Không chỉ phân tích cảm xúc
> → mà hiểu nguyên nhân
> → và đề xuất hành động cụ thể

---

# 🚀 Hướng dẫn triển khai cho Claude

Prompt:

```text
Implement the full pipeline:
1. Data loader (JSON schema above)
2. PhoBERT token classification model for joint aspect + sentiment + cause extraction
3. mT5 model for action generation
4. Training scripts
5. Evaluation scripts
6. Inference pipeline
7. Simple demo interface

Follow the exact schema and training configs.
```

---

# 🧠 Nhận xét cuối (rất thẳng)

Spec này:

* ✅ đúng trọng tâm Deep Learning
* ✅ giải quyết bài toán data (điểm mạnh nhất)
* ✅ có đủ độ “research + product”

👉 Nếu làm đúng:

> **Top bài trong lớp, không chỉ đủ điểm mà còn nổi bật**

---

Nếu bạn muốn bước tiếp:

* tôi có thể viết luôn **repo structure + code skeleton PyTorch**
* hoặc tạo **50 sample dataset chuẩn để bạn start ngay**

Chỉ cần nói:

> “code skeleton” hoặc “sample dataset”
