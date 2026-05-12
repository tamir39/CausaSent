# Báo cáo đồ án cuối kỳ — Vietnamese

`CausaSent_BaoCao.docx` là file báo cáo nộp cho trường, format Microsoft Word
chuẩn (A4, Times New Roman 13pt, line spacing 1.5).

**File này được sinh ra bằng script — KHÔNG sửa trực tiếp `.docx` rồi
expect re-generate sẽ giữ lại các sửa.** Nếu cần chỉnh nội dung, sửa
`scripts/build_report.py` rồi chạy lại.

## Cấu trúc

- Bìa (cần điền tên trường / khoa / MSSV / GVHD — xem mục tuỳ chỉnh)
- Lời cảm ơn
- Tóm tắt
- Mục lục (Word tự sinh — xem mục dưới)
- Chương 1: Giới thiệu
- Chương 2: Cơ sở lý thuyết (ABSA · PhoBERT · Supervised Contrastive · LLM)
- Chương 3: Xây dựng tập dữ liệu
- Chương 4: Phương pháp đề xuất
- Chương 5: Thực nghiệm và đánh giá (3 bảng kết quả)
- Chương 6: Kết luận và hướng phát triển
- Tài liệu tham khảo (9 entries)

## Cách tạo / cập nhật file

```bash
python scripts/build_report.py
```

Cần `python-docx` (đã `uv pip install` ở session trước):

```bash
uv pip install python-docx
```

Output: `report/CausaSent_BaoCao.docx` (~50 KB).

## Việc cần làm tay sau khi mở Word lần đầu

1. **Cập nhật mục lục**: mở file → click chuột phải vào dòng mục lục →
   chọn **"Update Field"** → chọn **"Update entire table"**. Word sẽ tự
   sinh mục lục với số trang.
2. **Cập nhật thông tin sinh viên trên trang bìa**:
   - `TRƯỜNG ĐẠI HỌC ……………` → tên trường
   - `KHOA ……………` → tên khoa
   - `Mã số sinh viên: ……………` → MSSV
   - `Lớp: ……………` → tên lớp
   - `Giảng viên hướng dẫn: ……………` → tên GVHD

   Có thể sửa bằng Word trực tiếp HOẶC sửa trong `scripts/build_report.py`
   ở phần `# COVER PAGE` rồi chạy lại script.

## Tuỳ biến nội dung

Tất cả nội dung nằm trong `scripts/build_report.py`. Tìm chapter cần sửa
(các phần được đánh dấu `# CHƯƠNG N`) và thay đổi text trong các lời gọi:

- `add_para(doc, "...")` — đoạn văn thường
- `add_bullet(doc, "...")` — gạch đầu dòng
- `add_numbered(doc, "...")` — đánh số
- `add_section(doc, "X.Y. Title")` — heading mục
- `add_subsection(doc, "X.Y.Z. Title")` — heading mục con
- `add_table(doc, headers, rows)` — bảng

Sau khi sửa, chạy lại `python scripts/build_report.py`.

## Thêm phụ lục (nếu muốn)

Có thể thêm phụ lục với:
- Screenshots của ứng dụng PWA (`apps/web`)
- Confusion matrix
- Code snippets quan trọng
- Bảng dữ liệu chi tiết

Insert trước hoặc sau phần "TÀI LIỆU THAM KHẢO" trong script. Ví dụ:

```python
add_page_break(doc)
p = doc.add_paragraph(style="Heading 1")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("PHỤ LỤC A — SCREENSHOTS ỨNG DỤNG")
_set_font(r, size=Pt(16), bold=True)
doc.add_picture("path/to/screenshot.png", width=Cm(14))
add_caption(doc, "Hình A.1. Trang chủ ứng dụng CausaSent.")
```
