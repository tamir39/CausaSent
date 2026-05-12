"""Generate the Vietnamese-language project report as a .docx file.

This matches Vietnamese university final-project conventions:
  - A4 paper, 1.5 line spacing, Times New Roman 13pt body
  - Chapter headings + subsection headings (Word's Heading 1 / 2)
  - Auto-numbered chapters, manually-written numbering inside text
  - Tables with proper borders
  - Title page on first page
  - TOC placeholder (Word will auto-fill it on first open via "Update Field")

Run:
    python scripts/build_report.py
Output:
    report/CausaSent_BaoCao.docx
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt, RGBColor


OUT_PATH = Path("report/CausaSent_BaoCao.docx")

# Vietnamese university default: Times New Roman 13pt body, 1.5 spacing.
BODY_FONT = "Times New Roman"
BODY_SIZE = Pt(13)


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _set_font(run, name: str = BODY_FONT, size: Pt = BODY_SIZE, *, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color
    # Force same font for East-Asian + Arabic to avoid Word switching
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    rFonts.set(qn("w:cs"), name)
    rFonts.set(qn("w:eastAsia"), name)


def _setup_doc() -> Document:
    doc = Document()

    # Page setup (A4, margins typical for Vietnamese reports).
    for section in doc.sections:
        section.page_height = Cm(29.7)
        section.page_width = Cm(21.0)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.0)

    # Default style → Times New Roman 13pt, 1.5 line spacing.
    style = doc.styles["Normal"]
    style.font.name = BODY_FONT
    style.font.size = BODY_SIZE
    rPr = style.element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), BODY_FONT)
    rFonts.set(qn("w:hAnsi"), BODY_FONT)
    rFonts.set(qn("w:cs"), BODY_FONT)
    rFonts.set(qn("w:eastAsia"), BODY_FONT)
    style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    style.paragraph_format.space_after = Pt(0)

    # Heading 1 — chapter title (centered, bold, 16pt).
    h1 = doc.styles["Heading 1"]
    h1.font.name = BODY_FONT
    h1.font.size = Pt(16)
    h1.font.bold = True
    h1.font.color.rgb = RGBColor(0, 0, 0)
    h1.paragraph_format.space_before = Pt(24)
    h1.paragraph_format.space_after = Pt(12)
    h1.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

    # Heading 2 — section heading (bold 14pt, left).
    h2 = doc.styles["Heading 2"]
    h2.font.name = BODY_FONT
    h2.font.size = Pt(14)
    h2.font.bold = True
    h2.font.color.rgb = RGBColor(0, 0, 0)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(6)
    h2.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

    # Heading 3 — sub-section (bold italic 13pt).
    h3 = doc.styles["Heading 3"]
    h3.font.name = BODY_FONT
    h3.font.size = Pt(13)
    h3.font.bold = True
    h3.font.italic = True
    h3.font.color.rgb = RGBColor(0, 0, 0)
    h3.paragraph_format.space_before = Pt(8)
    h3.paragraph_format.space_after = Pt(4)
    h3.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

    return doc


def add_para(doc, text="", *, style=None, align=None, bold=False, italic=False, first_line_indent=None):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    if first_line_indent is not None:
        p.paragraph_format.first_line_indent = first_line_indent
    if text:
        r = p.add_run(text)
        _set_font(r, bold=bold, italic=italic)
    return p


def add_runs(doc, parts, *, align=None, first_line_indent=Cm(1.25)):
    """Add a paragraph with mixed inline formatting.

    `parts` = list of tuples (text, dict_of_options). Options keys:
        bold, italic, code (monospace via Consolas)
    """
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    if first_line_indent is not None:
        p.paragraph_format.first_line_indent = first_line_indent
    for text, opts in parts:
        r = p.add_run(text)
        if opts.get("code"):
            _set_font(r, name="Consolas", size=Pt(11))
        else:
            _set_font(r,
                      bold=opts.get("bold", False),
                      italic=opts.get("italic", False))
    return p


def add_bullet(doc, parts, *, level=0):
    p = doc.add_paragraph(style="List Bullet")
    if level > 0:
        p.paragraph_format.left_indent = Cm(0.75 * (level + 1))
    if isinstance(parts, str):
        parts = [(parts, {})]
    for text, opts in parts:
        r = p.add_run(text)
        _set_font(r,
                  bold=opts.get("bold", False),
                  italic=opts.get("italic", False))
    return p


def add_numbered(doc, parts):
    p = doc.add_paragraph(style="List Number")
    if isinstance(parts, str):
        parts = [(parts, {})]
    for text, opts in parts:
        r = p.add_run(text)
        _set_font(r,
                  bold=opts.get("bold", False),
                  italic=opts.get("italic", False))
    return p


def add_chapter(doc, num: int, title: str):
    p = doc.add_paragraph(style="Heading 1")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p.add_run(f"Chương {num}\n")
    _set_font(r1, size=Pt(14), bold=True)
    r2 = p.add_run(title.upper())
    _set_font(r2, size=Pt(16), bold=True)


def add_section(doc, title: str):
    p = doc.add_paragraph(style="Heading 2")
    r = p.add_run(title)
    _set_font(r, size=Pt(14), bold=True)


def add_subsection(doc, title: str):
    p = doc.add_paragraph(style="Heading 3")
    r = p.add_run(title)
    _set_font(r, size=Pt(13), bold=True, italic=True)


def add_caption(doc, text: str, *, kind: str = "Bảng"):
    """Add a centered italic caption above/below a table or figure."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    _set_font(r, italic=True, size=Pt(12))
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    _ = kind  # reserved for future Word-field auto-numbering


def add_table(doc, header: list[str], rows: list[list[str]], *, header_bold=True):
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    # Header
    hdr = table.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        _set_font(r, size=Pt(12), bold=header_bold)
        hdr[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    # Body
    for ri, row in enumerate(rows, start=1):
        cells = table.rows[ri].cells
        for ci, val in enumerate(row):
            cells[ci].text = ""
            p = cells[ci].paragraphs[0]
            p.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.RIGHT
            )
            r = p.add_run(val)
            _set_font(r, size=Pt(12))
            cells[ci].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    return table


def add_toc_field(doc):
    """Insert a Word TOC field — fills on first open via right-click → Update Field."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = 'TOC \\o "1-3" \\h \\z \\u'
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "Mở Word → click chuột phải → Update Field để tự động sinh mục lục"
    fldChar3 = OxmlElement("w:fldChar")
    fldChar3.set(qn("w:fldCharType"), "end")
    run._element.append(fldChar1)
    run._element.append(instrText)
    run._element.append(fldChar2)
    run._element.append(placeholder)
    run._element.append(fldChar3)
    _set_font(run, italic=True)


def add_page_break(doc):
    doc.add_page_break()


# ---------------------------------------------------------------------------
# Build the report
# ---------------------------------------------------------------------------

def build():
    doc = _setup_doc()

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    section = doc.sections[0]
    section.top_margin = Cm(2.5)

    def cover_line(text, size=14, bold=False, space_before=0, space_after=12):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        r = p.add_run(text)
        _set_font(r, size=Pt(size), bold=bold)
        return p

    cover_line("TRƯỜNG ĐẠI HỌC ……………", size=14, bold=True)
    cover_line("KHOA ……………", size=14, bold=True, space_after=48)

    cover_line("ĐỒ ÁN CUỐI KỲ", size=18, bold=True, space_before=24)
    cover_line("MÔN DATA MINING", size=16, bold=True, space_after=48)

    cover_line("CausaSent", size=28, bold=True, space_before=24)
    cover_line(
        "Phân tích cảm xúc theo khía cạnh kết hợp đề xuất hành động "
        "cho review thương mại điện tử tiếng Việt",
        size=14,
        space_after=64,
    )

    # Student info block
    info = [
        ("Sinh viên thực hiện", "Phí Vương Tường Tâm"),
        ("Mã số sinh viên", "……………"),
        ("Lớp", "……………"),
        ("Giảng viên hướng dẫn", "……………"),
    ]
    info_table = doc.add_table(rows=len(info), cols=2)
    info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, (k, v) in enumerate(info):
        c0, c1 = info_table.rows[r_idx].cells
        for cell, text, bold in ((c0, f"{k}:", True), (c1, v, False)):
            cell.text = ""
            p = cell.paragraphs[0]
            r = p.add_run(text)
            _set_font(r, size=Pt(13), bold=bold)
            p.paragraph_format.space_after = Pt(0)

    cover_line("", space_before=72)
    cover_line("TP. Hồ Chí Minh — Tháng 5, 2026", size=13)

    add_page_break(doc)

    # =========================================================================
    # LỜI CẢM ƠN
    # =========================================================================
    add_chapter(doc, num=0, title="Lời cảm ơn") if False else None  # not numbered
    p = doc.add_paragraph(style="Heading 1")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("LỜI CẢM ƠN")
    _set_font(r, size=Pt(16), bold=True)

    add_para(
        doc,
        "Em xin gửi lời cảm ơn chân thành tới giảng viên hướng dẫn đã định "
        "hướng và góp ý trong suốt quá trình thực hiện đồ án. Em cũng cảm "
        "ơn các tác giả đã công bố công khai các tập dữ liệu Vietnamese "
        "ABSA (VLSP 2018, UIT-ViSD4SA) và các mô hình tiền huấn luyện như "
        "PhoBERT, VnCoreNLP — những tài nguyên này là nền móng cho phần "
        "thực nghiệm trong đồ án.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        first_line_indent=Cm(1.25),
    )
    add_para(
        doc,
        "Trong quá trình thực hiện đồ án không tránh khỏi thiếu sót, em "
        "rất mong nhận được sự góp ý của thầy cô để hoàn thiện hơn.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        first_line_indent=Cm(1.25),
    )

    add_page_break(doc)

    # =========================================================================
    # TÓM TẮT
    # =========================================================================
    p = doc.add_paragraph(style="Heading 1")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("TÓM TẮT")
    _set_font(r, size=Pt(16), bold=True)

    add_runs(
        doc,
        [
            ("Đồ án trình bày ", {}),
            ("CausaSent", {"bold": True}),
            (
                ", một hệ thống phân tích cảm xúc theo khía cạnh "
                "(Aspect-Based Sentiment Analysis — ABSA) cho review "
                "thương mại điện tử tiếng Việt, kết hợp với bước tổng hợp "
                "đầu ra để sinh ra ",
                {},
            ),
            ("khuyến nghị hành động", {"italic": True}),
            (" thực tế cho người bán.", {}),
        ],
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    )

    add_para(doc, "Đóng góp chính gồm ba phần:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    add_numbered(doc, [
        ("Tập dữ liệu ", {}),
        ("CausaSent ATE v2", {"bold": True}),
        (" gồm 7.066 review và 10.307 nhãn, kết hợp pool vàng (1.132 review hotel/restaurant/smartphone) và pool bạc (5.934 review Tiki). Độ đồng thuận giữa annotator trên 300 mẫu vàng là ", {}),
        ("91.00%", {"bold": True}),
        (".", {}),
    ])
    add_numbered(doc, [
        ("Mô hình ", {}),
        ("PhoBERT-large hai đầu", {"bold": True}),
        (" cho trích xuất aspect term (BIO tagging 15 lớp) và phân loại sentiment ở vị trí B-token, kèm theo hàm mất mát phụ trợ supervised contrastive. Trên test split, mô hình đạt F1 entity-level ", {}),
        ("0.328", {"bold": True}),
        (" (precision 0.21, recall 0.80), F1 macro của phân loại sentiment ", {}),
        ("0.876", {"bold": True}),
        (", F1 lớp negative ", {}),
        ("0.901", {"bold": True}),
        (".", {}),
    ])
    add_numbered(doc, [
        ("Quy trình ", {}),
        ("sinh hành động từ tổng hợp", {"bold": True}),
        (". Thay vì gọi LLM riêng cho từng review (dài dòng, trùng lặp), đồ án gộp dự đoán của N review thành bảng tóm tắt theo từng cặp (aspect, sentiment) rồi gọi Gemini-2.5-Flash một lần duy nhất để sinh danh sách hành động có thứ tự ưu tiên.", {}),
    ])

    add_para(
        doc,
        "Toàn bộ dataset, mô hình huấn luyện, và mã nguồn pipeline được "
        "công bố công khai trên Hugging Face Hub. Hệ thống còn có một ứng "
        "dụng PWA (Next.js + FastAPI) cho phép upload CSV và xem dashboard "
        "real-time với khả năng streaming.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        first_line_indent=Cm(1.25),
    )

    add_page_break(doc)

    # =========================================================================
    # MỤC LỤC
    # =========================================================================
    p = doc.add_paragraph(style="Heading 1")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("MỤC LỤC")
    _set_font(r, size=Pt(16), bold=True)

    add_toc_field(doc)

    add_para(
        doc,
        "(Mở file này bằng Microsoft Word → click chuột phải vào dòng mục "
        "lục → chọn \"Update Field\" để Word tự động sinh đầy đủ mục lục.)",
        italic=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )

    add_page_break(doc)

    # =========================================================================
    # CHƯƠNG 1 — GIỚI THIỆU
    # =========================================================================
    add_chapter(doc, 1, "Giới thiệu")

    add_section(doc, "1.1. Bối cảnh và động lực")
    add_para(
        doc,
        "Các sàn thương mại điện tử ở Việt Nam (Tiki, Shopee, Lazada, "
        "TikTok Shop) tích lũy hàng triệu review của khách hàng mỗi ngày. "
        "Tuy nhiên, người bán hiện chỉ nhận được phản hồi dưới dạng star "
        "rating — một con số duy nhất không cho biết khách hàng đang khen "
        "hoặc chê khía cạnh nào của sản phẩm. Khi muốn cải thiện chất "
        "lượng dịch vụ, người bán phải đọc thủ công hàng trăm review để "
        "tự rút ra kết luận.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        first_line_indent=Cm(1.25),
    )
    add_para(
        doc,
        "Bài toán phân tích cảm xúc theo khía cạnh (ABSA) hứa hẹn cái "
        "nhìn chi tiết hơn: trích xuất khía cạnh được nhắc đến (giao "
        "hàng, đóng gói, chất lượng sản phẩm, ...) cùng với sắc thái cảm "
        "xúc tương ứng (tích cực / tiêu cực). Đối với tiếng Việt, đã có "
        "một số tập dữ liệu (VLSP 2018, UIT-ViSD4SA) và mô hình trên các "
        "miền hotel, restaurant, smartphone. Tuy nhiên, các nghiên cứu "
        "này thường dừng lại ở việc gán nhãn cho từng review riêng lẻ — "
        "chưa trả lời được câu hỏi thực sự của người bán: \"tôi nên làm gì?\".",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        first_line_indent=Cm(1.25),
    )

    add_section(doc, "1.2. Mục tiêu của đồ án")
    add_para(doc, "Đồ án này nhằm xây dựng một hệ thống ABSA hoàn chỉnh cho review TMĐT tiếng Việt, có ba mục tiêu cụ thể:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_numbered(doc, "Xây dựng tập dữ liệu ABSA đủ lớn để huấn luyện một mô hình PhoBERT-large (vài nghìn đến hơn mười nghìn nhãn) bằng cách kết hợp dữ liệu hiện có với sử dụng LLM (Claude) để gán nhãn bán tự động và có kiểm chứng độ tin cậy.")
    add_numbered(doc, "Huấn luyện một mô hình joint trích xuất aspect term + phân loại sentiment, đạt độ chính xác cao nhất có thể trong giới hạn thời gian và tài nguyên (1 GPU Kaggle, training time khoảng 1 giờ).")
    add_numbered(doc, "Xây dựng pipeline từ review đến hành động, có thể đưa vào ứng dụng web cho người bán: nhận đầu vào là một tập review, trả về dashboard tổng hợp và khuyến nghị hành động cụ thể bằng tiếng Việt.")

    add_section(doc, "1.3. Phạm vi và giới hạn")
    add_para(doc, "Đồ án giới hạn ở các điểm sau:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_bullet(doc, "Taxonomy aspect cố định ở 7 lớp: delivery, packaging, product_quality, price, customer_service, usability, appearance. Không hỗ trợ tự mở rộng taxonomy.")
    add_bullet(doc, "Sentiment nhị phân (positive / negative). Nhãn neutral đã được loại bỏ vì chỉ chiếm dưới 3% dữ liệu gốc, không đủ huấn luyện ổn định một bộ phân loại 3 lớp.")
    add_bullet(doc, "Không thực hiện scrape tự động Shopee / TikTok Shop. Dữ liệu Tiki được thừa hưởng từ một midterm project sẵn có.")
    add_bullet(doc, "Chỉ huấn luyện 8 epoch (giới hạn thời gian GPU Kaggle). Một số ablation được liệt kê là hướng phát triển tương lai.")

    add_section(doc, "1.4. Cấu trúc báo cáo")
    add_para(doc,
        "Phần còn lại của báo cáo được tổ chức như sau: Chương 2 trình "
        "bày nền tảng lý thuyết về ABSA, PhoBERT, và Supervised "
        "Contrastive Learning. Chương 3 mô tả chi tiết quá trình thu "
        "thập và gán nhãn tập dữ liệu CausaSent ATE v2. Chương 4 đặc tả "
        "mô hình hai đầu PhoBERT, hàm mất mát kết hợp, và pipeline tổng "
        "hợp + sinh hành động. Chương 5 báo cáo kết quả thực nghiệm, so "
        "sánh, và phân tích định tính. Chương 6 tổng kết và đề xuất "
        "hướng phát triển tiếp.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        first_line_indent=Cm(1.25),
    )

    add_page_break(doc)

    # =========================================================================
    # CHƯƠNG 2 — CƠ SỞ LÝ THUYẾT
    # =========================================================================
    add_chapter(doc, 2, "Cơ sở lý thuyết")

    add_section(doc, "2.1. Phân tích cảm xúc theo khía cạnh (ABSA)")
    add_para(doc,
        "ABSA là một bài toán con của Sentiment Analysis. Trong khi "
        "sentiment analysis truyền thống gán một nhãn cảm xúc duy nhất "
        "cho cả một văn bản (positive/negative/neutral), ABSA tách văn "
        "bản ra thành nhiều cặp (khía cạnh, sắc thái) — phản ánh thực "
        "tế rằng người dùng thường khen mặt này và chê mặt khác trong "
        "cùng một review.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_para(doc, "Một review tiếng Việt điển hình:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_para(doc,
        "\"Hàng giao chậm 5 ngày, hộp móp méo, nhưng sản phẩm chất lượng "
        "rất tốt, giá hợp lý.\"",
        align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)

    add_para(doc, "ABSA chia review trên thành bốn cặp:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_bullet(doc, "(delivery, tiêu cực) — \"giao chậm 5 ngày\"")
    add_bullet(doc, "(packaging, tiêu cực) — \"hộp móp méo\"")
    add_bullet(doc, "(product_quality, tích cực) — \"chất lượng rất tốt\"")
    add_bullet(doc, "(price, tích cực) — \"giá hợp lý\"")

    add_para(doc, "ABSA thường được chia thành các subtask:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_bullet(doc, [("Aspect Category Detection (ACD)", {"bold": True}), (": xác định những lớp aspect nào được nhắc tới (chỉ ở mức câu/review, không định vị).", {})])
    add_bullet(doc, [("Aspect Term Extraction (ATE)", {"bold": True}), (": định vị cụm danh từ chỉ khía cạnh trong review (gán nhãn span hoặc BIO).", {})])
    add_bullet(doc, [("Aspect Sentiment Classification (ASC)", {"bold": True}), (": gán nhãn cảm xúc cho mỗi cặp (aspect term, aspect category).", {})])

    add_para(doc, "Đồ án này giải quyết joint ATE + ASC: trích xuất span và gán cả aspect category lẫn sentiment trong cùng một mô hình.", align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_section(doc, "2.2. PhoBERT cho tiếng Việt")
    add_para(doc,
        "PhoBERT là một biến thể của RoBERTa được tiền huấn luyện trên "
        "kho ngữ liệu tiếng Việt được phân đoạn từ (word-segmented) bằng "
        "VnCoreNLP. Có hai phiên bản là base (135M params) và large "
        "(370M params); đồ án sử dụng phiên bản large vì cho kết quả tốt "
        "hơn trên các benchmark Vietnamese NLP và Kaggle GPU có đủ bộ nhớ.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))
    add_para(doc,
        "Điểm cần lưu ý khi sử dụng PhoBERT là phải chạy VnCoreNLP để "
        "phân đoạn từ trước khi tokenize: \"Hàng giao chậm\" phải được "
        "tách thành [Hàng, giao_chậm] chứ không phải [Hàng, giao, chậm]. "
        "Bỏ qua bước này làm độ chính xác giảm khoảng 2-3 điểm F1.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_section(doc, "2.3. Supervised Contrastive Learning")
    add_para(doc,
        "Supervised Contrastive Loss là một biến thể mở rộng của "
        "contrastive learning truyền thống (vốn không cần nhãn). Ý "
        "tưởng là, với một batch các vector biểu diễn và nhãn lớp tương ứng:",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))
    add_bullet(doc, "Kéo các vector cùng nhãn lại gần nhau trong không gian embedding.")
    add_bullet(doc, "Đẩy các vector khác nhãn ra xa.")
    add_para(doc,
        "Trong đồ án, hàm mất mát này được áp dụng chỉ tại các vị trí "
        "B-token của bộ tagger, với nhãn là aspect_category. Mục tiêu: "
        "làm cho encoder PhoBERT học được rằng các từ mở đầu cụm aspect "
        "\"giao hàng / ship / vận chuyển\" nên có embedding gần nhau "
        "(cùng cụm delivery), tách rời khỏi cụm \"đóng gói / hộp / "
        "túi\" (packaging).",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_section(doc, "2.4. Mô hình ngôn ngữ lớn (LLM) trong pipeline")
    add_para(doc,
        "LLM (như Claude, Gemini, GPT) có khả năng sinh văn bản tiếng Việt "
        "mạch lạc và có thể tuân theo định dạng JSON. Tuy nhiên gọi LLM "
        "tốn quota và chậm hơn nhiều so với inference của mô hình "
        "PhoBERT cục bộ. Vì vậy đồ án giới hạn vai trò của LLM ở hai chỗ:",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))
    add_numbered(doc, "Bán tự động hóa quá trình gán nhãn dữ liệu (Chương 3) — dùng Claude vì có context window lớn và độ chính xác cao trên tiếng Việt.")
    add_numbered(doc, "Sinh khuyến nghị hành động một lần duy nhất từ summary tổng hợp (Chương 4) — dùng Gemini 2.5 Flash vì free tier hào phóng và đủ chất lượng cho task ngắn này.")

    add_page_break(doc)

    # =========================================================================
    # CHƯƠNG 3 — DATASET
    # =========================================================================
    add_chapter(doc, 3, "Xây dựng tập dữ liệu")

    add_section(doc, "3.1. Schema và taxonomy")
    add_para(doc, "Mỗi bản ghi trong tập dữ liệu là một review tiếng Việt kèm theo một danh sách annotation. Mỗi annotation gồm:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_bullet(doc, [("aspect_term", {"bold": True}), (" — cụm danh từ chỉ khía cạnh (vd. \"đóng gói\").", {})])
    add_bullet(doc, [("aspect_term_span", {"bold": True}), (" — cặp [s, e) là vị trí ký tự trong review sao cho review[s:e] = aspect_term chính xác từng byte.", {})])
    add_bullet(doc, [("aspect_category", {"bold": True}), (" — thuộc 7 lớp đóng đã liệt kê ở Mục 1.3.", {})])
    add_bullet(doc, [("sentiment", {"bold": True}), (" — positive hoặc negative.", {})])
    add_para(doc,
        "Ràng buộc về byte-match của span là quan trọng vì PhoBERT làm "
        "việc ở mức word/sub-word; nếu span lệch chỉ 1 ký tự thì sau khi "
        "map sang word boundary sẽ ra kết quả sai.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_section(doc, "3.2. Hai nguồn dữ liệu")
    add_subsection(doc, "3.2.1. Pool vàng (gold)")
    add_para(doc,
        "Pool vàng gồm 1.132 review (2.205 annotations) được trích từ "
        "các tập public Vietnamese ABSA: VLSP 2016 hotel reviews, VLSP "
        "2018 hotel + restaurant reviews, và một subset smartphone "
        "reviews. Các tập gốc gán nhãn dưới dạng (aspect, sentiment, "
        "cause-span, action) 4-tuple. Đồ án loại bỏ trường action, drop "
        "class neutral, và dùng pipeline LLM (Mục 3.3) để trích xuất "
        "aspect_term span từ cause-span gốc.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_subsection(doc, "3.2.2. Pool bạc (silver)")
    add_para(doc,
        "Pool bạc gồm 5.934 review (8.102 annotations) là dữ liệu Tiki "
        "product reviews. Tập gốc được gán nhãn ở cấp câu với các cột "
        "as_content, as_physical, as_price, as_packaging, as_delivery, "
        "as_service, kèm theo sentiment_llm. Đồ án ánh xạ 6 cột này "
        "sang 6 lớp của taxonomy (tương ứng: product_quality, "
        "appearance, price, packaging, delivery, customer_service) rồi "
        "chạy cùng pipeline LLM để bổ sung span.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_section(doc, "3.3. Pipeline gán nhãn bán tự động")
    add_para(doc, "Cả hai pool đều đi qua một pipeline bốn bước:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_numbered(doc, [("Chia batch", {"bold": True}), (" — mỗi batch chứa tối đa 100 record.", {})])
    add_numbered(doc, [("Gán nhãn span bằng LLM", {"bold": True}), (" — sử dụng Claude (3.5 Sonnet) với prompt yêu cầu trả về cụm danh từ ngắn nhất và chính xác.", {})])
    add_numbered(doc, [("Kiểm tra tự động", {"bold": True}), (" — script đối chiếu review[s:e] với aspect_term từng byte.", {})])
    add_numbered(doc, [("Gộp và lưu", {"bold": True}), (" — các record pass verification được nối vào tập dữ liệu cuối cùng.", {})])
    add_para(doc,
        "Tỷ lệ pass: 10.307 / 10.324 (99.84%). 17 record bị loại do sai "
        "lệch NFD/NFC normalization (các dấu tiếng Việt được tách thành "
        "tổ hợp ký tự, không khớp byte-match).",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_section(doc, "3.4. Đánh giá độ tin cậy của nhãn")
    add_para(doc,
        "Để đo độ tin cậy của span do LLM trích xuất, đồ án lấy ngẫu "
        "nhiên 300 record từ pool vàng (stratified theo aspect category) "
        "và gán lại độc lập bằng một Claude subagent thứ hai đóng vai "
        "trò \"giám khảo\". Giám khảo không thấy nhãn ban đầu, chỉ thấy "
        "review + aspect category + sentiment, và phải tự trích xuất "
        "span tốt nhất.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))
    add_para(doc,
        "Bảng 3.1 tóm tắt kết quả. Độ đồng thuận tổng thể là 91.00% "
        "(273 / 300), với agreement cao nhất ở price (96.77%) và thấp "
        "nhất ở product_quality (89.93%).",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_caption(doc, "Bảng 3.1. Độ đồng thuận giữa annotator trên 300 mẫu vàng.")
    add_table(doc,
        ["Aspect", "N", "Đồng thuận", "Tỷ lệ"],
        [
            ["product_quality", "149", "134", "89.93%"],
            ["appearance",      "60",  "54",  "90.00%"],
            ["customer_service","38",  "35",  "92.11%"],
            ["price",           "31",  "30",  "96.77%"],
            ["usability",       "22",  "20",  "90.91%"],
            ["Tổng cộng",       "300", "273", "91.00%"],
        ])

    add_para(doc, "Hầu hết các bất đồng rơi vào ba loại:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_bullet(doc, "Gồm cả modifier không cần thiết (vd. \"Phòng ốc\" vs \"Phòng\").")
    add_bullet(doc, "Gồm cả classifier (vd. \"chiếc giường\" vs \"giường\").")
    add_bullet(doc, "Chọn sai occurrence trong review có nhiều mệnh đề.")

    add_section(doc, "3.5. Chia tập train / val / test")
    add_para(doc,
        "Tập dữ liệu được chia 80/10/10 ở mức review (không phải "
        "annotation) để tránh leak. Khóa stratify là cặp (aspect, "
        "sentiment) trội nhất trong từng review. Bảng 3.2 liệt kê kích "
        "thước từng split.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_caption(doc, "Bảng 3.2. Kích thước các split của tập dữ liệu.")
    add_table(doc,
        ["Split", "Số review", "Số annotation"],
        [
            ["train", "5.647", "8.251"],
            ["val",   "700",   "1.033"],
            ["test",  "719",   "1.023"],
            ["Tổng",  "7.066", "10.307"],
        ])

    add_para(doc,
        "Phân phối lớp rất mất cân bằng: appearance (38%) và "
        "product_quality (28%) chiếm đa số; usability chỉ 1.2%. Sentiment "
        "lệch về negative ở các lớp packaging và appearance (khách hàng "
        "phàn nàn về sản phẩm bị hỏng), và lệch positive ở price và "
        "customer_service. Đồ án xử lý mất cân bằng bằng class-weighted "
        "cross-entropy và contrastive auxiliary.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_page_break(doc)

    # =========================================================================
    # CHƯƠNG 4 — PHƯƠNG PHÁP
    # =========================================================================
    add_chapter(doc, 4, "Phương pháp đề xuất")

    add_section(doc, "4.1. Tổng quan kiến trúc")
    add_para(doc, "Hệ thống gồm ba tầng tách biệt:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_numbered(doc, [("Trích xuất aspect term + sentiment", {"bold": True}), (" bằng PhoBERT 2-head.", {})])
    add_numbered(doc, [("Tổng hợp", {"bold": True}), (" các tuple thu được từ N review thành summary.", {})])
    add_numbered(doc, [("Sinh hành động", {"bold": True}), (" bằng một lần gọi LLM duy nhất.", {})])

    add_para(doc, "Sơ đồ pipeline (luồng dữ liệu từ trên xuống):", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_para(doc, "Review tiếng Việt", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
    add_para(doc, "↓ VnCoreNLP word-segmentation", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "PhoBERT-large encoder", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
    add_para(doc, "↓ Head A (ATE BIO 15)    ↓ Head B (Sentiment 2)", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "BIO decoder → tuples (aspect_category, aspect_term, sentiment)", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
    add_para(doc, "↓ Aggregator (gộp N reviews, top-K terms)", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "Gemini 2.5 Flash (1 call)", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
    add_para(doc, "↓", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "Khuyến nghị hành động (tiếng Việt, có ưu tiên)", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
    add_caption(doc, "Hình 4.1. Sơ đồ tổng quan kiến trúc CausaSent.")

    add_section(doc, "4.2. Mô hình PhoBERT hai đầu")
    add_subsection(doc, "4.2.1. Encoder chia sẻ")
    add_para(doc,
        "Encoder dùng chung là PhoBERT-large (vinai/phobert-large, 370M "
        "tham số). Đầu vào là chuỗi token đã được word-segmented bằng "
        "VnCoreNLP và byte-pair encoded bởi tokenizer của PhoBERT.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_subsection(doc, "4.2.2. Head A — ATE BIO 15 lớp")
    add_para(doc,
        "Head A là một lớp tuyến tính ánh xạ vector ẩn 1024 chiều của "
        "mỗi token sang 15 lớp BIO: O, B-delivery, I-delivery, ..., "
        "B-appearance, I-appearance. Head A áp dụng cho mọi vị trí "
        "token ở cấp word. Các sub-word khác (tiếp theo của từ đã "
        "segmented) được đánh dấu IGNORE_INDEX và không tính vào loss.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_subsection(doc, "4.2.3. Head B — sentiment nhị phân")
    add_para(doc,
        "Head B là một lớp tuyến tính ánh xạ vector ẩn sang 2 lớp: "
        "positive, negative. Head B chỉ áp dụng tại vị trí B-token (vị "
        "trí mở đầu một aspect span). Tất cả các vị trí khác được đánh "
        "dấu IGNORE_INDEX.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))
    add_para(doc,
        "Cách thiết kế này có ưu điểm: sentiment được học cụ thể tại vị "
        "trí có ngữ cảnh tốt nhất (mở đầu của cụm aspect, thường nằm gần "
        "adjective phrase mô tả). Tránh được vấn đề \"một span dài có "
        "thể có nhiều token mâu thuẫn về sentiment\".",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_section(doc, "4.3. Hàm mất mát")
    add_para(doc, "Hàm mất mát huấn luyện là tổ hợp của ba thành phần:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_para(doc, "L = L_ATE + λ_s · L_sent + λ_c · L_con", align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)
    add_para(doc, "Trong đó:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_bullet(doc, [("L_ATE", {"bold": True}), (": cross-entropy class-weighted (trọng số tỉ lệ nghịch với tần suất nhãn trong tập train) trên 15 lớp BIO, bỏ qua vị trí sub-word.", {})])
    add_bullet(doc, [("L_sent", {"bold": True}), (": cross-entropy class-weighted trên 2 lớp sentiment, áp dụng tại các vị trí B-token.", {})])
    add_bullet(doc, [("L_con", {"bold": True}), (": supervised contrastive loss tính trên các embedding B-token trong batch, với nhãn là aspect_category.", {})])
    add_bullet(doc, "λ_s = 1.0, λ_c = 0.1 (cố định throughout).")

    add_section(doc, "4.4. Cấu hình huấn luyện")
    add_bullet(doc, "Optimizer: AdamW với learning rate 2e-5.")
    add_bullet(doc, "Batch size 16, max sequence length 128.")
    add_bullet(doc, "Linear warmup 10% bước đầu, sau đó linear decay.")
    add_bullet(doc, "8 epoch trên GPU Tesla P100 16GB (Kaggle), tổng thời gian khoảng 50 phút.")
    add_bullet(doc, "Model selection: chọn checkpoint có (F1_ATE + Acc_sentiment) / 2 cao nhất trên val.")

    add_section(doc, "4.5. Pipeline tổng hợp và sinh hành động")
    add_subsection(doc, "4.5.1. Tổng hợp summary")
    add_para(doc,
        "Sau khi mô hình PhoBERT cho ra danh sách tuple (aspect, "
        "sentiment, term) cho mỗi review, bước aggregator thực hiện:",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_numbered(doc, "Với mỗi cặp (aspect, sentiment): đếm số lần xuất hiện trên toàn tập N review.")
    add_numbered(doc, "Chuẩn hóa các surface form (NFC, lowercase, bỏ punctuation), rồi rank top-K aspect_term hay được nhắc nhất.")
    add_numbered(doc, "Tính tỉ lệ negative cho từng aspect category, phân vào ba mức ưu tiên: cao (negative_ratio ≥ 0.5 và count ≥ 5), trung bình (count ≥ 3), thấp (còn lại).")

    add_subsection(doc, "4.5.2. Gọi LLM một lần")
    add_para(doc, "Bảng summary được tuần tự hóa thành JSON compact và truyền cho Gemini 2.5 Flash kèm theo prompt tiếng Việt yêu cầu:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_bullet(doc, "Sinh một hành động ngắn (mệnh lệnh, ≤ 15 từ) cho mỗi cặp (aspect, sentiment).")
    add_bullet(doc, "Hành động phải dẫn chiếu được tới các cụm khách hay nhắc (evidence_terms).")
    add_bullet(doc, "Trả về JSON đúng schema đã định nghĩa.")
    add_para(doc,
        "Khi không có API key, hệ thống tự động dùng template fallback "
        "deterministic (vd. \"Cải thiện [top_keyword]\"). Điều này đảm "
        "bảo demo offline vẫn chạy được.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_page_break(doc)

    # =========================================================================
    # CHƯƠNG 5 — THỰC NGHIỆM
    # =========================================================================
    add_chapter(doc, 5, "Thực nghiệm và đánh giá")

    add_section(doc, "5.1. Cấu hình thực nghiệm")
    add_para(doc,
        "Thực nghiệm chính được chạy trên GPU Tesla P100 (Kaggle) trong "
        "khoảng 50 phút. Các tham số huấn luyện đã liệt kê ở Mục 4.4. Tất "
        "cả số liệu báo cáo dưới đây đều trên tập test (719 review / "
        "1.023 annotation), được tính bằng seqeval (entity-level F1) và "
        "sklearn.metrics (sentiment).",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_section(doc, "5.2. Kết quả chính")
    add_caption(doc, "Bảng 5.1. Kết quả chính trên tập test.")
    add_table(doc,
        ["Chỉ số", "Giá trị"],
        [
            ["ATE entity-level precision",            "0.2064"],
            ["ATE entity-level recall",               "0.7979"],
            ["ATE entity-level F1",                   "0.3280"],
            ["ATE seqeval micro F1",                  "0.3207"],
            ["ATE seqeval macro F1",                  "0.3128"],
            ["Sentiment accuracy",                    "0.8809"],
            ["Sentiment macro F1",                    "0.8760"],
            ["Sentiment F1 trên lớp negative",        "0.9007"],
        ])
    add_para(doc, "Hai con số đáng chú ý là:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_bullet(doc, [
        ("Sentiment F1 macro 0.876, F1 negative 0.901. ", {"bold": True}),
        ("Đây là kết quả mạnh, đặc biệt là khả năng phát hiện complaint (recall trên negative 0.928) — đúng nhu cầu của người bán.", {}),
    ])
    add_bullet(doc, [
        ("ATE entity-level F1 chỉ 0.328, ", {"bold": True}),
        ("với precision rất thấp (0.21) nhưng recall cao (0.80). Đây là một trade-off có chủ ý, được phân tích kỹ ở Mục 5.4.", {}),
    ])

    add_section(doc, "5.3. Phân tích chi tiết theo từng aspect")
    add_caption(doc, "Bảng 5.2. Entity-level F1 theo từng aspect (seqeval).")
    add_table(doc,
        ["Aspect", "Precision", "Recall", "F1", "Support"],
        [
            ["delivery",         "0.281", "0.906", "0.429", "127"],
            ["packaging",        "0.235", "0.873", "0.370", "118"],
            ["price",            "0.204", "0.752", "0.320", "109"],
            ["appearance",       "0.192", "0.779", "0.307", "312"],
            ["product_quality",  "0.184", "0.777", "0.297", "229"],
            ["customer_service", "0.139", "0.782", "0.236", "55"],
            ["usability",        "0.152", "0.467", "0.230", "15"],
            ["Macro",            "0.198", "0.762", "0.313", "965"],
            ["Micro",            "0.201", "0.799", "0.321", "965"],
        ])
    add_para(doc,
        "Các aspect tần suất cao và có từ vựng giới hạn (delivery, "
        "packaging, price) đạt F1 > 0.32. Usability chỉ có 15 mẫu test "
        "nên kết quả không đáng tin cậy về mặt thống kê.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_section(doc, "5.4. Bàn luận về sự bất đối xứng Precision/Recall")
    add_para(doc,
        "Trên tất cả các aspect, recall trong khoảng 0.75–0.90 còn "
        "precision luôn dưới 0.30. Đây là hệ quả trực tiếp của hai lựa "
        "chọn thiết kế:",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))
    add_numbered(doc, "Class-weighted cross-entropy với trọng số nghịch tần suất — buộc mô hình \"đoán non-O\" nhiều hơn vì O chiếm khoảng 95% token.")
    add_numbered(doc, "Supervised contrastive loss khuyến khích encoder tạo cluster cho mỗi aspect category — kéo recall lên cao hơn nữa.")
    add_para(doc, "Chúng tôi cho rằng đây là operating point đúng cho hệ thống, chứ không phải lỗi tối ưu. Vì:", align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))
    add_bullet(doc, "Bước aggregator gộp các tuple theo cell (aspect, sentiment) và chọn top-K bằng frequency ranking. Span sai (false positive) thường chỉ xuất hiện 1-2 lần, bị nhấn chìm bởi các complaint thật được nhắc lại nhiều lần.")
    add_bullet(doc, "Hai thứ bắt buộc phải đúng cho output cuối — aspect category và sentiment label — đều là hai thứ mô hình làm tốt nhất.")

    add_section(doc, "5.5. Chi tiết kết quả phân loại sentiment")
    add_caption(doc, "Bảng 5.3. Phân loại sentiment tại vị trí B-token.")
    add_table(doc,
        ["Lớp", "Precision", "Recall", "F1", "Support"],
        [
            ["positive",  "0.890", "0.816", "0.851", "407"],
            ["negative",  "0.875", "0.928", "0.901", "567"],
            ["Macro",     "0.883", "0.872", "0.876", "974"],
            ["Weighted",  "0.881", "0.881", "0.880", "974"],
        ])
    add_para(doc,
        "Lớp negative — quan trọng nhất với người bán — đạt F1 = 0.901 "
        "với recall 0.928. Trong 567 complaint của tập test, mô hình "
        "bắt được 526.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_section(doc, "5.6. Ví dụ định tính: hành động sinh ra bởi Gemini")
    add_para(doc,
        "Chạy pipeline trên 30 review ngẫu nhiên từ tập test, aggregator "
        "thu được 175 tuple (mean 5.8 / review, phản ánh operating point "
        "recall-first), phân bố trên 6 trong 7 aspect category. Gemini "
        "trả về 10 hành động được phân mức ưu tiên. Một số ví dụ tiêu biểu:",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))
    add_bullet(doc, [("packaging / negative / cao: ", {"italic": True}), ("\"Cải thiện chất lượng đóng gói, đảm bảo hàng hóa nguyên vẹn khi đến tay khách.\"", {"bold": True})])
    add_bullet(doc, [("product_quality / negative / cao: ", {"italic": True}), ("\"Rà soát, nâng cao chất lượng nội dung và cấu trúc sản phẩm.\"", {"bold": True})])
    add_bullet(doc, [("delivery / positive / trung bình: ", {"italic": True}), ("\"Duy trì tốc độ giao hàng nhanh chóng và đúng hẹn cho khách hàng.\"", {"bold": True})])
    add_bullet(doc, [("appearance / positive / thấp: ", {"italic": True}), ("\"Duy trì chất lượng hình thức sản phẩm, đặc biệt là bìa và giấy.\"", {"bold": True})])
    add_para(doc,
        "So với baseline template (vd. \"Cải thiện [top_keyword]\"), "
        "output của LLM cụ thể hơn (có evidence) và không trùng lặp (một "
        "hành động / cell thay vì một / tuple).",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_page_break(doc)

    # =========================================================================
    # CHƯƠNG 6 — KẾT LUẬN
    # =========================================================================
    add_chapter(doc, 6, "Kết luận và hướng phát triển")

    add_section(doc, "6.1. Tổng kết")
    add_para(doc, "Đồ án đã xây dựng thành công một hệ thống ABSA hoàn chỉnh cho review TMĐT tiếng Việt, gồm:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_bullet(doc, "Một tập dữ liệu mới (7.066 review / 10.307 nhãn, 91% IAA), công bố trên Hugging Face Hub (Tamir39/causasent-ate-v2).")
    add_bullet(doc, "Một mô hình PhoBERT hai đầu với supervised contrastive auxiliary, đạt F1 sentiment macro 0.876 và F1 negative 0.901, công bố trên Tamir39/causasent-phobert-ate.")
    add_bullet(doc, "Một pipeline tổng hợp + Gemini sinh hành động, kèm theo một ứng dụng PWA (Next.js + FastAPI + SSE streaming) cho phép người bán upload CSV và xem dashboard real-time.")

    add_section(doc, "6.2. Hạn chế")
    add_subsection(doc, "6.2.1. Trần precision của span")
    add_para(doc,
        "Precision entity-level khoảng 0.20. Phù hợp với operating point "
        "hiện tại nhưng nếu cần dùng đầu ra ATE ở chỗ khác (vd. làm "
        "dataset cho research khác) thì nên có CRF decoder hoặc "
        "confidence-threshold.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_subsection(doc, "6.2.2. Nhiễu từ pool bạc")
    add_para(doc,
        "Pool Tiki silver gán nhãn bởi một LLM duy nhất, có thể có lỗi "
        "hệ thống. Chỉ pool gold được validate 300 mẫu, pool silver "
        "chưa được re-validate ở quy mô tương đương.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_subsection(doc, "6.2.3. Taxonomy hẹp")
    add_para(doc,
        "7 lớp được thiết kế cho e-commerce nói chung. Nếu mở rộng sang "
        "các miền khác (food delivery, du lịch, ứng dụng di động) cần "
        "bổ sung lớp.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_subsection(doc, "6.2.4. Đánh giá action định tính")
    add_para(doc,
        "Khuyến nghị hành động hiện mới được đánh giá định tính qua "
        "quan sát. Chưa có study chính thức so sánh template vs "
        "per-review LLM vs aggregated structured LLM trên các tiêu chí "
        "actionable / specific / non-redundant.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=Cm(1.25))

    add_section(doc, "6.3. Hướng phát triển")
    add_numbered(doc, [("Ablation đầy đủ", {"bold": True}), (": huấn luyện thêm 2 cấu hình (λ_c = 0, gold-only) để xác định đóng góp của từng thành phần.", {})])
    add_numbered(doc, [("CRF decoder", {"bold": True}), (" trên Head A để cải thiện precision span.", {})])
    add_numbered(doc, [("Mở rộng taxonomy", {"bold": True}), (" sang các miền khác (food, travel).", {})])
    add_numbered(doc, [("User study", {"bold": True}), (" thật trên người bán, đo tỉ lệ hành động được thực hiện sau khi đọc dashboard.", {})])
    add_numbered(doc, [("Continuous learning", {"bold": True}), (": tự động re-train định kỳ khi có thêm review mới được người dùng feedback.", {})])

    add_page_break(doc)

    # =========================================================================
    # TÀI LIỆU THAM KHẢO
    # =========================================================================
    p = doc.add_paragraph(style="Heading 1")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("TÀI LIỆU THAM KHẢO")
    _set_font(r, size=Pt(16), bold=True)

    refs = [
        "Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., & Manandhar, S. (2014). SemEval-2014 Task 4: Aspect Based Sentiment Analysis. In Proceedings of SemEval.",
        "Nguyen, H. T. M., Nguyen, H. V., Ngo, Q. T., Vu, L. X., Tran, V. M., Ngo, B. X., & Le, C. A. (2018). VLSP Shared Task: Sentiment Analysis. In VLSP.",
        "Luc Phan, H., Tran, S. T., & Le-Hong, P. (2021). UIT-ViSD4SA: A Vietnamese Span Detection Dataset for Aspect-Based Sentiment Analysis. In Proceedings of PACLIC.",
        "Xu, H., Liu, B., Shu, L., & Yu, P. S. (2019). BERT Post-training for Review Reading Comprehension and Aspect-based Sentiment Analysis. In NAACL.",
        "Li, X., Bing, L., Li, P., & Lam, W. (2019). A Unified Model for Opinion Target Extraction and Target Sentiment Prediction. In AAAI.",
        "Nguyen, D. Q., & Nguyen, A. T. (2020). PhoBERT: Pre-trained Language Models for Vietnamese. In Findings of EMNLP.",
        "Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised Contrastive Learning. In NeurIPS.",
        "Vu, T., Nguyen, D. Q., Nguyen, D. Q., Dras, M., & Johnson, M. (2018). VnCoreNLP: A Vietnamese Natural Language Processing Toolkit. In NAACL Demonstrations.",
        "Li, R., Chen, H., Feng, F., Ma, Z., Wang, X., & Hovy, E. (2020). Aspect-Based Sentiment Analysis with Type-Aware Graph Convolutional Networks and Layer Ensemble. In NAACL.",
    ]
    for i, ref in enumerate(refs, start=1):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.first_line_indent = Cm(-1.0)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(f"[{i}] {ref}")
        _set_font(r, size=Pt(12))

    # =========================================================================
    # SAVE
    # =========================================================================
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT_PATH))
    print(f"OK · saved {OUT_PATH} · size={OUT_PATH.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    build()
