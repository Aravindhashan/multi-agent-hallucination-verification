"""Script to compile the full Case Study Markdown into a publication-quality PDF.
Uses ReportLab with custom styles, headers, footers, tables, and mathematical callouts.
"""

import os
import re
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total pages dynamically for 'Page X of Y' footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running Header on page 2+
        if self._pageNumber > 1:
            self.drawString(36, 762, "Dynamic Adversarial Multi-Agent Hallucination Verification — Academic Case Study")
            self.drawRightString(576, 762, "NLP Research Report | v1.1")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 756, 576, 756)

        # Footer on all pages
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 42, 576, 42)

        self.drawString(36, 30, "100% Free Open-Source Architecture | DeBERTa-v3 Cross-Encoder & Dual-Agent Debate")
        self.drawRightString(576, 30, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


def parse_markdown_to_flowables(md_path, styles):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    lines = md_text.split('\n')
    story = []

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=4,
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor('#0284C7'),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=11.2,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=5,
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#334155'),
        leftIndent=12,
        spaceAfter=2,
    )

    math_box_style = ParagraphStyle(
        'MathBox',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#0F172A'),
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=8.8,
        textColor=colors.HexColor('#1E293B'),
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell_style,
        fontName='Helvetica-Bold',
    )

    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#475569'),
        spaceAfter=4,
    )

    ref_style = ParagraphStyle(
        'RefText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#1E293B'),
        leftIndent=16,
        firstLineIndent=-16,
        spaceAfter=4,
    )

    i = 0
    in_table = False
    table_lines = []
    in_code_block = False
    in_references = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Handle mermaid blocks
        if stripped.startswith('```'):
            if in_code_block:
                in_code_block = False
                i += 1
                continue
            else:
                if 'mermaid' in stripped:
                    i += 1
                    while i < len(lines) and not lines[i].strip() == '```':
                        i += 1
                    i += 1
                    callout_p = Paragraph("<i>[Pipeline Architecture Diagram — 12-Stage Dual-Agent Orchestration Flow]</i>", meta_style)
                    callout_tbl = Table([[callout_p]], colWidths=[540])
                    callout_tbl.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
                        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#94A3B8')),
                        ('PADDING', (0, 0), (-1, -1), 4),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ]))
                    story.append(callout_tbl)
                    story.append(Spacer(1, 4))
                    continue
                else:
                    in_code_block = True
                    i += 1
                    continue

        if in_code_block:
            i += 1
            continue

        # Handle table accumulation
        if stripped.startswith('|'):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(stripped)
            i += 1
            continue
        elif in_table:
            in_table = False
            # Render table
            rows_data = []
            for t_line in table_lines:
                if re.match(r'^\|[\s\-:|]+\|$', t_line):
                    continue
                cells = [c.strip() for c in t_line.split('|')[1:-1]]
                rows_data.append(cells)
            
            if rows_data:
                num_cols = len(rows_data[0])
                # Determine sensible col widths totaling ~540
                if num_cols == 6:
                    col_widths = [85, 85, 90, 85, 95, 100]
                elif num_cols == 5:
                    col_widths = [140, 80, 85, 75, 160]
                elif num_cols == 3:
                    col_widths = [130, 150, 260]
                else:
                    w = 540.0 / num_cols
                    col_widths = [w] * num_cols

                formatted_rows = []
                for row_idx, row in enumerate(rows_data):
                    f_row = []
                    for col_idx, cell in enumerate(row):
                        cell_clean = cell.replace('\\$', '$').replace('\\_', '_')
                        cell_clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', cell_clean)
                        cell_clean = re.sub(r'\*(.*?)\*', r'<i>\1</i>', cell_clean)
                        if row_idx == 0:
                            p = Paragraph(f"<b>{cell_clean}</b>", table_header_style)
                        else:
                            p = Paragraph(cell_clean, table_cell_style)
                        f_row.append(p)
                    formatted_rows.append(f_row)

                tbl = Table(formatted_rows, colWidths=col_widths)
                tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                    ('LEFTPADDING', (0, 0), (-1, -1), 4),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 5))
            table_lines = []

        if not stripped:
            i += 1
            continue

        if stripped == '---':
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceBefore=4, spaceAfter=6))
            i += 1
            continue

        # Main Title
        if stripped.startswith('# ') and not stripped.startswith('## '):
            t_text = stripped[2:].strip()
            story.append(Paragraph(t_text, title_style))
            i += 1
            continue

        # Check section headings
        if stripped.startswith('## '):
            h_text = stripped[3:].strip()
            if '6. References' in h_text:
                in_references = True
            else:
                in_references = False
            story.append(Paragraph(h_text, h1_style))
            i += 1
            continue

        if stripped.startswith('### '):
            h2_text = stripped[4:].strip()
            story.append(Paragraph(h2_text, h2_style))
            i += 1
            continue

        # Math display
        if stripped.startswith('$$'):
            m_text = stripped[2:]
            if m_text.endswith('$$'):
                m_text = m_text[:-2].strip()
            else:
                m_text = m_text.strip()
            
            # Clean math for readable text display in PDF box
            c_math = m_text.replace('\\!', '').replace('\\;', ' ')
            c_math = re.sub(r'\\Big\b', '', c_math)
            c_math = re.sub(r'\\text\{([^}]*)\}', r'\1', c_math)
            c_math = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1) / (\2)', c_math)
            c_math = re.sub(r'\\sum_\{([^}]*)\}\^\{([^}]*)\}', r'∑(from \1 to \2)', c_math)
            c_math = re.sub(r'\\sum', '∑', c_math)
            c_math = re.sub(r'\\ln', 'ln', c_math)
            c_math = re.sub(r'\\left', '', c_math)
            c_math = re.sub(r'\\right', '', c_math)
            c_math = re.sub(r'\\mid', '|', c_math)
            c_math = re.sub(r'\\in', '∈', c_math)
            c_math = re.sub(r'\\to', '→', c_math)
            c_math = re.sub(r'\\cdot', '·', c_math)
            c_math = re.sub(r'\\min', 'min', c_math)
            c_math = re.sub(r'\\geq', '≥', c_math)
            c_math = re.sub(r'\\leq', '≤', c_math)
            c_math = re.sub(r'\\ll', '≪', c_math)
            c_math = re.sub(r'\\cap', '∩', c_math)
            c_math = re.sub(r'\\cup', '∪', c_math)

            m_table = Table([[Paragraph(c_math, math_box_style)]], colWidths=[540])
            m_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#94A3B8')),
                ('PADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(m_table)
            story.append(Spacer(1, 3))
            i += 1
            continue

        # Bullets
        if stripped.startswith('- ') or stripped.startswith('* ') or re.match(r'^\d+\.\s', stripped):
            clean_b = re.sub(r'^[-*]\s+', '', stripped)
            clean_b = re.sub(r'^\d+\.\s+', '', clean_b)
            clean_b = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_b)
            clean_b = re.sub(r'\*(.*?)\*', r'<i>\1</i>', clean_b)
            clean_b = re.sub(r'`([^`]+)`', r'<code>\1</code>', clean_b)
            story.append(Paragraph(f"• {clean_b}", bullet_style))
            i += 1
            continue

        # Metadata headers (e.g., Title:, Domain:, Core Technologies:)
        if stripped.startswith('**') and ':' in stripped:
            clean_meta = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', stripped)
            clean_meta = re.sub(r'\*(.*?)\*', r'<i>\1</i>', clean_meta)
            story.append(Paragraph(clean_meta, meta_style))
            i += 1
            continue

        # Reference entries
        if in_references and len(stripped) > 20:
            clean_ref = re.sub(r'\*(.*?)\*', r'<i>\1</i>', stripped)
            clean_ref = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_ref)
            story.append(Paragraph(clean_ref, ref_style))
            i += 1
            continue

        # Regular body text
        clean_p = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', stripped)
        clean_p = re.sub(r'\*(.*?)\*', r'<i>\1</i>', clean_p)
        clean_p = re.sub(r'`([^`]+)`', r'<code>\1</code>', clean_p)
        clean_p = clean_p.replace('\\$', '$').replace('\\_', '_')
        story.append(Paragraph(clean_p, body_style))
        i += 1

    return story


def build_pdf(md_path, pdf_path):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=48,
    )

    styles = getSampleStyleSheet()
    story = parse_markdown_to_flowables(md_path, styles)
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {pdf_path}")
    print(f"File size: {os.path.getsize(pdf_path):,} bytes")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_file = os.path.join(script_dir, 'CASE_STUDY_HALLUCINATION_VERIFIER.md')
    pdf_file = os.path.join(script_dir, 'Case_Study_Hallucination_Verifier.pdf')

    if not os.path.exists(md_file):
        print(f"Error: Markdown source not found at {md_file}")
        sys.exit(1)

    build_pdf(md_file, pdf_file)
