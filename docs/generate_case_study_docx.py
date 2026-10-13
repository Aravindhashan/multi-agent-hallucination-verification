"""Generate a professional Word document (.docx) from the Case Study Markdown.

Uses python-docx to produce a publication-quality document with proper headings,
formatted paragraphs, bold/italic styling, tables, and Harvard-style references.
"""

import os
import re
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn


def setup_styles(doc):
    """Configure custom styles for the document."""
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    font.color.rgb = RGBColor(0, 0, 0)
    para = style.paragraph_format
    para.space_after = Pt(6)
    para.line_spacing = 1.5

    # Title style
    if 'DocTitle' not in [s.name for s in doc.styles]:
        title_style = doc.styles.add_style('DocTitle', WD_STYLE_TYPE.PARAGRAPH)
        title_style.font.name = 'Times New Roman'
        title_style.font.size = Pt(16)
        title_style.font.bold = True
        title_style.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_style.paragraph_format.space_after = Pt(4)

    # Subtitle style
    if 'DocSubtitle' not in [s.name for s in doc.styles]:
        sub_style = doc.styles.add_style('DocSubtitle', WD_STYLE_TYPE.PARAGRAPH)
        sub_style.font.name = 'Times New Roman'
        sub_style.font.size = Pt(11)
        sub_style.font.italic = True
        sub_style.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
        sub_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sub_style.paragraph_format.space_after = Pt(12)


def add_formatted_text(paragraph, text):
    """Parse simple markdown inline formatting (*italic*, **bold**) and add runs."""
    # Process bold and italic markers
    parts = re.split(r'(\*\*\*.*?\*\*\*|\*\*.*?\*\*|\*.*?\*|`[^`]+`)', text)
    for part in parts:
        if part.startswith('***') and part.endswith('***'):
            run = paragraph.add_run(part[3:-3])
            run.bold = True
            run.italic = True
        elif part.startswith('**') and part.endswith('**'):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith('*') and part.endswith('*') and len(part) > 2:
            run = paragraph.add_run(part[1:-1])
            run.italic = True
        elif part.startswith('`') and part.endswith('`'):
            run = paragraph.add_run(part[1:-1])
            run.font.name = 'Consolas'
            run.font.size = Pt(10)
        else:
            paragraph.add_run(part)


def parse_table(lines):
    """Parse markdown table lines into a list of rows (each row is a list of cells)."""
    rows = []
    for line in lines:
        line = line.strip()
        if line.startswith('|') and not re.match(r'^\|[\s\-:|]+\|$', line):
            cells = [c.strip() for c in line.split('|')[1:-1]]
            rows.append(cells)
    return rows


def add_table_to_doc(doc, rows):
    """Add a formatted table to the document."""
    if not rows:
        return
    num_cols = len(rows[0])
    table = doc.add_table(rows=len(rows), cols=num_cols)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, row_data in enumerate(rows):
        for j, cell_text in enumerate(row_data):
            if j < num_cols:
                cell = table.cell(i, j)
                cell.text = ''
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.space_before = Pt(2)

                # Clean markdown formatting for table cells
                clean_text = cell_text.replace('\\$', '$').replace('\\_', '_')
                add_formatted_text(p, clean_text)

                # Bold header row
                if i == 0:
                    for run in p.runs:
                        run.bold = True
                        run.font.size = Pt(9)
                else:
                    for run in p.runs:
                        run.font.size = Pt(9)

    # Shade header row
    for cell in table.rows[0].cells:
        shading = cell._tc.get_or_add_tcPr()
        shading_elem = shading.makeelement(qn('w:shd'), {
            qn('w:val'): 'clear',
            qn('w:color'): 'auto',
            qn('w:fill'): '1E3A8A',
        })
        shading.append(shading_elem)
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)


def build_docx(md_path, output_path):
    """Read the markdown case study and produce a formatted Word document."""
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    doc = Document()

    # Page setup
    section = doc.sections[0]
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

    setup_styles(doc)

    lines = md_content.split('\n')
    i = 0
    in_table = False
    table_lines = []
    in_code_block = False

    while i < len(lines):
        line = lines[i]

        # Skip mermaid / code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                in_code_block = False
                i += 1
                continue
            else:
                # Check if it's mermaid - skip entirely since diagram image is embedded
                if 'mermaid' in line.strip():
                    i += 1
                    while i < len(lines):
                        if lines[i].strip() == '```':
                            i += 1
                            break
                        i += 1
                    continue
                else:
                    in_code_block = True
                    i += 1
                    continue

        if in_code_block:
            i += 1
            continue

        # Handle markdown image syntax ![caption](path)
        if stripped.startswith('![') and '](' in stripped:
            img_match = re.search(r'\!\[(.*?)\]\((.*?)\)', stripped)
            if img_match:
                caption = img_match.group(1)
                img_file = img_match.group(2)
                img_dir = os.path.dirname(os.path.abspath(md_path))
                candidates = [
                    os.path.join(img_dir, img_file),
                    os.path.join(img_dir, '..', img_file),
                    os.path.join(os.getcwd(), img_file),
                ]
                resolved_img = None
                for c in candidates:
                    if os.path.exists(c):
                        resolved_img = c
                        break
                if resolved_img:
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p.paragraph_format.space_before = Pt(8)
                    p.paragraph_format.space_after = Pt(2)
                    run = p.add_run()
                    run.add_picture(resolved_img, width=Inches(6.2))
                    if caption:
                        cp = doc.add_paragraph()
                        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        cp.paragraph_format.space_after = Pt(8)
                        c_run = cp.add_run(f"Figure 1: {caption}")
                        c_run.font.size = Pt(9)
                        c_run.font.italic = True
                        c_run.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
                i += 1
                continue

        # Handle tables
        if line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
            i += 1
            continue
        elif in_table:
            in_table = False
            rows = parse_table(table_lines)
            if rows:
                add_table_to_doc(doc, rows)
                doc.add_paragraph()  # spacing after table
            table_lines = []

        stripped = line.strip()

        # Skip horizontal rules
        if stripped == '---' or stripped == '***':
            i += 1
            continue

        # Skip empty lines
        if not stripped:
            i += 1
            continue

        # Main title (# )
        if stripped.startswith('# ') and not stripped.startswith('## '):
            title_text = stripped[2:].strip()
            p = doc.add_paragraph(style='DocTitle')
            add_formatted_text(p, title_text)
            i += 1
            continue

        # Section headings (## )
        if stripped.startswith('## '):
            heading_text = stripped[3:].strip()
            h = doc.add_heading(level=1)
            run = h.add_run(heading_text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(14)
            run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
            i += 1
            continue

        # Subsection headings (### )
        if stripped.startswith('### '):
            heading_text = stripped[4:].strip()
            h = doc.add_heading(level=2)
            run = h.add_run(heading_text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)
            i += 1
            continue

        # Math display blocks ($$...$$)
        if stripped.startswith('$$'):
            math_text = stripped[2:]
            if math_text.endswith('$$'):
                math_text = math_text[:-2].strip()
            else:
                math_text = math_text.strip()
            if not math_text:
                # Multi-line math block
                i += 1
                math_lines = []
                while i < len(lines) and not lines[i].strip().startswith('$$'):
                    math_lines.append(lines[i].strip())
                    i += 1
                math_text = ' '.join(math_lines)
                i += 1  # skip closing $$
            else:
                i += 1  # single-line math, move to next line
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            # Clean LaTeX for Word display
            clean_math = math_text.replace('\\!', '').replace('\\;', ' ')
            clean_math = re.sub(r'\\Big\b', '', clean_math)
            clean_math = re.sub(r'\\text\{([^}]*)\}', r'\1', clean_math)
            clean_math = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1) / (\2)', clean_math)
            clean_math = re.sub(r'\\sum_\{([^}]*)\}\^\{([^}]*)\}', r'Σ(from \1 to \2)', clean_math)
            clean_math = re.sub(r'\\sum', 'Σ', clean_math)
            clean_math = re.sub(r'\\ln', 'ln', clean_math)
            clean_math = re.sub(r'\\left', '', clean_math)
            clean_math = re.sub(r'\\right', '', clean_math)
            clean_math = re.sub(r'\\mid', '|', clean_math)
            clean_math = re.sub(r'\\in', '∈', clean_math)
            clean_math = re.sub(r'\\to', '→', clean_math)
            clean_math = re.sub(r'\\cdot', '·', clean_math)
            clean_math = re.sub(r'\\min', 'min', clean_math)
            clean_math = re.sub(r'\\geq', '≥', clean_math)
            clean_math = re.sub(r'\\leq', '≤', clean_math)
            clean_math = re.sub(r'\\ll', '≪', clean_math)
            clean_math = re.sub(r'\\cap', '∩', clean_math)
            clean_math = re.sub(r'\\cup', '∪', clean_math)
            run = p.add_run(clean_math)
            run.font.name = 'Cambria Math'
            run.font.size = Pt(11)
            run.italic = True
            continue

        # Metadata line (bold domain/framework/execution)
        if stripped.startswith('**') and ':' in stripped:
            p = doc.add_paragraph()
            add_formatted_text(p, stripped)
            i += 1
            continue

        # Regular paragraph text
        p = doc.add_paragraph()
        add_formatted_text(p, stripped)
        i += 1

    # Handle any remaining table
    if in_table and table_lines:
        rows = parse_table(table_lines)
        if rows:
            add_table_to_doc(doc, rows)

    # Add page numbers in footer
    section = doc.sections[0]
    footer = section.footer
    footer.is_linked_to_previous = False
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.paragraph_format.space_before = Pt(6)

    doc.save(output_path)
    print(f"Successfully generated Word document: {output_path}")
    print(f"File size: {os.path.getsize(output_path):,} bytes")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(script_dir, 'CASE_STUDY_HALLUCINATION_VERIFIER.md')
    output_path = os.path.join(script_dir, 'Case_Study_Hallucination_Verifier.docx')

    if not os.path.exists(md_path):
        print(f"ERROR: Markdown source not found: {md_path}")
        sys.exit(1)

    build_docx(md_path, output_path)
