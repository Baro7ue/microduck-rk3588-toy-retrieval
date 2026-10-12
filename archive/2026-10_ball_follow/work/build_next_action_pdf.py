from pathlib import Path
import html
import json
import re
import subprocess

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, XPreformatted,
)

ROOT = Path('/Users/ethan/Documents/RK3588 Microduck')
SOURCE = ROOT / 'output/pdf/MicroDuck_下一阶段行动指南_20261004.md'
OUTPUT = SOURCE.with_suffix('.pdf')
CHECKS = ROOT / 'work/next_action_pdf_checks.json'

pdfmetrics.registerFont(TTFont('GuideCN', '/System/Library/Fonts/STHeiti Light.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('GuideBold', '/System/Library/Fonts/STHeiti Medium.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('GuideMono',
    '/Users/ethan/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/'
    'libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype/'
    'DejaVuSansMono.ttf'))
pdfmetrics.registerFontFamily('GuideCN', normal='GuideCN', bold='GuideBold',
                             italic='GuideCN', boldItalic='GuideBold')
ink = colors.HexColor('#172F3C')
accent = colors.HexColor('#22596D')
styles = {
    'title': ParagraphStyle('Title', fontName='GuideBold', fontSize=23, leading=31,
                           textColor=ink, spaceAfter=10),
    'body': ParagraphStyle('Body', fontName='GuideCN', fontSize=9.8, leading=15,
                          wordWrap='CJK', spaceAfter=7, textColor=ink),
    'h2': ParagraphStyle('H2', fontName='GuideBold', fontSize=14.1, leading=21,
                        wordWrap='CJK', spaceBefore=8, spaceAfter=8,
                        keepWithNext=True, textColor=accent),
    'h3': ParagraphStyle('H3', fontName='GuideBold', fontSize=11.2, leading=18,
                        wordWrap='CJK', spaceBefore=7, spaceAfter=6,
                        keepWithNext=True, textColor=accent),
    'code': ParagraphStyle('Code', fontName='GuideMono', fontSize=7.4, leading=11.1,
                          spaceBefore=3, spaceAfter=10, leftIndent=8, rightIndent=8,
                          borderPadding=8, backColor=colors.HexColor('#EDF4F7'),
                          textColor=ink),
    'cell': ParagraphStyle('Cell', fontName='GuideCN', fontSize=9.4, leading=14,
                          wordWrap='CJK', textColor=ink),
    'header': ParagraphStyle('Header', fontName='GuideBold', fontSize=9.4, leading=14,
                            wordWrap='CJK', textColor=colors.white),
}

def inline(text):
    escaped = html.escape(text)
    escaped = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', escaped)
    return escaped

lines = SOURCE.read_text().splitlines()
story, commands, widths = [], [], []
i = 0
while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1
        continue
    if line == '<!-- page -->':
        story.append(PageBreak())
        i += 1
        continue
    if line.startswith('```'):
        language, block = line[3:], []
        i += 1
        while i < len(lines) and not lines[i].startswith('```'):
            block.append(lines[i])
            i += 1
        command = '\n'.join(block)
        commands.append((language, command))
        widths.extend(pdfmetrics.stringWidth(part, 'GuideMono', 7.4) for part in block)
        story.append(XPreformatted(html.escape(command), styles['code']))
        i += 1
        continue
    if line.startswith('|'):
        rows = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            row = [cell.strip() for cell in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?', cell.replace(' ', '')) for cell in row):
                rows.append(row)
            i += 1
        data = [[Paragraph(inline(cell), styles['header' if n == 0 else 'cell'])
                 for cell in row] for n, row in enumerate(rows)]
        table = Table(data, colWidths=[108, A4[0] - 90 - 108], repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), accent),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F3F7F8')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#D8E5E9')),
        ]))
        story.extend([table, Spacer(1, 8)])
        continue
    style, text = ('title', line[2:]) if line.startswith('# ') else (
        ('h2', line[3:]) if line.startswith('## ') else (
        ('h3', line[4:]) if line.startswith('### ') else ('body', line)))
    story.append(Paragraph(inline(text), styles[style]))
    i += 1

def decorate(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(colors.HexColor('#5B7480'))
    canvas.setFont('GuideCN', 8)
    canvas.drawString(45, h - 27, 'RK3588 × MicroDuck  /  下一阶段行动指南  /  2026-10-04')
    canvas.setStrokeColor(colors.HexColor('#D8E5E9'))
    canvas.line(45, 37, w - 45, 37)
    canvas.drawString(45, 24, '当前任务：相机与实时性  ·  下一阶段：RK3588 CPU 部署')
    canvas.drawRightString(w - 45, 24, f'{doc.page}')
    canvas.restoreState()

SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=45, rightMargin=45,
                  topMargin=48, bottomMargin=50,
                  title='MicroDuck 下一阶段行动指南：相机验收到 RK3588 部署准备',
                  author='Codex', subject='逐步操作、作用解释、证据与阶段验收')\
    .build(story, onFirstPage=decorate, onLaterPages=decorate)
reader = PdfReader(OUTPUT)
texts = [page.extract_text() for page in reader.pages]
all_text = '\n'.join(texts)
for phrase in ('60.018', '50.002', '0.0333%', '第一步', '第六步',
               '--camera', '--duration 60', 'UYVY', '上板前', 'G3'):
    assert phrase in all_text, phrase
assert max(widths) < A4[0] - 90 - 22, max(widths)
for language, command in commands:
    if language == 'bash':
        check = subprocess.run(['/bin/zsh', '-n'], input=command, text=True, capture_output=True)
        assert check.returncode == 0, check.stderr
checks = {'pages': len(texts), 'page_text_lengths': [len(text) for text in texts],
          'command_blocks': len(commands), 'max_command_width_pt': max(widths),
          'shell_syntax': 'pass', 'output_bytes': OUTPUT.stat().st_size}
CHECKS.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(checks, ensure_ascii=False))
