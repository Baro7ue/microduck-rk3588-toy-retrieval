from pathlib import Path
import re, html, json, ast, subprocess
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, XPreformatted, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

ROOT = Path('/Users/ethan/Documents/RK3588 Microduck')
SOURCE = ROOT / 'outputs/任务卡1_Mac串口启动日志_逐步教程.md'
OUTPUT = ROOT / 'outputs/任务卡1_Mac串口启动日志_逐步教程.pdf'
pdfmetrics.registerFont(TTFont('CN', '/System/Library/Fonts/STHeiti Light.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('CNBold', '/System/Library/Fonts/STHeiti Medium.ttc', subfontIndex=0))
mono = '/Users/ethan/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype/DejaVuSansMono.ttf'
pdfmetrics.registerFont(TTFont('Code', mono))
pdfmetrics.registerFontFamily('CN', normal='CN', bold='CNBold', italic='CN', boldItalic='CNBold')
ink = colors.HexColor('#172A36')
blue = colors.HexColor('#2B5268')
styles = {
    'title': ParagraphStyle('Title', fontName='CNBold', fontSize=21, leading=29, textColor=ink, spaceAfter=8),
    'body': ParagraphStyle('Body', fontName='CN', fontSize=10.1, leading=15.2, wordWrap='CJK', spaceAfter=7, textColor=ink),
    'h2': ParagraphStyle('H2', fontName='CNBold', fontSize=14, leading=20, wordWrap='CJK', spaceBefore=11, spaceAfter=9, keepWithNext=True, textColor=ink),
    'h3': ParagraphStyle('H3', fontName='CNBold', fontSize=11.2, leading=17, wordWrap='CJK', spaceBefore=8, spaceAfter=6, keepWithNext=True, textColor=blue),
    'code': ParagraphStyle('Code', fontName='Code', fontSize=7.6, leading=10.5, spaceBefore=3, spaceAfter=9, leftIndent=8, rightIndent=7, borderPadding=8, backColor=colors.HexColor('#EFF4F7'), textColor=colors.HexColor('#183447')),
    'cell': ParagraphStyle('Cell', fontName='CN', fontSize=9.5, leading=14.2, wordWrap='CJK', textColor=ink),
    'headcell': ParagraphStyle('HeadCell', fontName='CNBold', fontSize=9.5, leading=14.2, wordWrap='CJK', textColor=colors.white),
    'small': ParagraphStyle('Small', fontName='CN', fontSize=9.1, leading=13, wordWrap='CJK', spaceAfter=7, textColor=colors.HexColor('#50636D')),
}
inline_re = re.compile(r'\[([^\]]+)\]\(([^)]+)\)|\*\*([^*]+)\*\*|`([^`]+)`')
def inline(s):
    parts=[]; end=0
    for m in inline_re.finditer(s):
        parts.append(html.escape(s[end:m.start()]))
        if m.group(1) is not None:
            parts.append(f'<a href="{html.escape(m.group(2), quote=True)}" color="#255B7A"><u>{html.escape(m.group(1))}</u></a>')
        elif m.group(3) is not None:
            parts.append('<b>'+html.escape(m.group(3))+'</b>')
        else:
            text=m.group(4)
            font='CN' if any(ord(c)>127 for c in text) else 'Code'
            parts.append(f'<font name="{font}" size="9">{html.escape(text)}</font>')
        end=m.end()
    parts.append(html.escape(s[end:]))
    return ''.join(parts)

source=SOURCE.read_text()
lines=source.splitlines(); story=[]; i=0; code_blocks=[]; code_widths=[]
while i<len(lines):
    line=lines[i].strip()
    if not line:
        i+=1; continue
    if line=='<!-- page -->':
        story.append(PageBreak()); i+=1; continue
    if line.startswith('```'):
        lang=line[3:]; block=[]; i+=1
        while i<len(lines) and not lines[i].startswith('```'):
            block.append(lines[i]); i+=1
        code='\n'.join(block)
        code_blocks.append((lang,code))
        escaped=html.escape(code)
        escaped=re.sub(r'([^\x00-\x7f]+)', lambda m: '<font name="CN">'+m.group(1)+'</font>', escaped)
        code_widths.extend(pdfmetrics.stringWidth(t,'Code',7.6) for t in block if all(ord(c)<128 for c in t))
        story.append(XPreformatted(escaped,styles['code']))
        i+=1; continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            row=[c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?', c.replace(' ','')) for c in row): rows.append(row)
            i+=1
        data=[[Paragraph(inline(c),styles['headcell' if ri==0 else 'cell']) for c in row] for ri,row in enumerate(rows)]
        table=Table(data,colWidths=[150,355],repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),blue),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F6F8')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('LINEBELOW',(0,0),(-1,0),.5,blue),
            ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D8E2E8')),
        ]))
        story.extend([table,Spacer(1,9)]); continue
    if line.startswith('# '):
        story.append(Paragraph(inline(line[2:]),styles['title']))
    elif line.startswith('## '):
        story.append(Paragraph(inline(line[3:]),styles['h2']))
    elif line.startswith('### '):
        story.append(Paragraph(inline(line[4:]),styles['h3']))
    else:
        style=styles['small'] if line.startswith('任务卡 #1 ·') else styles['body']
        story.append(Paragraph(inline(line),style))
    i+=1

def decorate(c,doc):
    c.saveState()
    w,h=A4
    c.setFont('CN',8)
    c.setFillColor(colors.HexColor('#627783'))
    c.drawString(45,h-27,'RK3588  /  任务卡 #1  /  Mac 调试串口与启动记录')
    c.setStrokeColor(colors.HexColor('#D8E2E8')); c.setLineWidth(.5)
    c.line(45,37,w-45,37)
    c.drawString(45,24,'先记录，再上电  ·  不烧录、不升级、不改板上设置')
    c.drawRightString(w-45,24,str(doc.page))
    c.restoreState()

doc=SimpleDocTemplate(str(OUTPUT),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=45,bottomMargin=49,title='任务卡1：Mac串口登录与出厂启动日志逐步教程',author='',subject='按任务卡1顺序，每步解释目的、命令、成功判据和证据')
doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
reader=PdfReader(OUTPUT)
texts=[p.extract_text() for p in reader.pages]
assert len(texts)>=7
assert all(len(t)>100 for t in texts)
alltext='\n'.join(texts)
for phrase in ['第 1 步','第 9 步','Terminal ready','1500000','--logfile','uname -a','/etc/os-release']:
    assert phrase in alltext,phrase
assert max(code_widths)<488,max(code_widths)
for index,(lang,code) in enumerate(code_blocks):
    if lang=='bash':
        p=subprocess.run(['/bin/zsh','-n'],input=code,text=True,capture_output=True)
        assert p.returncode==0,(index,p.stderr)
report={'pages':len(reader.pages),'page_text_lengths':[len(t) for t in texts],'code_blocks':len(code_blocks),'max_code_width_pt':max(code_widths),'shell_syntax':'pass','output_bytes':OUTPUT.stat().st_size}
(ROOT/'work/task1_pdf_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
