# -*- coding: utf-8 -*-
"""Generate fig1_architecture.drawio (same layout as the PPT version).

draw.io desktop can export it headlessly:
    draw.io -x -f pdf --crop -o ../fig1_architecture_drawio.pdf fig1_architecture.drawio
"""

from xml.sax.saxutils import escape

MM = 3.7795  # px per mm at 96 dpi


def px(v):
    return round(v * MM, 1)


INK = '#262626'
GRAY = '#6B6B6B'
BLUE = '#004472'
BLUE_FILL = '#DEEBF7'
GREEN_FILL = '#E2F0D9'
GREEN_LINE = '#37702F'
ORANGE_FILL = '#FBE5D6'
ORANGE_LINE = '#C55A11'
PURPLE_FILL = '#E8E0F0'
PURPLE_LINE = '#5B3E8E'
WHITE = '#FFFFFF'

cells = []
_n = [0]


def nid():
    _n[0] += 1
    return f'c{_n[0]}'


def box(x, y, w, h, html, fill=BLUE_FILL, line=BLUE, fs=10, sw=1.2, rounded=True):
    style = (f'rounded={1 if rounded else 0};arcSize=10;whiteSpace=wrap;html=1;'
             f'fillColor={fill};strokeColor={line};strokeWidth={sw};'
             f'fontFamily=Times New Roman;fontSize={fs};fontColor={INK};'
             f'verticalAlign=middle;align=center;spacing=1;')
    value = escape(html, {chr(34): chr(38)+chr(113)+chr(117)+chr(111)+chr(116)+chr(59)}) if html else ''
    cells.append(f'<mxCell id="{nid()}" value="{value}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{px(x)}" y="{px(y)}" width="{px(w)}" height="{px(h)}" as="geometry"/></mxCell>')


def label(x, y, w, h, html, fs=9, color=INK, align='left', bold=False, italic=False):
    st = 'align=' + align
    if bold:
        st += ';fontStyle=1'
    if italic:
        st += ';fontStyle=2'
    style = (f'text;html=1;{st};fontFamily=Times New Roman;fontSize={fs};'
             f'fontColor={color};verticalAlign=top;')
    cells.append(f'<mxCell id="{nid()}" value="{escape(html, {chr(34): chr(38)+chr(113)+chr(117)+chr(111)+chr(116)+chr(59)})}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{px(x)}" y="{px(y)}" width="{px(w)}" height="{px(h)}" as="geometry"/></mxCell>')


def arrow(x1, y1, x2, y2, color='#404040', sw=1.5):
    style = (f'endArrow=block;endFill=1;html=1;strokeColor={color};strokeWidth={sw};'
             f'fontFamily=Times New Roman;')
    cells.append(f'<mxCell id="{nid()}" style="{style}" edge="1" parent="1">'
                 f'<mxGeometry relative="1" as="geometry">'
                 f'<mxPoint x="{px(x1)}" y="{px(y1)}" as="sourcePoint"/>'
                 f'<mxPoint x="{px(x2)}" y="{px(y2)}" as="targetPoint"/>'
                 f'</mxGeometry></mxCell>')


SUP = lambda s: f'<sup>{s}</sup>'
SUB = lambda s: f'<sub>{s}</sub>'
I = lambda s: f'<i>{s}</i>'

# ---------------- panel (a) ----------------
label(4, 1, 8, 6, 'a', fs=13, bold=True)
cards = [
    ('<b><font color="#0072B2">XJTU</font></b><br><font style="font-size:8.5px" color="#6B6B6B">NCM, 6 protocols</font>', WHITE, '#0072B2'),
    ('<b><font color="#D55E00">TJU</font></b><br><font style="font-size:8.5px" color="#6B6B6B">NCA / NCM / NCM+NCA</font>', WHITE, '#D55E00'),
    ('<b><font color="#009E73">HUST / MIT</font></b><br><font style="font-size:8.5px" color="#6B6B6B">LFP, multi-stage &amp; fast charge</font>', WHITE, '#009E73'),
]
for k, (html, fill, line) in enumerate(cards):
    box(6, 8 + k * 19, 40, 16, html, fill=fill, line=line, fs=11, sw=1.6)
label(6, 66, 42, 12, 'batteries with different chemistries, protocols and aging levels',
      fs=8.5, color=GRAY, italic=True)
arrow(48, 33, 56, 33)

# ---------------- panel (b) ----------------
label(57, 1, 8, 6, 'b', fs=13, bold=True)
box(58, 22, 21, 20, f'cycle row {I("x")}{SUB("i")} ∈ ℝ{SUP("17")}', fill=WHITE, line=INK, fs=9.5)
label(58, 45, 22, 14, '16 charge-window<br>features + t', fs=8, color=GRAY, align='center')
box(84, 10, 30, 56, None, fill=GREEN_FILL, line=GREEN_LINE, sw=1.3)
label(85, 11.5, 28, 8, f'<b>label-free invariant map {I("φ")}</b>', fs=9.5, color='#2A501E', align='center', bold=True)
minis = ['row-standardized block (16)', f'anchored block {I("x")} − {I("x")}{SUB("1")}  (16)',
         'progress t (1)', 'row mean (1)']
for k, m in enumerate(minis):
    box(86.5, 20 + k * 11.5, 25, 9.5, m, fill=WHITE, line=GREEN_LINE, fs=8.5, sw=0.8)
arrow(79, 32, 84, 32)
box(119, 24, 21, 16, f'{I("φ(")}{I("x")}{SUB("i")}{I(")")} ∈ ℝ{SUP("34")}', fill=GREEN_FILL, line=GREEN_LINE, fs=9.5)
arrow(114, 32, 119, 32)
label(58, 68.5, 46, 12, 'row-wise transform: identical for every battery;<br>no source statistics fitted',
      fs=8, color=GRAY, align='center')
arrow(140, 32, 148, 32)

# ---------------- panel (c) ----------------
label(149, 1, 8, 6, 'c', fs=13, bold=True)
box(150, 10, 26, 11, f'encoder {I("g")}{SUB(I("θ"))}', fs=10)
box(158, 26, 20, 11, f'{I("z")}{SUB("i")} ∈ ℝ{SUP("8")}', fs=10)
arrow(168, 21, 168, 26)
arrow(140, 28.5, 150, 16.5)
box(129, 42, 26, 11, f'readout {I("h")}{SUB(I("ω"))}', fill=PURPLE_FILL, line=PURPLE_LINE, fs=10)
box(129, 56, 26, 11, f'{I("ŷ")}{SUB("i")}', fill=PURPLE_FILL, line=PURPLE_LINE, fs=10)
box(158, 42, 20, 11, f'operator {I("F̂")}{SUB(I("ψ"))}', fill=ORANGE_FILL, line=ORANGE_LINE, fs=9)
box(158, 56, 20, 11, f'Δ{I("z")}{SUB("i")}', fill=ORANGE_FILL, line=ORANGE_LINE, fs=10)
arrow(160, 37, 145, 42)
arrow(142, 53, 142, 56)
arrow(172, 37, 170, 42)
arrow(168, 53, 168, 56)
label(178.5, 40, 10, 6, f'{I("c")}{SUB("i")}', fs=8.5)
label(149, 70, 40, 8, f'next cycle: {I("z")}{SUB("i+1")} likewise from {I("φ(")}{I("x")}{SUB("i+1")}{I(")")}',
      fs=8, color=GRAY, italic=True)

# ---------------- losses strip ----------------
chips = [
    (4, f'<b>L{SUB("data")}</b>{""} :  MSE({I("ŷ")}{SUB("i")}, {I("y")}{SUB("i")})', PURPLE_FILL, PURPLE_LINE),
    (50, f'<b>L{SUB("op")}</b> :  MSE(Δ{I("z")}{SUB("i")}, {I("z")}{SUB("i+1")} − {I("z")}{SUB("i")})', ORANGE_FILL, ORANGE_LINE),
    (96, f'<b>L{SUB("mono")}</b> :  ReLU(({I("ŷ")}{SUB("i+1")}−{I("ŷ")}{SUB("i")})({I("y")}{SUB("i")}−{I("y")}{SUB("i+1")}))', PURPLE_FILL, PURPLE_LINE),
    (142, f'<b>L</b> = <b>L{SUB("data")} + α L{SUB("op")} + γ L{SUB("mono")}', WHITE, INK),
]
for x, html, fill, line in chips:
    box(x, 80, 44, 9, html, fill=fill, line=line, fs=9.5, sw=1.2)

xml = f'''<mxfile host="app.diagrams.net" modified="{__import__('datetime').datetime.now().isoformat()}" agent="script" version="21.0.0">
  <diagram id="fig1" name="Fig1">
    <mxGraphModel dx="800" dy="500" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{px(190)}" pageHeight="{px(92)}" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {chr(10).join(cells)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

with open('fig1_architecture.drawio', 'w', encoding='utf-8') as f:
    f.write(xml)
print('saved fig1_architecture.drawio')
