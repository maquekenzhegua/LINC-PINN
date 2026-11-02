# -*- coding: utf-8 -*-
"""Build Fig 1 (method overview) as a fully editable PowerPoint drawing.

Slide is sized to the final print width (190 mm) so the exported PDF drops
into LaTeX without rescaling.  Run, then export to PDF with
export_fig1.ps1 (PowerPoint COM).
"""

from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn

MM = 36000  # EMU per mm
SW, SH = 190, 92

INK = RGBColor(0x26, 0x26, 0x26)
GRAY = RGBColor(0x6B, 0x6B, 0x6B)
BLUE = RGBColor(0x00, 0x44, 0x72)
BLUE_FILL = RGBColor(0xDE, 0xEB, 0xF7)
GREEN_FILL = RGBColor(0xE2, 0xF0, 0xD9)
GREEN_LINE = RGBColor(0x37, 0x70, 0x2F)
ORANGE_FILL = RGBColor(0xFB, 0xE5, 0xD6)
ORANGE_LINE = RGBColor(0xC5, 0x5A, 0x11)
PURPLE_FILL = RGBColor(0xE8, 0xE0, 0xF0)
PURPLE_LINE = RGBColor(0x5B, 0x3E, 0x8E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DATASET_COLS = [RGBColor(0x00, 0x72, 0xB2), RGBColor(0xD5, 0x5E, 0x00), RGBColor(0x00, 0x9E, 0x73)]

prs = Presentation()
prs.slide_width = Emu(SW * MM)
prs.slide_height = Emu(SH * MM)
slide = prs.slides.add_slide(prs.slide_layouts[6])


def sub_run(r, baseline):
    r.font._rPr.set('baseline', str(baseline))


def rich(shape, parts, size=8, align=PP_ALIGN.CENTER, color=INK, bold=False):
    if parts is None:
        return
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(int(0.6 * MM))
    tf.margin_top = tf.margin_bottom = Emu(int(0.3 * MM))
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    if isinstance(parts, str):
        parts = [(parts, '')]
    for txt, st in parts:
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(size)
        r.font.name = 'Times New Roman'
        r.font.color.rgb = color
        r.font.bold = bold or ('b' in st)
        if 'i' in st:
            r.font.italic = True
        if 'sub' in st:
            sub_run(r, -25000)
        if 'sup' in st:
            sub_run(r, 30000)


def box(x, y, w, h, parts, fill=BLUE_FILL, line=BLUE, size=8, lw=1.0,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE, align=PP_ALIGN.CENTER, color=INK, bold=False, dash=None):
    sp = slide.shapes.add_shape(shape, Emu(int(x * MM)), Emu(int(y * MM)), Emu(int(w * MM)), Emu(int(h * MM)))
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(lw)
    sp.shadow.inherit = False
    if dash:
        ln = sp.line._get_or_add_ln()
        d = ln.makeelement(qn('a:prstDash'), {'val': dash})
        ln.append(d)
    try:
        sp.adjustments[0] = 0.12
    except Exception:
        pass
    rich(sp, parts, size=size, align=align, color=color, bold=bold)
    return sp


def plain_text(x, y, w, h, parts, size=8, align=PP_ALIGN.LEFT, color=INK, bold=False):
    tb = slide.shapes.add_textbox(Emu(int(x * MM)), Emu(int(y * MM)), Emu(int(w * MM)), Emu(int(h * MM)))
    rich(tb, parts, size=size, align=align, color=color, bold=bold)
    return tb


def arrow(x1, y1, x2, y2, color=RGBColor(0x40, 0x40, 0x40), w=1.3, dash=None, elbow=False):
    kind = MSO_CONNECTOR.ELBOW if elbow else MSO_CONNECTOR.STRAIGHT
    cn = slide.shapes.add_connector(kind, Emu(int(x1 * MM)), Emu(int(y1 * MM)), Emu(int(x2 * MM)), Emu(int(y2 * MM)))
    cn.line.color.rgb = color
    cn.line.width = Pt(w)
    cn.shadow.inherit = False
    ln = cn.line._get_or_add_ln()
    if dash:
        ln.append(ln.makeelement(qn('a:prstDash'), {'val': dash}))
    ln.append(ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'}))
    return cn


# ================= panel (a): multi-domain scenarios =================
plain_text(4, 1.5, 8, 5, 'a', size=10, bold=True)
cards = [
    ('XJTU', 'NCM, 6 protocols', DATASET_COLS[0]),
    ('TJU', 'NCA / NCM / NCM+NCA', DATASET_COLS[1]),
    ('HUST / MIT', 'LFP, multi-stage & fast charge', DATASET_COLS[2]),
]
for k, (name, sub, col) in enumerate(cards):
    y = 8 + k * 19
    box(6, y, 40, 16, None, fill=WHITE, line=col, lw=1.4)
    tb = slide.shapes.add_textbox(Emu(int(7 * MM)), Emu(int((y + 1.2) * MM)), Emu(int(38 * MM)), Emu(int(13.5 * MM)))
    tf = tb.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.LEFT
    r = p1.add_run(); r.text = name
    r.font.size = Pt(8.5); r.font.bold = True; r.font.name = 'Times New Roman'; r.font.color.rgb = col
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.LEFT
    r = p2.add_run(); r.text = sub
    r.font.size = Pt(6.8); r.font.name = 'Times New Roman'; r.font.color.rgb = GRAY
plain_text(6, 66, 40, 10, [('batteries with different chemistries, protocols and aging levels', 'i')],
           size=6.6, color=GRAY)
arrow(48, 33, 56, 33)

# ================= panel (b): invariant feature map =================
plain_text(57, 1.5, 8, 5, 'b', size=10, bold=True)
box(58, 22, 21, 20, [('cycle row x', ''), ('i', 'sub'), (' ∈ ℝ', 'i'), ('17', 'sup')],
    fill=WHITE, line=INK, size=7.5)
box(58, 44, 21, 16, [('16 charge-window', ''), ('\nfeatures + t', '')],
    fill=None, line=None, size=6.4, color=GRAY)
box(84, 10, 30, 56, None, fill=GREEN_FILL, line=GREEN_LINE, lw=1.2)
plain_text(85, 11.5, 28, 6, [('label-free invariant map φ', 'ib')], size=7.2, color=RGBColor(0x2A, 0x50, 0x1E))
minis = [
    ('row-standardized block (16)', 19),
    ('anchored block x − x', 19),
    ('progress t (1)', 19),
    ('row mean (1)', 19),
]
for k, (label, _) in enumerate(minis):
    y = 20 + k * 11.5
    if k == 1:
        parts = [('anchored block x − x', ''), ('1', 'sub'), ('  (16)', '')]
    else:
        parts = [(label, '')]
    box(86.5, y, 25, 9.5, parts, fill=WHITE, line=GREEN_LINE, lw=0.75, size=6.6)
arrow(79, 32, 84, 32)
box(119, 24, 21, 16, [('φ(x', 'ib'), ('i', 'isub'), (') ∈ ℝ', 'ib'), ('34', 'sup')],
    fill=GREEN_FILL, line=GREEN_LINE, lw=1.1, size=7.5)
arrow(114, 32, 119, 32)
plain_text(58, 68.5, 44, 12, [('row-wise transform: identical for every', ''), ('\nbattery; no source statistics fitted', '')],
           size=6.4, color=GRAY)
arrow(140, 32, 148, 32)

# ================= panel (c): shared networks and losses =================
plain_text(149, 1.5, 8, 5, 'c', size=10, bold=True)
box(150, 10, 26, 11, [('encoder g', 'ib'), ('θ', 'isub')], size=7.5)
box(158, 26, 20, 11, [('z', 'ib'), ('i', 'isub'), (' ∈ ℝ', 'ib'), ('8', 'sup')], size=7.5)
arrow(168, 21, 168, 26)                     # encoder -> latent
arrow(140, 28.5, 150, 16.5)                 # phi(x_i) -> encoder
box(129, 42, 26, 11, [('readout h', 'ib'), ('ω', 'isub')], fill=PURPLE_FILL, line=PURPLE_LINE, size=7.5)
box(129, 56, 26, 11, [('ŷ', 'ib'), ('i', 'isub')], fill=PURPLE_FILL, line=PURPLE_LINE, size=7.5)
box(158, 42, 20, 11, [('operator F̂', 'ib'), ('ψ', 'isub')], fill=ORANGE_FILL, line=ORANGE_LINE, size=7.2)
box(158, 56, 20, 11, [('Δz', 'ib'), ('i', 'isub')], fill=ORANGE_FILL, line=ORANGE_LINE, size=7.5)
arrow(160, 37, 145, 42)                     # z_i -> readout
arrow(142, 53, 142, 56)                     # readout -> ŷ
arrow(172, 37, 170, 42)                     # z_i -> operator
arrow(168, 53, 168, 56)                     # operator -> Δz
plain_text(178.5, 40, 11, 5, [('c', 'i'), ('i', 'isub')], size=6.8)
plain_text(149, 70, 40, 6, [('next cycle: z', 'i'), ('i+1', 'isub'), (' likewise from φ(x', 'i'), ('i+1', 'isub'), (')', 'i')],
           size=6.2, color=GRAY)

# ================= losses strip =================
chips = [
    (4, [('L', 'ib'), ('data', 'isub'), (' :  MSE(ŷ', 'ib'), ('i', 'isub'), (', y', 'ib'), ('i', 'isub'), (')', 'ib')], PURPLE_FILL, PURPLE_LINE),
    (50, [('L', 'ib'), ('op', 'isub'), (' :  MSE(Δz', 'ib'), ('i', 'isub'), (', z', 'ib'), ('i+1', 'isub'), (' − z', 'ib'), ('i', 'isub'), (')', 'ib')], ORANGE_FILL, ORANGE_LINE),
    (96, [('L', 'ib'), ('mono', 'isub'), (' :  ReLU((ŷ', 'ib'), ('i+1', 'isub'), ('−ŷ', 'ib'), ('i', 'isub'), (')(y', 'ib'), ('i', 'isub'), ('−y', 'ib'), ('i+1', 'isub'), ('))', 'ib')], PURPLE_FILL, PURPLE_LINE),
    (142, [('L = L', 'ib'), ('data', 'isub'), (' + α L', 'ib'), ('op', 'isub'), (' + γ L', 'ib'), ('mono', 'isub')], WHITE, INK),
]
for x, parts, fill, line in chips:
    box(x, 80, 44, 9, parts, fill=fill, line=line, lw=1.1, size=7.2)

prs.save('fig1_architecture.pptx')
print('saved fig1_architecture.pptx')
