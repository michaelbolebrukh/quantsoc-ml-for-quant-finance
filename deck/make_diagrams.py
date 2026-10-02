"""Draw the explanatory D_ diagrams for the deck (concept pictures, not results).

Run `python3 deck/make_diagrams.py` (build_deck.py calls it on every build). Pure Pillow, because the deck
build must not depend on matplotlib. Drawn for the dark canvas (#0a0a0a, the site's dark --background) in
QuantSoc's dark tokens, set in Inter from deck/assets/fonts (semibold for emphasis), so the pictures sit
seamlessly on the slide. Each picture is drawn in inches at the size it is shown on the slide
(P pixels per inch), and every font is given in points, so type on the slide is never below 12pt.
Writes deck/figures/D_timeline.png, D_leak.png, D_split.png, D_tree.png, D_forest.png, D_survivorship.png.
Numbers in the tree, forest and survivorship pictures are illustrative and each picture says so.
The result figures F1..F4 are NOT drawn here: the notebook pipeline writes them.
"""
from pathlib import Path
import math
import random

from PIL import Image, ImageDraw, ImageFont

FIG_DIR = Path(__file__).resolve().parent / "figures"
P = 240  # pixels per inch

# QuantSoc dark tokens (next-pwa/app/styles/tokens.css, .dark). Contrast against BG in brackets.
BG = "#0a0a0a"                       # --background (neutral-950)
INK = "#dae0e7"                      # --foreground, 14.89:1
MUTED = "#acb6c3"                    # --muted-foreground, 9.65:1
GREY = "#717a88"                     # connector lines and arrows, 4.56:1 (non-text floor 3:1)
LIGHT = "#2a2e36"                    # an empty slot's fill (decorative)
EDGE = "#6b7280"                     # an empty slot's outline, 4.10:1
BLUE = "#3b82f6"                     # --brand-500 as a fill, 5.44:1
BLUE_D = "#61a6fa"                   # --brand-400 for blue text, 7.87:1
PRIMARY = "#6da2f8"                  # dark --primary, a fill that carries BG-coloured text at 7.71:1
TINT = "#142239"                     # brand-500 at 20% over BG; brand-400 text on it 6.33:1, INK 11.99:1
SURFACE = "#22262e"                  # a raised neutral panel; INK on it 11.41:1
HAIR = "#3a3f49"                     # 1px panel hairline (decorative)
WARN = "#f26464"                     # dark --destructive, 6.39:1
WHITE = "#ffffff"

_ASSET_FONTS = Path(__file__).resolve().parent / "assets" / "fonts"
_FONT_DIRS = [str(_ASSET_FONTS), "/usr/share/fonts/truetype/liberation", "/usr/share/fonts/truetype/dejavu",
              "/Library/Fonts", "C:/Windows/Fonts"]
_FONT_FILES = {False: ["Inter-Regular.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf", "Arial.ttf", "arial.ttf"],
               True: ["Inter-SemiBold.ttf", "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf",
                      "arialbd.ttf"]}


def font(pt: float, bold: bool = False):
    size = int(round(pt * P / 72))
    for d in _FONT_DIRS:
        for f in _FONT_FILES[bold]:
            p = Path(d) / f
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size=size)


def i(v):
    """inches -> pixels"""
    return int(round(v * P))


class Pic:
    def __init__(self, w_in, h_in):
        self.img = Image.new("RGB", (i(w_in), i(h_in)), BG)
        self.d = ImageDraw.Draw(self.img)

    def text(self, x, y, s, pt, bold=False, fill=INK, anchor="mm"):
        self.d.text((i(x), i(y)), s, font=font(pt, bold), fill=fill, anchor=anchor)

    def rbox(self, x0, y0, x1, y1, fill, r=0.1, outline=None, width=0.02):
        self.d.rounded_rectangle([i(x0), i(y0), i(x1), i(y1)], i(r), fill=fill, outline=outline,
                                 width=i(width) if outline else 0)

    def rect(self, x0, y0, x1, y1, fill, outline=None, width=0.015):
        self.d.rectangle([i(x0), i(y0), i(x1), i(y1)], fill=fill, outline=outline, width=i(width) if outline else 0)

    def line(self, pts, fill=INK, width=0.025):
        self.d.line([(i(a), i(b)) for a, b in pts], fill=fill, width=max(1, i(width)))

    def dot(self, x, y, r, fill=None, outline=None, width=0.015):
        self.d.ellipse([i(x - r), i(y - r), i(x + r), i(y + r)], fill=fill, outline=outline,
                       width=i(width) if outline else 0)

    def cross(self, x, y, r, fill=WARN, width=0.035):
        self.line([(x - r, y - r), (x + r, y + r)], fill, width)
        self.line([(x - r, y + r), (x + r, y - r)], fill, width)

    def arrow(self, x0, y0, x1, y1, fill=INK, width=0.025, head=0.1):
        self.line([(x0, y0), (x1, y1)], fill, width)
        ang = math.atan2(y1 - y0, x1 - x0)
        pts = [(x1, y1)]
        for da in (150, -150):
            a = ang + math.radians(da)
            pts.append((x1 + head * math.cos(a), y1 + head * math.sin(a)))
        self.d.polygon([(i(a), i(b)) for a, b in pts], fill=fill)

    def save(self, name):
        self.img.save(FIG_DIR / name)


def timeline(leaky: bool):
    p = Pic(10.0, 2.5)
    n, step, x0, t = 19, 0.45, 0.45, 13
    xs = [x0 + k * step for k in range(n)]
    ya = 1.55
    if leaky:
        p.text(5.0, 0.22, "A leaky feature reads a day after t: the model is cheating", 16, True, WARN)
    else:
        p.text(5.0, 0.22, "On day t we decide using only the past; the target is learned later", 16, True, INK)
    p.rbox(0.25, 0.5, xs[t] + 0.12, 1.22, TINT)
    fx1 = 9.7  # Inter sets wider than Liberation Sans: the future box runs to the right edge so its caption fits
    p.rbox(xs[t] + 0.22, 0.5, fx1, 1.22, SURFACE, outline=HAIR, width=0.012)
    p.text((0.25 + xs[t]) / 2, 0.72, "PAST: features may use these days", 16, True, BLUE_D)
    p.text((0.25 + xs[t]) / 2, 1.01, "returns, volatility, volume, up to and including day t", 13, False, INK)
    fx = (xs[t] + 0.22 + fx1) / 2
    p.text(fx, 0.72, "FUTURE: the target", 15, True, INK)
    p.text(fx, 1.01, "return over the next 5 days", 13, False, MUTED)
    p.line([(xs[0] - 0.15, ya), (xs[-1] + 0.35, ya)], INK, 0.022)
    p.arrow(xs[-1] + 0.2, ya, xs[-1] + 0.45, ya, INK, 0.022, 0.09)
    p.text(xs[-1] + 0.85, ya, "time", 14, False, MUTED)
    for x in xs:
        p.line([(x, ya - 0.06), (x, ya + 0.06)], INK, 0.018)
    p.dot(xs[t], ya, 0.075, BLUE_D)
    p.text(xs[t], ya + 0.24, "day t", 14, True, BLUE_D)
    p.text(xs[t + 5], ya + 0.24, "t+5", 14, True, INK)
    p.text(xs[0] + 0.2, ya + 0.24, "earlier", 14, False, MUTED)
    if not leaky:
        p.text(xs[t] - 1.8, ya + 0.62, "decision at the close of day t (a simplification)", 14, False, MUTED)
    else:
        by = ya + 0.48
        p.line([(xs[t - 5], by), (xs[t] - 0.03, by)], BLUE_D, 0.035)
        for xx in (xs[t - 5], xs[t] - 0.03):
            p.line([(xx, by - 0.08), (xx, by + 0.08)], BLUE_D, 0.035)
        p.line([(xs[t] + 0.03, by), (xs[t + 1], by)], WARN, 0.035)
        for xx in (xs[t] + 0.03, xs[t + 1]):
            p.line([(xx, by - 0.08), (xx, by + 0.08)], WARN, 0.035)
        p.cross((xs[t] + xs[t + 1]) / 2, by, 0.1)
        p.text(xs[t] - 0.15, by + 0.3, "Fine: ret_5 uses days t-5 to t", 14, True, BLUE_D, anchor="rm")
        p.text(xs[t] + 0.15, by + 0.3, "LEAKY_ret_next reads day t+1", 14, True, WARN, anchor="lm")
    p.save("D_leak.png" if leaky else "D_timeline.png")


def split():
    p = Pic(6.4, 2.45)
    n, cw, x0 = 30, 0.2, 0.2
    rnd = random.Random(7)
    rows = [("Random split: wrong for time series", [rnd.random() < 0.3 for _ in range(n)], WARN, 0.15),
            ("Time split: train, a gap, then test", [k >= 21 for k in range(n)], BLUE_D, 0.95)]
    for r, (title, is_test, colour, top) in enumerate(rows):
        p.text(x0, top, title, 14, True, colour, anchor="lm")
        for k in range(n):
            xx = x0 + k * cw
            if r == 1 and k == 20:
                p.rect(xx + 0.01, top + 0.2, xx + cw - 0.01, top + 0.55, BG, outline=GREY)
                continue
            p.rect(xx + 0.01, top + 0.2, xx + cw - 0.01, top + 0.55, MUTED if is_test[k] else BLUE)
    p.text(x0 + 20.5 * cw, 1.72, "gap", 13, True, MUTED)
    ly = 2.15
    p.rect(x0, ly - 0.09, x0 + 0.18, ly + 0.09, BLUE)
    p.text(x0 + 0.28, ly, "train", 14, False, INK, anchor="lm")
    p.rect(x0 + 1.15, ly - 0.09, x0 + 1.33, ly + 0.09, MUTED)
    p.text(x0 + 1.43, ly, "test", 14, False, INK, anchor="lm")
    p.text(x0 + 2.5, ly, "time", 14, False, MUTED, anchor="lm")
    p.arrow(x0 + 3.05, ly, x0 + 6.0, ly, MUTED, 0.022, 0.09)
    p.save("D_split.png")


def _box(p, cx, cy, w, h, lines, fill, colour, outline=None):
    p.rbox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, fill, r=0.09, outline=outline, width=0.012)
    k = len(lines)
    for j, (s, pt, b) in enumerate(lines):
        p.text(cx, cy + (j - (k - 1) / 2) * pt * 1.35 / 72, s, pt, b, colour)


def tree():
    p = Pic(6.4, 4.5)
    root, mids = (3.2, 0.5), [(1.6, 1.85), (4.8, 1.85)]
    leaves = [(0.8, 3.25), (2.4, 3.25), (4.0, 3.25), (5.6, 3.25)]
    for m in mids:
        p.line([root, m], GREY, 0.03)
    for k, lf in enumerate(leaves):
        p.line([mids[k // 2], lf], GREY, 0.03)
    _box(p, *root, 3.7, 0.78, [("Question 1", 14, True), ("Past 21-day return above 0?", 16, False)], SURFACE, INK, HAIR)
    _box(p, *mids[0], 2.95, 0.78, [("Question 2", 14, True), ("Volatility above normal?", 14, False)], TINT, INK)
    _box(p, *mids[1], 2.95, 0.78, [("Question 2", 14, True), ("Volume above normal?", 14, False)], TINT, INK)
    for (lx, ly), val in zip(leaves, ["+0.3%", "-0.1%", "+0.2%", "-0.2%"]):
        _box(p, lx, ly, 1.45, 0.8, [("Leaf", 13, False), (val, 18, True)], PRIMARY, BG)
    for (x, y, s) in [(2.4, 1.17, "yes"), (4.0, 1.17, "no"), (1.2, 2.55, "yes"), (2.0, 2.55, "no"),
                      (4.4, 2.55, "yes"), (5.2, 2.55, "no")]:
        p.rbox(x - 0.24, y - 0.13, x + 0.24, y + 0.13, BG, r=0.06)
        p.text(x, y, s, 13, True, MUTED)
    p.text(3.2, 4.2, "Illustrative numbers, not fitted to our data", 13, False, MUTED)
    p.save("D_tree.png")


def _mini_tree(p, cx, top):
    pts = [(cx, top), (cx - 0.35, top + 0.38), (cx + 0.35, top + 0.38),
           (cx - 0.55, top + 0.76), (cx - 0.17, top + 0.76), (cx + 0.17, top + 0.76), (cx + 0.55, top + 0.76)]
    for a, b in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
        p.line([pts[a], pts[b]], GREY, 0.025)
    for k, (x, y) in enumerate(pts):
        p.dot(x, y, 0.1 if k < 3 else 0.09, INK if k == 0 else ("#2f5a9e" if k < 3 else BLUE))


def forest():
    p = Pic(6.4, 4.5)
    xs, vals = [0.85, 2.45, 4.05], ["+0.3%", "-0.1%", "+0.4%"]
    for k, (x, v) in enumerate(zip(xs, vals)):
        _mini_tree(p, x, 0.2)
        p.text(x, 1.3, f"Tree {k + 1}", 14, True, INK)
        p.text(x, 1.58, f"forecast {v}", 14, False, INK)
    p.text(5.55, 0.62, "... and", 14, True, MUTED)
    p.text(5.55, 0.88, "many more", 14, False, MUTED)
    p.text(5.55, 1.14, "trees", 14, False, MUTED)
    bx, by = 3.2, 2.62
    for x, y0 in zip(xs + [5.55], [1.78, 1.78, 1.78, 1.4]):
        p.arrow(x, y0, bx + (x - bx) * 0.3, by - 0.38, GREY, 0.025, 0.09)
    _box(p, bx, by, 3.4, 0.66, [("Average of all the trees", 16, True)], SURFACE, INK, HAIR)
    p.arrow(bx, by + 0.34, bx, by + 0.7, GREY, 0.03, 0.1)
    _box(p, bx, 3.68, 3.8, 0.66, [("Forest forecast +0.2%", 18, True)], PRIMARY, BG)
    p.text(3.2, 4.3, "Illustrative: the average of the three trees shown", 13, False, MUTED)
    p.save("D_forest.png")


def survivorship():
    p = Pic(6.4, 4.5)
    cols, rows, sp, r = 8, 6, 0.33, 0.12
    rnd = random.Random(3)
    gone = set(rnd.sample(range(cols * rows), 12))
    for px0, title, show in [(0.2, "Back then: all firms", True), (3.55, "Our data: survivors", False)]:
        p.text(px0 + 1.3, 0.25, title, 15, True, INK)
        for k in range(cols * rows):
            cx, cy = px0 + 0.15 + (k % cols) * sp, 0.75 + (k // cols) * sp
            if k in gone:
                if show:
                    p.dot(cx, cy, r, LIGHT)
                    p.cross(cx, cy, 0.065, WARN, 0.025)
                else:
                    p.dot(cx, cy, r, None, outline=EDGE)
            else:
                p.dot(cx, cy, r, BLUE)
    p.arrow(2.95, 1.6, 3.4, 1.6, GREY, 0.035, 0.11)
    p.dot(0.35, 3.0, r, BLUE)
    p.text(0.6, 3.0, "still trading today: in our data", 15, False, INK, anchor="lm")
    p.dot(0.35, 3.45, r, LIGHT)
    p.cross(0.35, 3.45, 0.065, WARN, 0.025)
    p.text(0.6, 3.45, "failed, taken over or delisted: missing", 15, False, INK, anchor="lm")
    p.text(3.2, 4.15, "Illustration, not real counts", 13, False, MUTED)
    p.save("D_survivorship.png")


def make_all():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    timeline(False)
    timeline(True)
    split()
    tree()
    forest()
    survivorship()


if __name__ == "__main__":
    make_all()
    print("wrote", sorted(p.name for p in FIG_DIR.glob("D_*.png")))
