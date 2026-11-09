#!/usr/bin/env python3
"""The on-screen controls' art: every control as a PNG, normal and held, into app/src/main/res/drawable-nodpi, drawn
by TouchControls.java. Each is drawn here as an SVG and rendered by rsvg-convert (run it again after changing a label).

The design shows the Xbox button prominently, with the game's action small below it where the port's keyboard
mapping (src/common/keyboard.cpp) names one, enclosed in a clean circle line.
"""
import math
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # android/
OUT = Path(os.environ.get("CONTROLS_OUT", ROOT / "app" / "src" / "main" / "res" / "drawable-nodpi"))
PX = 256          # the PNG's side
C, R = 130, 100   # the control's centre and radius in the box

GLYPH = "#e6e6e6"
DEFS = """<defs>
<radialGradient id='body' cx='50%' cy='50%' r='50%'>
  <stop offset='0' stop-color='#fff' stop-opacity='0.03'/>
  <stop offset='0.62' stop-color='#fff' stop-opacity='0.03'/>
  <stop offset='1' stop-color='#fff' stop-opacity='0.10'/>
</radialGradient>
<radialGradient id='bodyon' cx='50%' cy='50%' r='50%'>
  <stop offset='0' stop-color='#fff' stop-opacity='0.10'/>
  <stop offset='0.62' stop-color='#fff' stop-opacity='0.14'/>
  <stop offset='1' stop-color='#fff' stop-opacity='0.24'/>
</radialGradient>
<filter id='soft' x='-20%' y='-20%' width='140%' height='140%'><feGaussianBlur stdDeviation='5'/></filter>
</defs>"""


def svg(body):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 260 260' width='{PX}' height='{PX}'>"
            f"{DEFS}{body}</svg>")


def circle_line(on, r=R):
    """A clean circle outline: subtle fill and line normally, brighter when held."""
    fill = f"rgba(255,255,255,{0.25 if on else 0.06})"
    stroke_op = "0.95" if on else "0.60"
    stroke_w = "4.5" if on else "3.0"
    return (f"<circle cx='{C}' cy='{C}' r='{r - 4}' fill='{fill}' stroke='#ffffff' "
            f"stroke-opacity='{stroke_op}' stroke-width='{stroke_w}'/>")


def button_combo(xbox, action, on, xbox_size=68, r=R):
    """A button with prominent Xbox indicator and small action label below it."""
    text_op = "1.0" if on else "0.92"
    if action:
        text_svg = (
            f"<text x='{C}' y='{C - 12}' font-size='{xbox_size}' font-family='Noto Sans' font-weight='700' "
            f"text-anchor='middle' dominant-baseline='central' fill='#ffffff' fill-opacity='{text_op}' "
            f"stroke='#000000' stroke-opacity='0.40' stroke-width='3' paint-order='stroke'>{xbox}</text>"
            f"<text x='{C}' y='{C + 42}' font-size='22' font-family='Noto Sans' font-weight='800' letter-spacing='1.5' "
            f"text-anchor='middle' dominant-baseline='central' fill='#ffffff' fill-opacity='{0.95 if on else 0.75}' "
            f"stroke='#000000' stroke-opacity='0.40' stroke-width='2' paint-order='stroke'>{action}</text>"
        )
    else:
        text_svg = (
            f"<text x='{C}' y='{C}' font-size='{xbox_size}' font-family='Noto Sans' font-weight='700' "
            f"text-anchor='middle' dominant-baseline='central' fill='#ffffff' fill-opacity='{text_op}' "
            f"stroke='#000000' stroke-opacity='0.40' stroke-width='3' paint-order='stroke'>{xbox}</text>"
        )
    return svg(circle_line(on, r) + text_svg)


def arrow(dx, dy, on):
    """A d-pad arrow: a round key with a solid triangle pointing its way."""
    t = R * 0.36
    tip = (C + dx * t, C + dy * t)
    b1 = (C - dx * t * 0.55 - dy * t * 0.85, C - dy * t * 0.55 + dx * t * 0.85)
    b2 = (C - dx * t * 0.55 + dy * t * 0.85, C - dy * t * 0.55 - dx * t * 0.85)
    tri = (f"<polygon points='{tip[0]:.2f},{tip[1]:.2f} {b1[0]:.2f},{b1[1]:.2f} {b2[0]:.2f},{b2[1]:.2f}' "
           f"fill='{GLYPH}' stroke='{GLYPH}' stroke-width='8' stroke-linejoin='round'/>")
    return svg(circle_line(on) + tri)


def disc(on, r=R):
    shape = f"<circle cx='{C}' cy='{C}' r='{r - 1.5}' {{style}}/>"
    glow = f"<circle cx='{C}' cy='{C}' r='{r - 4}' {{style}}/>"
    out = shape.format(style=f"fill='url(#{'bodyon' if on else 'body'})'")
    out += (f"<clipPath id='in{r:.0f}'><circle cx='{C}' cy='{C}' r='{r}'/></clipPath>"
            f"<g clip-path='url(#in{r:.0f})'>" +
            glow.format(style=f"fill='none' stroke='#fff' stroke-opacity='{0.35 if on else 0.18}' "
                              f"stroke-width='10' filter='url(#soft)'") + "</g>")
    out += shape.format(style=f"fill='none' stroke='#fff' stroke-opacity='{0.65 if on else 0.4}' stroke-width='2.5'")
    return out


def stick_base():
    """The stick's well: a wide see-through disc, an inner ring where the thumb travels, and four small notches."""
    out = disc(False)
    out += (f"<circle cx='{C}' cy='{C}' r='{R * 0.62}' fill='none' stroke='#fff' stroke-opacity='0.14' "
            f"stroke-width='3'/>")
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        x, y = C + dx * R * 0.82, C + dy * R * 0.82
        out += f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4.5' fill='#fff' fill-opacity='0.35'/>"
    return svg(out)


def knob(on):
    """The thumb: the same soft raised disc, fuller in the middle."""
    r = R * 0.94
    return svg(f"<circle cx='{C}' cy='{C}' r='{r}' fill='#fff' fill-opacity='{0.18 if on else 0.10}'/>"
               f"<clipPath id='in'><circle cx='{C}' cy='{C}' r='{r}'/></clipPath>"
               f"<g clip-path='url(#in)'><circle cx='{C}' cy='{C}' r='{r - 3}' fill='none' stroke='#fff' "
               f"stroke-opacity='{0.35 if on else 0.2}' stroke-width='10' filter='url(#soft)'/></g>"
               f"<circle cx='{C}' cy='{C}' r='{r - 1.5}' fill='none' stroke='#fff' "
               f"stroke-opacity='{0.65 if on else 0.45}' stroke-width='2.5'/>")


def gear(on):
    """Settings: a cog with a hole inside a circle line."""
    pts = []
    step = 2 * math.pi / 8
    outer, inner = R * 0.56, R * 0.42
    for i in range(8):
        for k, rr in ((-0.28, inner), (-0.15, outer), (0.15, outer), (0.28, inner)):
            a = i * step + k * step
            pts.append(f"{C + math.cos(a) * rr:.2f},{C + math.sin(a) * rr:.2f}")
    h = R * 0.18
    return svg(circle_line(on) +
               f"<path fill-rule='evenodd' fill='{GLYPH}' d='M{' L'.join(pts)} Z "
               f"M{C + h},{C} A{h},{h} 0 1 0 {C - h},{C} A{h},{h} 0 1 0 {C + h},{C} Z' stroke='{GLYPH}' "
               f"stroke-width='3' stroke-linejoin='round'/>")


# Xbox button + small in-game action label (empty where the keyboard mapping gives the button no name):
BUTTONS = {
    "a": ("A", "", 68),
    "b": ("B", "CROUCH", 68),
    "x": ("X", "RELOAD", 68),
    "y": ("Y", "WEAPON", 68),
    "lt": ("LT", "", 58),
    "rt": ("RT", "FIRE", 58),
    "l3": ("L3", "", 50),
    "r3": ("R3", "AIM", 50),
    "white": ("WHITE", "", 36),
    "black": ("BLACK", "", 36),
    "start": ("START", "PAUSE", 36),
    "back": ("BACK", "", 44),
}


def art():
    out = {}
    for on in (False, True):
        s = "_on" if on else ""
        for key, (xbox, action, size) in BUTTONS.items():
            out[f"ctl_{key}{s}"] = button_combo(xbox, action, on, size)
        out[f"ctl_gear{s}"] = gear(on)
        for n, d in (("up", (0, -1)), ("down", (0, 1)), ("left", (-1, 0)), ("right", (1, 0))):
            out[f"ctl_arrow_{n}{s}"] = arrow(*d, on)
        out[f"ctl_knob{s}"] = knob(on)
    out["ctl_stick"] = stick_base()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, body in art().items():
            src = Path(tmp) / f"{name}.svg"
            src.write_text(body)
            subprocess.run(["rsvg-convert", str(src), "-w", str(PX), "-h", str(PX), "-o", str(OUT / f"{name}.png")],
                           check=True)
    print(f"{OUT}: {len(art())} images")


if __name__ == "__main__":
    main()
