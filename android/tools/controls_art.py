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




def svg(body):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 260 260' width='{PX}' height='{PX}'>"
            f"{body}</svg>")


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
    """A d-pad arrow: an open chevron line pointing its way in a circle line, brighter and bolder when held."""
    t = R * 0.42
    tip = (C + dx * t, C + dy * t)
    b1 = (C - dx * t * 0.25 - dy * t * 0.95, C - dy * t * 0.25 + dx * t * 0.95)
    b2 = (C - dx * t * 0.25 + dy * t * 0.95, C - dy * t * 0.25 - dx * t * 0.95)
    pts = f"{b1[0]:.2f},{b1[1]:.2f} {tip[0]:.2f},{tip[1]:.2f} {b2[0]:.2f},{b2[1]:.2f}"
    w = 17 if on else 14
    line = "<polyline points='{p}' fill='none' stroke='{c}' stroke-opacity='{o}' stroke-width='{w}' " \
           "stroke-linecap='round' stroke-linejoin='round'/>"
    return svg(circle_line(on) + line.format(p=pts, c="#000000", o="0.30", w=w + 5) +
               line.format(p=pts, c="#ffffff", o="0.95" if on else "0.60", w=w))


def stick_base():
    """The stick's well: one thin ring."""
    return svg(f"<circle cx='{C}' cy='{C}' r='{R - 3}' fill='#ffffff' fill-opacity='0.04' stroke='#ffffff' "
               f"stroke-opacity='0.45' stroke-width='3'/>")


def knob(on):
    """The thumb: a plain filled disc."""
    return svg(f"<circle cx='{C}' cy='{C}' r='{R * 0.9:.1f}' fill='#ffffff' fill-opacity='{0.45 if on else 0.25}' "
               f"stroke='#ffffff' stroke-opacity='{0.8 if on else 0.5}' stroke-width='3'/>")


def gear(on):
    """Settings: a cog drawn as a line, six teeth round a hole, inside a circle line."""
    pts = []
    teeth = 6
    step = 2 * math.pi / teeth
    outer, inner = R * 0.50, R * 0.36
    for i in range(teeth):
        for k, rr in ((-0.25, inner), (-0.14, outer), (0.14, outer), (0.25, inner)):
            a = i * step + k * step - math.pi / 2
            pts.append(f"{C + math.cos(a) * rr:.2f},{C + math.sin(a) * rr:.2f}")
    stroke = (f"fill='none' stroke='#ffffff' stroke-opacity='{0.95 if on else 0.75}' stroke-width='{9 if on else 7}' "
              f"stroke-linejoin='round'")
    return svg(circle_line(on) +
               f"<polygon points='{' '.join(pts)}' {stroke}/>"
               f"<circle cx='{C}' cy='{C}' r='{R * 0.14:.1f}' {stroke}/>")


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
    "start": ("START", "", 40),
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
