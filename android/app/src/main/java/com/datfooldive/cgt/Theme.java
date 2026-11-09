package com.datfooldive.cgt;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.drawable.GradientDrawable;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

// The desktop F10 panel's look (src/common/menu.cpp), shared by the settings panel (SettingsDialog) and the layout
// editor's bar (LayoutEditor): the game's menus' steel blue bars over a dark night map, square corners, the game's
// gold for the chosen value.
final class Theme {

    // menu.cpp's colours, as ImGui had them
    static final int BG = 0xF2080D17;         // kPanelBg
    static final int BORDER = 0xD9738CF2;     // kBorder
    static final int HEADING = 0xFF9EB8FF;    // kHeading
    static final int TEXT = 0xFFE6EDFA;       // kText
    static final int DIM = 0xFF738099;        // kTextDisabled
    static final int FRAME = 0xFF101A2E;      // a value's well
    static final int BUTTON = 0xFF1A2952;     // kButton
    static final int BUTTON_ON = 0xFF294D9E;  // kHighlight
    static final int GRAB = 0xFF2E478A;       // kButtonHovered
    static final int SELECTED = 0xFFD9C780;   // kSelected, the game's gold

    private Theme() {}

    static int dp(Context c, float value) {
        return (int) (value * c.getResources().getDisplayMetrics().density + 0.5f);
    }

    static TextView text(Context c, String s, float sp, int color) {
        TextView t = new TextView(c);
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp);
        t.setTextColor(color);
        return t;
    }

    // a panel button
    static TextView chip(Context c, String s, float sp) {
        TextView t = text(c, s, sp, TEXT);
        t.setGravity(Gravity.CENTER);
        t.setBackground(box(BUTTON, false));
        t.setClickable(true);
        t.setFocusable(true);
        t.setPadding(dp(c, 6), dp(c, 4), dp(c, 6), dp(c, 4));
        return t;
    }

    // square, filled, a one pixel border when asked
    static GradientDrawable box(int color, boolean border) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(color);
        if (border) d.setStroke(1, BORDER);
        return d;
    }

    // the panel's separator
    static View line(Context c) {
        View rule = new View(c);
        rule.setBackgroundColor(BORDER);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 1);
        lp.setMargins(0, dp(c, 3), 0, dp(c, 3));
        rule.setLayoutParams(lp);
        return rule;
    }

    // A value dragged along a bar: a flat well, a narrow grab at the value, the value written across the middle. A
    // finger anywhere on it takes the grab there, and the value is written once more when it lets go.
    static class Slider extends View {
        interface Listener {
            void changed(int progress);
            void released(int progress);
        }

        final int max;
        int progress;
        Listener listener;
        private String text = "";
        private boolean held;
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final float grabW;

        Slider(Context c, int max) {
            super(c);
            this.max = Math.max(max, 1);
            float density = c.getResources().getDisplayMetrics().density;
            grabW = 10 * density;
            paint.setTextSize(TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, 13,
                    c.getResources().getDisplayMetrics()));
            paint.setTextAlign(Paint.Align.CENTER);
        }

        void setProgress(int p) {
            progress = Math.max(0, Math.min(p, max));
            invalidate();
        }

        void setText(String s) {
            text = s;
            invalidate();
        }

        @Override
        protected void onDraw(Canvas canvas) {
            float w = getWidth(), h = getHeight(), pad = 2;
            paint.setColor(FRAME);
            canvas.drawRect(0, 0, w, h, paint);
            float x = pad + (w - 2 * pad - grabW) * progress / max;
            paint.setColor(held ? SELECTED : GRAB);
            canvas.drawRect(x, pad, x + grabW, h - pad, paint);
            paint.setColor(held ? SELECTED : TEXT);
            canvas.drawText(text, w / 2, h / 2 - (paint.descent() + paint.ascent()) / 2, paint);
        }

        @Override
        public boolean onTouchEvent(MotionEvent e) {
            switch (e.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                    held = true;
                    getParent().requestDisallowInterceptTouchEvent(true);
                    // fall through: the press moves the grab as a drag does
                case MotionEvent.ACTION_MOVE: {
                    float span = Math.max(getWidth() - grabW, 1);
                    int p = Math.round((e.getX() - grabW / 2) / span * max);
                    p = Math.max(0, Math.min(p, max));
                    if (p != progress) {
                        progress = p;
                        if (listener != null) listener.changed(p);
                    }
                    invalidate();
                    return true;
                }
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    held = false;
                    invalidate();
                    if (listener != null) listener.released(progress);
                    return true;
                default:
                    return true;
            }
        }
    }
}
