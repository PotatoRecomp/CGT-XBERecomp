package com.datfooldive.cgt;

import android.app.Dialog;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.GradientDrawable;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

// The recomp's settings panel, drawn by Android rather than by ImGui: the activity's own dialog over the running
// picture, with the rows of the desktop F10 panel (src/common/menu.cpp) that mean something on a phone and the touch
// controls' own. The values live in native/settings_bridge.cpp, which this reads once when it opens and writes back one
// at a time; a slider only writes the file when it is let go. A controller's D-pad or a keyboard's arrows move through
// the rows and step the value of the one focused, as on the desktop.
class SettingsDialog extends Dialog {

    // the numbers native/settings_bridge.cpp reads and writes, in its order
    static final int RESOLUTION = 0, MSAA = 1, SMAA = 2, FXAA = 3, ANISO = 4, BLOOM = 5, FOG = 6, HDR = 7,
            HDR_AVAILABLE = 8, SCREEN_FILTER = 9, VSYNC = 10, FPS_CAP = 11, FRAME_RATE_60 = 12, FPS_COUNTER = 13,
            TOUCH_SENSITIVITY = 14, TOUCH_INVERT_Y = 15, OPACITY = 16, MOUSE_LOOK = 17, MOUSE_SENSITIVITY = 18,
            MOUSE_INVERT_Y = 19, VOLUME = 20, OUT_W = 21, OUT_H = 22, COUNT = 23;

    private static final int BG = Theme.BG, TEXT = Theme.TEXT, DIM = Theme.DIM, FRAME = Theme.FRAME,
            BUTTON = Theme.BUTTON, BUTTON_ON = Theme.BUTTON_ON, SELECTED = Theme.SELECTED;

    private final CGTActivity activity;  // the dialog's own context is a wrapper around it, not it
    private final int[] v = new int[COUNT];
    private boolean loading;  // a reload filling the widgets is not a change to write back

    private static final String[] TAB_NAMES = {"DISPLAY", "GRAPHICS", "CONTROLS", "AUDIO"};
    private final TextView[] tabs = new TextView[TAB_NAMES.length];
    private final LinearLayout[] pages = new LinearLayout[TAB_NAMES.length];
    private ScrollView scroll;
    // what reload() puts back into each widget from v
    private final List<Runnable> refreshers = new ArrayList<>();
    private View hdrRow;

    SettingsDialog(CGTActivity activity) {
        super(activity);
        this.activity = activity;
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setCanceledOnTouchOutside(true);

        // never taller than most of the screen: past that the pages scroll between the tabs and the footer
        LinearLayout frame = new LinearLayout(activity) {
            @Override
            protected void onMeasure(int widthSpec, int heightSpec) {
                int most = (int) (getResources().getDisplayMetrics().heightPixels * 0.92f);
                if (MeasureSpec.getMode(heightSpec) == MeasureSpec.UNSPECIFIED || MeasureSpec.getSize(heightSpec) > most)
                    heightSpec = MeasureSpec.makeMeasureSpec(most, MeasureSpec.AT_MOST);
                super.onMeasure(widthSpec, heightSpec);
            }
        };
        frame.setOrientation(LinearLayout.VERTICAL);
        frame.setBackground(Theme.box(BG, true));
        frame.setPadding(dp(16), dp(10), dp(16), dp(10));

        TextView title = text("RECOMP SETTINGS", 15, Theme.HEADING);
        title.setGravity(Gravity.CENTER);
        title.setPadding(0, 0, 0, dp(6));
        frame.addView(title, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        LinearLayout bar = new LinearLayout(activity);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setPadding(0, dp(2), 0, dp(4));
        for (int i = 0; i < TAB_NAMES.length; i++) {
            final int at = i;
            tabs[i] = chip(TAB_NAMES[i], 12);
            tabs[i].setOnClickListener(x -> select(at));
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
            lp.setMargins(i == 0 ? 0 : dp(2), 0, 0, 0);
            bar.addView(tabs[i], lp);
        }
        frame.addView(bar);
        frame.addView(line());

        for (int i = 0; i < pages.length; i++) {
            pages[i] = new LinearLayout(activity);
            pages[i].setOrientation(LinearLayout.VERTICAL);
            pages[i].setVisibility(View.GONE);
        }
        display(pages[0]);
        graphics(pages[1]);
        controls(pages[2]);
        audio(pages[3]);

        LinearLayout holder = new LinearLayout(activity);
        holder.setOrientation(LinearLayout.VERTICAL);
        for (LinearLayout p : pages) holder.addView(p);
        scroll = new ScrollView(activity);
        scroll.addView(holder);
        // the weight lets the page shrink to what the footer leaves it, rather than push the footer off
        frame.addView(scroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        frame.addView(line());
        frame.addView(footer());

        setContentView(frame);
        select(0);
        reload();
    }

    // the panel a fixed share of the screen wide, as tall as its rows ask for
    @Override
    protected void onStart() {
        super.onStart();
        Window window = getWindow();
        if (window == null) return;
        window.setBackgroundDrawable(new ColorDrawable(0));
        window.addFlags(WindowManager.LayoutParams.FLAG_DIM_BEHIND);
        WindowManager.LayoutParams lp = window.getAttributes();
        lp.dimAmount = 0.6f;
        lp.gravity = Gravity.CENTER;
        int screenW = getContext().getResources().getDisplayMetrics().widthPixels;
        lp.width = Math.min((int) (screenW * 0.66f), dp(700));
        lp.height = WindowManager.LayoutParams.WRAP_CONTENT;
        window.setAttributes(lp);
    }

    // F10 closes the panel as it opened it, as on the desktop; Escape and Back cancel the dialog by themselves
    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_F10 || keyCode == KeyEvent.KEYCODE_BUTTON_START) {
            dismiss();
            return true;
        }
        if (keyCode == KeyEvent.KEYCODE_BUTTON_L1 || keyCode == KeyEvent.KEYCODE_BUTTON_R1) {
            select((current() + (keyCode == KeyEvent.KEYCODE_BUTTON_R1 ? 1 : TAB_NAMES.length - 1)) % TAB_NAMES.length);
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }

    // ---------------------------------------------------------------- the pages

    private void display(LinearLayout box) {
        slider(box, "RENDER RESOLUTION", RESOLUTION, 25, 200, 5, value -> String.format(Locale.US, "%d%% (%dx%d)",
                value, v[OUT_W] * value / 100, v[OUT_H] * value / 100),
                "Of the screen's height: below 100 is faster and cooler, above sharper.");
        toggle(box, "VERTICAL SYNC", VSYNC, null);
        stepper(box, "MAX FRAME RATE", MAX_FPS_NAMES, MAX_FPS, FPS_CAP, "The console ran the game at 30.");
        stepper(box, "GAME FRAME RATE", new String[] {"30 (CONSOLE)", "60"}, new int[] {0, 1}, FRAME_RATE_60,
                "The game paces itself at 30 like the console; 60 runs it twice as smooth at the same speed.");
        toggle(box, "FPS COUNTER", FPS_COUNTER, null);
        // the row and its hint together, shown only where the screen has HDR
        LinearLayout hdr = new LinearLayout(getContext());
        hdr.setOrientation(LinearLayout.VERTICAL);
        box.addView(hdr);
        toggle(hdr, "HDR OUTPUT", HDR, "HDR10 on a screen with HDR turned on.");
        hdrRow = hdr;
    }

    private static final String[] MAX_FPS_NAMES = {"30", "60", "90", "120", "UNLIMITED"};
    private static final int[] MAX_FPS = {30, 60, 90, 120, 0};

    private void graphics(LinearLayout box) {
        stepper(box, "MSAA", new String[] {"OFF", "2X", "4X", "8X"}, new int[] {0, 2, 4, 8}, MSAA,
                "Multisampling: smooth geometry edges, sharp image.");
        toggle(box, "SMAA", SMAA, "Smooths every edge of the frame, alpha tested ones too.");
        toggle(box, "FXAA", FXAA, "A cheap edge blur at the present.");
        stepper(box, "ANISOTROPIC FILTERING", new String[] {"GAME DEFAULT", "2X", "4X", "8X", "16X"},
                new int[] {0, 2, 4, 8, 16}, ANISO, null);
        toggle(box, "BLOOM", BLOOM, null);
        toggle(box, "FOG", FOG, null);
        stepper(box, "SCREEN FILTER", new String[] {"OFF", "SHARPEN", "VIVID", "SCANLINES", "CRT", "HDR LOOK"},
                new int[] {0, 1, 2, 3, 4, 5}, SCREEN_FILTER,
                "SHARPEN crisper edges, VIVID richer colour, SCANLINES and CRT a TV's look, HDR LOOK deeper shadows "
                        + "and held back highlights.");
    }

    private void controls(LinearLayout box) {
        heading(box, "TOUCH");
        slider(box, "LOOK SENSITIVITY", TOUCH_SENSITIVITY, 10, 300, 5, SettingsDialog::scale,
                "A swipe on the right of the screen, or a thumb held on FIRE and dragged, turns the view directly as a "
                        + "mouse does. 1.00 turns 0.25 degrees a dp; scopes and zoom scale it down with the view.");
        toggle(box, "LOOK INVERT Y", TOUCH_INVERT_Y, null);
        slider(box, "BUTTON OPACITY", OPACITY, 0, 100, 5, value -> value + "%", null);
        TextView edit = chip("MOVE CONTROLS", 13);
        edit.setPadding(0, dp(7), 0, dp(7));
        edit.setOnClickListener(x -> {
            activity.beginLayoutEdit();
            dismiss();
        });
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(6);
        lp.bottomMargin = dp(4);
        box.addView(edit, lp);

        heading(box, "MOUSE");
        toggle(box, "MOUSE LOOK", MOUSE_LOOK, "A mouse plugged into the phone turns the view directly.");
        slider(box, "MOUSE SENSITIVITY", MOUSE_SENSITIVITY, 5, 500, 5, SettingsDialog::scale,
                "1.00 turns 0.022 degrees a count, as most PC shooters.");
        toggle(box, "MOUSE INVERT Y", MOUSE_INVERT_Y, null);
    }

    private void audio(LinearLayout box) {
        slider(box, "VOLUME", VOLUME, 50, 400, 25, value -> value + "%",
                "The game mixes quietly for a TV: 200% matches the phone's other sound.");
    }

    // a percentage of a sensitivity as the desktop panel writes it, 1.00 for 100
    private static String scale(int value) {
        return String.format(Locale.US, "%d.%02d", value / 100, value % 100);
    }

    // ---------------------------------------------------------------- the rows

    interface Format {
        String of(int value);
    }

    private LinearLayout row(LinearLayout box, String label) {
        LinearLayout row = new LinearLayout(getContext());
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(3), 0, dp(3));
        row.addView(text(label, 13, TEXT), new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        box.addView(row);
        return row;
    }

    // the desktop panel's tooltip, under its row
    private void hint(LinearLayout box, String hint) {
        if (hint == null) return;
        TextView t = text(hint, 11, DIM);
        t.setPadding(0, 0, dp(8), dp(4));
        box.addView(t);
    }

    private void heading(LinearLayout box, String title) {
        TextView t = text(title, 12, Theme.HEADING);
        t.setPadding(0, dp(8), 0, dp(2));
        box.addView(t);
    }

    // a value dragged along a bar; the arrows of a focused one step it
    private void slider(LinearLayout box, String label, int key, int min, int max, int step, Format format,
                        String hint) {
        LinearLayout row = row(box, label);
        Theme.Slider bar = new Theme.Slider(getContext(), (max - min) / step);
        Runnable show = () -> bar.setText(format.of(min + bar.progress * step));
        bar.listener = new Theme.Slider.Listener() {
            @Override
            public void changed(int progress) {
                show.run();
                if (loading) return;
                v[key] = min + progress * step;
                CGTActivity.nativeSettingsSet(key, v[key], false);
                activity.settingsChanged();
            }

            @Override
            public void released(int progress) {
                if (loading) return;
                v[key] = min + progress * step;
                CGTActivity.nativeSettingsSet(key, v[key], true);
                activity.settingsChanged();
            }
        };
        focusable(bar, delta -> {
            bar.setProgress(bar.progress + delta);
            bar.listener.changed(bar.progress);
            bar.listener.released(bar.progress);
        });
        row.addView(bar, new LinearLayout.LayoutParams(0, dp(28), 1.1f));
        hint(box, hint);
        refreshers.add(() -> {
            bar.setProgress((Math.max(min, Math.min(v[key], max)) - min) / step);
            show.run();
        });
    }

    // a value from a list, < value > as on the desktop panel
    private void stepper(LinearLayout box, String label, String[] names, int[] values, int key, String hint) {
        LinearLayout row = row(box, label);
        LinearLayout control = new LinearLayout(getContext());
        control.setOrientation(LinearLayout.HORIZONTAL);
        TextView down = chip("<", 13);
        TextView value = text(names[0], 13, TEXT);
        value.setGravity(Gravity.CENTER);
        value.setBackgroundColor(FRAME);
        TextView up = chip(">", 13);
        // the arrows take no focus: the value does, and steps with the arrow keys
        down.setFocusable(false);
        up.setFocusable(false);
        Step step = delta -> {
            if (loading) return;
            int i = Math.floorMod(indexOf(values, v[key]) + delta, values.length);
            v[key] = values[i];
            CGTActivity.nativeSettingsSet(key, v[key], true);
            activity.settingsChanged();
            value.setText(names[i]);
        };
        down.setOnClickListener(x -> step.by(-1));
        up.setOnClickListener(x -> step.by(1));
        focusable(value, step);
        LinearLayout.LayoutParams arrow = new LinearLayout.LayoutParams(dp(30), ViewGroup.LayoutParams.MATCH_PARENT);
        LinearLayout.LayoutParams middle = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f);
        middle.setMargins(dp(2), 0, dp(2), 0);
        control.addView(down, arrow);
        control.addView(value, middle);
        control.addView(up, new LinearLayout.LayoutParams(arrow));
        row.addView(control, new LinearLayout.LayoutParams(0, dp(28), 1.1f));
        hint(box, hint);
        refreshers.add(() -> value.setText(names[indexOf(values, v[key])]));
    }

    // ON or OFF, a tap or either arrow flips it; the row is returned so it can be dimmed
    private View toggle(LinearLayout box, String label, int key, String hint) {
        LinearLayout row = row(box, label);
        TextView sw = text("", 13, TEXT);
        sw.setGravity(Gravity.CENTER);
        sw.setBackgroundColor(FRAME);
        Step flip = delta -> {
            if (loading || !sw.isEnabled()) return;
            v[key] = v[key] != 0 ? 0 : 1;
            paint(sw, v[key] != 0);
            CGTActivity.nativeSettingsSet(key, v[key], true);
            activity.settingsChanged();
        };
        sw.setClickable(true);
        sw.setOnClickListener(x -> flip.by(1));
        focusable(sw, flip);
        row.addView(sw, new LinearLayout.LayoutParams(0, dp(28), 1.1f));
        hint(box, hint);
        refreshers.add(() -> paint(sw, v[key] != 0));
        return row;
    }

    private void paint(TextView sw, boolean on) {
        sw.setText(on ? "ON" : "OFF");
        sw.setTextColor(on ? SELECTED : TEXT);
    }

    interface Step {
        void by(int delta);
    }

    // the row a controller or keyboard is on: framed in the game's gold, the arrows step it, A or Enter steps it on
    private void focusable(View view, Step step) {
        view.setFocusable(true);
        view.setOnFocusChangeListener((x, focused) -> view.setForeground(focused ? border() : null));
        view.setOnKeyListener((x, keyCode, e) -> {
            if (e.getAction() != KeyEvent.ACTION_DOWN) return false;
            switch (keyCode) {
                case KeyEvent.KEYCODE_DPAD_LEFT:
                    step.by(-1);
                    return true;
                case KeyEvent.KEYCODE_DPAD_RIGHT:
                case KeyEvent.KEYCODE_ENTER:
                case KeyEvent.KEYCODE_DPAD_CENTER:
                case KeyEvent.KEYCODE_BUTTON_A:
                    step.by(1);
                    return true;
                default:
                    return false;
            }
        });
    }

    private GradientDrawable border() {
        GradientDrawable d = new GradientDrawable();
        d.setColor(0);
        d.setStroke(dp(2), SELECTED);
        return d;
    }

    private static int indexOf(int[] values, int value) {
        for (int i = 0; i < values.length; i++)
            if (values[i] == value) return i;
        return 0;
    }

    // ---------------------------------------------------------------- the panel

    private int current() {
        for (int i = 0; i < pages.length; i++)
            if (pages[i].getVisibility() == View.VISIBLE) return i;
        return 0;
    }

    private void select(int at) {
        for (int i = 0; i < pages.length; i++) {
            pages[i].setVisibility(i == at ? View.VISIBLE : View.GONE);
            tabs[i].setBackground(Theme.box(i == at ? BUTTON_ON : BUTTON, false));
            tabs[i].setTextColor(i == at ? SELECTED : TEXT);
        }
        scroll.scrollTo(0, 0);
    }

    private LinearLayout footer() {
        LinearLayout row = new LinearLayout(getContext());
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(6), 0, 0);
        row.addView(text("Saved as you change them. F10 / BACK close", 11, DIM),
                new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        TextView defaults = chip("DEFAULTS", 13);
        defaults.setOnClickListener(x -> {
            CGTActivity.nativeSettingsDefaults();
            activity.settingsChanged();
            reload();
        });
        TextView close = chip("CLOSE", 13);
        close.setOnClickListener(x -> dismiss());
        LinearLayout.LayoutParams one = new LinearLayout.LayoutParams(dp(96), dp(30));
        one.setMargins(dp(4), 0, 0, 0);
        row.addView(defaults, one);
        row.addView(close, new LinearLayout.LayoutParams(one));
        return row;
    }

    // the values as the file has them now, into the widgets
    private void reload() {
        int[] now = CGTActivity.nativeSettings();
        if (now == null) return;
        System.arraycopy(now, 0, v, 0, Math.min(now.length, COUNT));
        loading = true;
        for (Runnable r : refreshers) r.run();
        // HDR only where the screen has it turned on, as the desktop panel shows the row
        hdrRow.setVisibility(v[HDR_AVAILABLE] != 0 || v[HDR] != 0 ? View.VISIBLE : View.GONE);
        loading = false;
    }

    // ---------------------------------------------------------------- the pieces

    private TextView text(String s, float sp, int color) {
        return Theme.text(getContext(), s, sp, color);
    }

    private TextView chip(String s, float sp) {
        return Theme.chip(getContext(), s, sp);
    }

    private View line() {
        return Theme.line(getContext());
    }

    private int dp(float value) {
        return Theme.dp(getContext(), value);
    }
}
