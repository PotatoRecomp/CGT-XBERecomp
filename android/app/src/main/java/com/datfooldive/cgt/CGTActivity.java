package com.datfooldive.cgt;

import android.Manifest;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.storage.StorageManager;
import android.os.storage.StorageVolume;
import android.provider.Settings;
import android.view.InputDevice;
import android.view.InputEvent;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
import android.widget.Toast;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.libsdl.app.SDLActivity;
import org.libsdl.app.SDLControllerManager;

// SDL's activity running libmain.so. The game goes in a CGT folder on the phone's storage
// (/storage/emulated/0/CGT) or at the top of an SD card. Reading it needs all files access, a special access Android
// grants on a settings page of its own, never among the app's ordinary permissions. This asks for it, and says where
// the game goes, each time the app comes back without them; the game itself waits and starts as soon as it finds the
// files.
public class CGTActivity extends SDLActivity {
    static final String SHARED = "CGT";

    private static CGTActivity current;  // for the toasts the game's threads ask for
    private AlertDialog dialog;
    private SettingsDialog settings;
    private boolean notNow;  // the player turned the access down this run
    private final GameImporter importer = new GameImporter(this);
    private boolean importing;  // the file picker or the copy is up: nothing else is asked meanwhile
    private TouchControls controls;

    @Override
    protected String[] getLibraries() {
        return new String[] {"SDL3", "main"};
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        current = this;
        fillDisplay();
        // the controls over the game's surface, taking every touch on it
        if (mLayout != null) {
            controls = new TouchControls(this);
            mLayout.addView(controls, new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT));
        }
        File own = getExternalFilesDir(null);
        if (own != null) new File(own, "assets").mkdirs();
    }

    @Override
    protected void onDestroy() {
        if (current == this) current = null;
        if (editor != null) editor.hide();
        super.onDestroy();
        System.exit(0);
    }

    // native/main.cpp's ShowError: a toast, from the game's thread
    public static void showToast(String text) {
        CGTActivity a = current;
        if (a == null || text == null) return;
        a.runOnUiThread(() -> Toast.makeText(a, text, Toast.LENGTH_LONG).show());
    }

    // A controller some phones' game services pass on as a virtual device (MIUI's took a DualShock 4 on a Redmi
    // Note 8 Pro): SDL sees no controller in it and drops its input, so the game never learns it is there and the
    // on-screen controls never give way to it. Its buttons and sticks go to the game as port 0's here
    // (native/loose.cpp); a controller SDL has goes its own way as before.
    private static native void nativeLooseKey(int keyCode, boolean down);
    private static native void nativeLooseAxes(float lx, float ly, float rx, float ry, float lt, float rt, float hx,
                                               float hy);
    private static native void nativeLooseRelease();

    // native/settings_bridge.cpp: the panel's values, a change to one of them, back to the defaults, and the panel
    // opened or closed (the game's pad stands idle while it is up, as behind the desktop's F10 panel)
    static native int[] nativeSettings();
    static native void nativeSettingsSet(int key, int value, boolean commit);
    static native void nativeSettingsDefaults();
    private static native void nativeSettingsOpen(boolean open);
    // native/main.cpp: SAVE REPORT, the report written into cgt.log in the game's folder; the file's path back
    static native String nativeSaveReport();

    // native/settings_bridge.cpp: true while the game still counts itself in the background, for the resume-freeze
    // recovery in onWindowFocusChanged
    private static native boolean nativeStuckInBackground();

    private LayoutEditor editor;

    // the panel's MOVE CONTROLS: it closes, the controls stay drawn where they are and a finger moves them; the bar
    // this puts at the bottom of the screen is the editor's size bar, Done and Reset
    public void beginLayoutEdit() {
        if (controls == null) return;
        if (editor == null) editor = new LayoutEditor(this, controls);
        editor.show();
        controls.beginEdit(editor);
        hideSystemBars();
    }

    // the gear (TouchControls), or F10 on a keyboard: the settings panel over the game, which keeps running behind it
    void showSettings() {
        if (settings != null && settings.isShowing()) return;
        if (controls != null) controls.releaseAll();
        settings = new SettingsDialog(this);
        settings.setOnDismissListener(d -> {
            settings = null;
            nativeSettingsOpen(false);
            hideSystemBars();
        });
        nativeSettingsOpen(true);
        settings.show();
        hideSystemBars();
    }

    // native/settings_bridge.cpp's app::ToggleMenu, from the window thread: F10, or Escape while the panel is up
    public void showSettingsFromNative(boolean show) {
        runOnUiThread(() -> {
            if (show) showSettings();
            else if (settings != null) settings.dismiss();
        });
    }

    // a value in the panel changed: the controls read their opacity from it, the layout with DEFAULTS
    void settingsChanged() {
        if (controls == null) return;
        controls.reloadSettings();
        controls.reloadLayout();
    }

    // native/touch.cpp, from the game's thread: the settings file is loaded (the controls were made before it was, and
    // read the layout and the panel's values again now), or a controller came or went
    public void controlsSync(boolean pad) {
        runOnUiThread(() -> {
            if (controls == null) return;
            controls.reloadLayout();
            controls.reloadSettings();
            controls.setPad(pad);
        });
    }

    private static boolean loose(InputEvent e, int sources) {
        return (e.getSource() & sources) != 0 && !SDLControllerManager.isDeviceSDLJoystick(e.getDeviceId());
    }

    @Override
    public boolean dispatchKeyEvent(KeyEvent e) {
        boolean pad = KeyEvent.isGamepadButton(e.getKeyCode())
                || (e.getKeyCode() >= KeyEvent.KEYCODE_DPAD_UP && e.getKeyCode() <= KeyEvent.KEYCODE_DPAD_RIGHT);
        if (pad && loose(e, InputDevice.SOURCE_GAMEPAD | InputDevice.SOURCE_JOYSTICK)) {
            if (e.getRepeatCount() == 0) nativeLooseKey(e.getKeyCode(), e.getAction() == KeyEvent.ACTION_DOWN);
            return true;
        }
        return super.dispatchKeyEvent(e);
    }

    @Override
    public boolean dispatchGenericMotionEvent(MotionEvent e) {
        if (loose(e, InputDevice.SOURCE_JOYSTICK) && e.getActionMasked() == MotionEvent.ACTION_MOVE) {
            // the right stick on Z and RZ, the triggers on LTRIGGER and RTRIGGER or BRAKE and GAS, as Android has them
            nativeLooseAxes(e.getAxisValue(MotionEvent.AXIS_X), e.getAxisValue(MotionEvent.AXIS_Y),
                    e.getAxisValue(MotionEvent.AXIS_Z), e.getAxisValue(MotionEvent.AXIS_RZ),
                    Math.max(e.getAxisValue(MotionEvent.AXIS_LTRIGGER), e.getAxisValue(MotionEvent.AXIS_BRAKE)),
                    Math.max(e.getAxisValue(MotionEvent.AXIS_RTRIGGER), e.getAxisValue(MotionEvent.AXIS_GAS)),
                    e.getAxisValue(MotionEvent.AXIS_HAT_X), e.getAxisValue(MotionEvent.AXIS_HAT_Y));
            return true;
        }
        return super.dispatchGenericMotionEvent(e);
    }

    @Override
    protected void onResume() {
        super.onResume();
        hideSystemBars();
        askForGame();
    }

    @Override
    protected void onPause() {
        nativeLooseRelease();  // its ups go elsewhere while the game is not in front
        if (controls != null) controls.releaseAll();
        super.onPause();
    }

    // the game fills the display, cutout and all: without this a phone with a notch or a camera keeps a black band
    // on the side of it, and the status bar's space stays black above the picture
    @SuppressWarnings("deprecation")
    private void fillDisplay() {
        Window window = getWindow();
        WindowManager.LayoutParams lp = window.getAttributes();
        if (Build.VERSION.SDK_INT >= 30) {
            window.setDecorFitsSystemWindows(false);
            lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS;
        } else {
            lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
        }
        window.setAttributes(lp);
        // A swipe brings the bars back for a moment; they are hidden again as soon as the system has laid the window
        // out with them, so the game never keeps a strip of status bar or navigation bar over its picture.
        if (Build.VERSION.SDK_INT >= 30) {
            window.getDecorView().setOnApplyWindowInsetsListener((v, insets) -> {
                if (insets.isVisible(WindowInsets.Type.statusBars())
                        || insets.isVisible(WindowInsets.Type.navigationBars())) {
                    v.removeCallbacks(hideBars);
                    v.post(hideBars);
                }
                return insets;
            });
        }
    }

    private final Runnable hideBars = this::hideSystemBars;

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            hideSystemBars();
            recoverIfStuck();
        }
    }

    // A rapid surfaceDestroyed -> surfaceCreated -> surfaceDestroyed burst on resume can leave SDL's surface not ready
    // while Android keeps the Activity resumed and focused. handleNativeState() will not move the app to RESUMED
    // without mIsSurfaceReady, so the game thread stays parked in its native background wait (WaitWhileUnfocused) and
    // the picture is frozen until Android happens to hand a fresh surface back, which can take many seconds or not
    // come at all. Shortly after we regain the window focus, if the game still counts itself in the background, make
    // the SurfaceView build a new surface (GONE then VISIBLE across two traversals): that fires surfaceChanged, sets
    // mIsSurfaceReady, and lets SDL resume at once.
    private void recoverIfStuck() {
        final View s = mSurface;
        if (s == null) {
            return;
        }
        s.postDelayed(() -> {
            if (hasWindowFocus() && mSurface == s && nativeStuckInBackground()) {
                s.setVisibility(View.GONE);
                s.postDelayed(() -> s.setVisibility(View.VISIBLE), 32);
            }
        }, 800);
    }

    @SuppressWarnings("deprecation")
    private void hideSystemBars() {
        // SDL clears this when the game leaves fullscreen; left on, the system keeps room for the status bar
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_FORCE_NOT_FULLSCREEN);
        if (Build.VERSION.SDK_INT < 30) {
            getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                    | View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                    | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                    | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
            return;
        }
        getWindow().setDecorFitsSystemWindows(false);
        WindowInsetsController c = getWindow().getInsetsController();
        if (c == null) return;
        c.hide(WindowInsets.Type.systemBars());
        c.setSystemBarsBehavior(WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
    }

    private static final boolean ALL_FILES_ACCESS = Build.VERSION.SDK_INT >= 30;
    private static final int STORAGE_REQUEST = 0x4457;
    private boolean storageAsked;

    private boolean hasStorageAccess() {
        if (ALL_FILES_ACCESS) return Environment.isExternalStorageManager();
        return checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
                && checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
    }

    private void askForGame() {
        if (importing || (dialog != null && dialog.isShowing())) return;
        File own = getExternalFilesDir(null);
        if (own != null && hasGame(new File(own, "assets"))) return;
        if (!hasStorageAccess()) {
            if (!notNow) askAllFilesAccess();
            return;
        }
        File shared = new File(Environment.getExternalStorageDirectory(), SHARED);
        shared.mkdirs();
        if (hasGame(shared) || hasGame(new File(shared, "assets"))) return;
        for (File card : cardFolders())
            if (hasGame(card) || hasGame(new File(card, "assets"))) return;
        tellWhereTheGameGoes();
    }

    // a disc image in the folder, or default.xbe with the game's data beside it, as native/game_files.cpp looks for them
    private static boolean hasGame(File dir) {
        File[] files = dir.listFiles();
        if (files == null) return false;
        boolean data = false;
        for (File f : files) {
            String name = f.getName().toLowerCase();
            if (name.endsWith(".iso")) return true;
            if (!name.endsWith(".xbe")) data = true;
        }
        return data && new File(dir, "default.xbe").exists();
    }

    private void askAllFilesAccess() {
        dialog = new AlertDialog.Builder(this)
                .setTitle("Storage access")
                .setMessage("Conflict: Global Terror reads the game from a CGT folder on your phone's storage:\n\n"
                        + "/storage/emulated/0/" + SHARED + "\n\n"
                        + (ALL_FILES_ACCESS
                                ? "On the next screen turn on \"Allow access to manage all files\" for Conflict: Global Terror, then come back."
                                : "Allow Conflict: Global Terror to access your storage on the next screen."))
                .setCancelable(false)
                .setPositiveButton("Allow access", (d, w) -> openAccessSettings())
                .setNegativeButton("Not now", (d, w) -> notNow = true)
                .setNeutralButton("Import game", (d, w) -> startImport())
                .show();
    }

    private void tellWhereTheGameGoes() {
        dialog = new AlertDialog.Builder(this)
                .setTitle("Game files")
                .setMessage("Copy the game into a CGT folder on your phone's storage:\n\n"
                        + "/storage/emulated/0/" + SHARED + "\n\n"
                        + "or into a CGT folder at the top of your SD card.\n\n"
                        + "Its disc image (.iso) of the USA release, or default.xbe with the data folder and the mission "
                        + "files beside it. The game starts as soon as they are there.")
                .setPositiveButton("OK", null)
                .setNeutralButton("Import game", (d, w) -> startImport())
                .show();
    }

    private void openAccessSettings() {
        if (!ALL_FILES_ACCESS) {
            boolean ask = !storageAsked || shouldShowRequestPermissionRationale(Manifest.permission.READ_EXTERNAL_STORAGE);
            if (ask) {
                storageAsked = true;
                requestPermissions(new String[] {Manifest.permission.READ_EXTERNAL_STORAGE,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE}, STORAGE_REQUEST);
            } else {
                startActivity(new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                        Uri.parse("package:" + getPackageName())));
            }
            return;
        }
        try {
            startActivity(new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    Uri.parse("package:" + getPackageName())));
        } catch (ActivityNotFoundException e) {
            try {
                startActivity(new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
            } catch (ActivityNotFoundException e2) {
                startActivity(new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                        Uri.parse("package:" + getPackageName())));
            }
        }
    }

    // the folder on each SD card or USB stick mounted, as native/game_files.cpp looks in them
    private List<File> cardFolders() {
        List<File> out = new ArrayList<>();
        if (!ALL_FILES_ACCESS) {
            // Android 10: each card's root above the app's own folder on it, <root>/Android/data/<package>/files
            File[] own = getExternalFilesDirs(null);
            for (int i = 1; i < own.length; i++) {
                File root = own[i];
                for (int up = 0; root != null && up < 4; up++) root = root.getParentFile();
                if (root != null) out.add(new File(root, SHARED));
            }
            return out;
        }
        StorageManager sm = getSystemService(StorageManager.class);
        if (sm == null) return out;
        for (StorageVolume v : sm.getStorageVolumes()) {
            File root = v.getDirectory();
            if (v.isPrimary() || root == null || !Environment.MEDIA_MOUNTED.equals(v.getState())) continue;
            out.add(new File(root, SHARED));
        }
        return out;
    }

    // the same folders for native/game_files.cpp, from the game's thread: some phones don't let the app list /storage,
    // though each card's folder opens
    public String[] cardFolderPaths() {
        List<File> cards = cardFolders();
        String[] out = new String[cards.size()];
        for (int i = 0; i < out.length; i++) out[i] = cards.get(i).getPath();
        return out;
    }

    // the disc image the player picks, copied where the game looks: the shared folder with all files access, else the
    // app's own assets folder, which needs none
    private void startImport() {
        importing = importer.pick();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode != GameImporter.REQUEST) {
            super.onActivityResult(requestCode, resultCode, data);
            return;
        }
        Uri uri = resultCode == RESULT_OK && data != null ? data.getData() : null;
        if (uri == null) {
            importing = false;  // the picker was closed: onResume asks again
            return;
        }
        File dir = hasStorageAccess()
                ? new File(Environment.getExternalStorageDirectory(), SHARED)
                : new File(getExternalFilesDir(null), "assets");
        importer.copy(uri, dir, ok -> {
            importing = false;
            if (!ok) askForGame();
        });
    }
}
