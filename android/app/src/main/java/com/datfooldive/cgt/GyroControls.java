package com.datfooldive.cgt;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;

final class GyroControls implements SensorEventListener {
    private final CGTActivity activity;
    private final SensorManager manager;
    private final Sensor sensor;
    private final GyroMotion motion = new GyroMotion();
    private boolean listening;

    private static native void nativeGyroLook(float yaw, float pitch);

    GyroControls(CGTActivity activity) {
        this.activity = activity;
        manager = activity.getSystemService(SensorManager.class);
        sensor = manager == null ? null : manager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
    }

    boolean available() { return sensor != null; }

    void setActive(boolean active) {
        active &= available();
        if (active == listening) return;
        motion.reset();
        nativeGyroLook(0, 0);
        if (active) {
            listening = manager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_GAME);
            if (!listening) CGTActivity.showToast("Could not start gyroscope");
        } else {
            listening = false;
            manager.unregisterListener(this);
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (!listening) return;
        int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        if (motion.update(event.timestamp, rotation, event.values[0], event.values[1]))
            nativeGyroLook(motion.yaw, motion.pitch);
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) { }
}
