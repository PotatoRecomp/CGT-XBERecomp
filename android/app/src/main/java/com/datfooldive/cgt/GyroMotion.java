package com.datfooldive.cgt;

// Android sensor axes stay in the device's natural orientation when the display rotates.
final class GyroMotion {
    private long timestamp;
    private int rotation = -1;
    float yaw, pitch;

    void reset() {
        timestamp = 0;
        rotation = -1;
        yaw = pitch = 0;
    }

    boolean update(long time, int displayRotation, float x, float y) {
        long previous = timestamp;
        int previousRotation = rotation;
        timestamp = time;
        rotation = displayRotation;
        yaw = pitch = 0;
        float dt = (time - previous) * 1e-9f;
        if (previous == 0 || previousRotation != rotation || dt <= 0 || dt > 0.1f
                || !Float.isFinite(x) || !Float.isFinite(y)) return false;
        float screenX, screenY;
        switch (rotation) {
            case 1: screenX = -y; screenY = x; break;   // Surface.ROTATION_90
            case 2: screenX = -x; screenY = -y; break;  // Surface.ROTATION_180
            case 3: screenX = y; screenY = -x; break;   // Surface.ROTATION_270
            default: screenX = x; screenY = y; break;
        }
        // TYPE_GYROSCOPE supplies Android's bias calibration; ignore its remaining stationary noise.
        float degrees = dt * (180f / (float) Math.PI);
        yaw = Math.abs(screenY) < 0.005f ? 0 : -screenY * degrees;
        pitch = Math.abs(screenX) < 0.005f ? 0 : -screenX * degrees;
        return yaw != 0 || pitch != 0;
    }
}
