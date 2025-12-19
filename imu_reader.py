#!/usr/bin/env python3
import threading

import pyrealsense2 as rs

class IMUReader:
    """Reads accel/gyro directly from the Motion Module sensor via callback."""
    def __init__(self, device: rs.device, accel_hz=250, gyro_hz=400):
        self.lock = threading.Lock()
        self.accel = None
        self.gyro = None

        # Find a sensor that actually has accel/gyro profiles (don't rely on name)
        self.motion_sensor = None
        for s in device.query_sensors():
            has_accel = False
            has_gyro = False
            for p in s.get_stream_profiles():
                if p.stream_type() == rs.stream.accel:
                    has_accel = True
                elif p.stream_type() == rs.stream.gyro:
                    has_gyro = True
            if has_accel and has_gyro:
                self.motion_sensor = s
                break

        if self.motion_sensor is None:
            # Helpful debug: list sensor names we did see
            names = []
            for s in device.query_sensors():
                try:
                    names.append(s.get_info(rs.camera_info.name))
                except Exception:
                    names.append("<unknown>")
            raise RuntimeError(f"No accel+gyro sensor found. Sensors seen: {names}")

        # Select stream profiles that match supported rates
        accel_prof = None
        gyro_prof = None
        for p in self.motion_sensor.get_stream_profiles():
            if p.stream_type() == rs.stream.accel and p.format() == rs.format.motion_xyz32f and p.fps() == accel_hz:
                accel_prof = p
            if p.stream_type() == rs.stream.gyro and p.format() == rs.format.motion_xyz32f and p.fps() == gyro_hz:
                gyro_prof = p

        if accel_prof is None or gyro_prof is None:
            raise RuntimeError(f"Could not find accel@{accel_hz} and gyro@{gyro_hz} motion profiles")

        self.motion_sensor.open([accel_prof, gyro_prof])
        self.motion_sensor.start(self._cb)

    def _cb(self, f):
        if not f.is_motion_frame():
            return
        m = f.as_motion_frame()
        data = m.get_motion_data()
        st = m.get_profile().stream_type()
        with self.lock:
            if st == rs.stream.accel:
                self.accel = (data.x, data.y, data.z)
            elif st == rs.stream.gyro:
                self.gyro = (data.x, data.y, data.z)

    def get_latest(self):
        with self.lock:
            return self.accel, self.gyro

    def stop(self):
        try:
            self.motion_sensor.stop()
        except Exception:
            pass
        try:
            self.motion_sensor.close()
        except Exception:
            pass
