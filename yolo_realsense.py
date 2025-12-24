# ----------------------------------------------------------------------
# RealSense D455 + YOLO + OSC sender
# ----------------------------------------------------------------------
from pysofaconventions import SOFAFile
from scipy import signal
from scipy.spatial.distance import cdist
import threading
import soundfile as sf
import sounddevice as sd
from pythonosc import dispatcher, osc_server
from pythonosc.udp_client import SimpleUDPClient
from queue import Queue
import numpy as np
import time
from os.path import join

import pyrealsense2 as rs
import cv2
from ultralytics import YOLO

from yolo_sofa import SpatialSoundHeadphoneYOLO


class RealSenseD455YOLOOSC:
    """
    Handles Intel RealSense D455 streaming, YOLO inference, and sends detections via OSC.

    - Captures color + depth frames from D455
    - Runs YOLO on the color frame
    - For each detection, computes:
        - x_norm, y_norm in [0, 1] (bbox center)
        - depth_m at the center pixel from depth frame

    Message format (to SOFA side):

        /yolo [x_norm, class_id_or_label, y_norm, depth_m, frame_index]

    Sender-side logic:
    ------------------
    1) CID + position stability:
       - Only consider a detection if the same cid stays in the same
         local x-region for at least cid_stability_threshold (0.2 s).
       - Flickers shorter than 200 ms are ignored.

    2) Per-frame scene:
       - For each frame, we collect at most one candidate per class (highest conf).
       - We send ALL stable candidates for that frame.
       - Fairness across classes within a scene is then handled on the SOFA side.
    """

    def __init__(self,
                 model_path="yolov8n.pt",
                 osc_ip="127.0.0.1",
                 osc_port=6969,
                 conf_threshold=0.3,
                 show_debug=False,
                 x_size=640, y_size=480,
                 fps=30):

        self.model = YOLO(model_path).to('cuda')
        self.conf_threshold = conf_threshold
        self.show_debug = show_debug
        self.max_det = 10

        self.cls_names = self.model.names

        # OSC client to talk to SpatialSoundHeadphoneYOLO
        self.osc_client = SimpleUDPClient(osc_ip, osc_port)

        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.x_size = x_size
        self.y_size = y_size
        self.fps = fps

        # Configure streams
        self.config.enable_stream(rs.stream.color, self.x_size, self.y_size,
                                  rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.x_size, self.y_size,
                                  rs.format.z16, self.fps)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Get depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"RealSense depth scale: {self.depth_scale} m/unit")

        # --- CID + position stability configuration --------------------
        # 0.2 s = 200 ms threshold for "stable" presence in a local area
        self.cid_stability_threshold = 0.2  # seconds

        # How many horizontal buckets to divide x into (0..1)
        self.position_bucket_count = 10

        # If a cid hasn't been seen in a bucket for this long, reset its window
        self.cid_stability_reset_time = 1.0  # seconds

        # (cid, bucket) -> {"first_seen": t0, "last_seen": t_last}
        self.cid_position_state = {}

        # Frame counter so SOFA can group detections by frame
        self.frame_index = 0

    def stop(self):
        self.pipeline.stop()

    # --- helper to quantise x into a "local area" bucket ----------
    def _x_to_bucket(self, x_norm: float) -> int:
        x_clamped = max(0.0, min(1.0, float(x_norm)))
        bucket = int(x_clamped * self.position_bucket_count)
        if bucket == self.position_bucket_count:
            bucket -= 1
        return bucket

    # --- CID + position stability check ---------------------------
    def _is_cid_position_stable(self, cid, x_norm: float, now: float = None) -> bool:
        """
        Return True only if this (cid, local x-region) has been observed
        for at least self.cid_stability_threshold seconds.

        If it only appears briefly (< threshold) in that region,
        this returns False and we skip it.
        """
        if now is None:
            now = time.time()

        bucket = self._x_to_bucket(x_norm)
        key = (cid, bucket)

        state = self.cid_position_state.get(key)

        # First time we see this cid in this bucket -> start a stability window
        if state is None:
            self.cid_position_state[key] = {"first_seen": now, "last_seen": now}
            print(f"[STAB] CID {cid} bucket {bucket}: start window")
            return False

        # If it's been gone from this bucket for too long, reset window
        if now - state["last_seen"] > self.cid_stability_reset_time:
            self.cid_position_state[key] = {"first_seen": now, "last_seen": now}
            print(f"[STAB] CID {cid} bucket {bucket}: reset window (gap "
                  f"{now - state['last_seen']:.3f}s)")
            return False

        # Update last_seen and compute stable duration
        state["last_seen"] = now
        stable_duration = now - state["first_seen"]

        if stable_duration < self.cid_stability_threshold:
            print(f"[STAB] CID {cid} bucket {bucket}: {stable_duration:.3f}s "
                  f"< {self.cid_stability_threshold:.3f}s -> not stable")
            return False

        print(f"[STAB] CID {cid} bucket {bucket}: STABLE for {stable_duration:.3f}s")
        return True

    # -------------------------------------------------------------------
    def run(self):
        try:
            while True:
                self.frame_index += 1

                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                h, w, _ = color_image.shape

                # YOLO inference
                results = self.model(color_image, verbose=False,
                                     max_det=self.max_det)
                result = results[0]

                now = time.time()

                # Per-frame candidate store: one best detection per class
                # cid -> dict(conf, x_norm, y_norm, depth_m, cls_name)
                stable_candidates = {}

                # Collect stable detections
                if result.boxes is not None:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf < self.conf_threshold:
                            continue

                        cls_id = int(box.cls[0])
                        cls_name = self.cls_names[cls_id]
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy

                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0

                        # Normalised coordinates
                        x_norm = cx / w
                        y_norm = cy / h

                        # CID + position stability gating
                        if not self._is_cid_position_stable(cls_id, x_norm, now):
                            continue

                        # Depth at center pixel (in meters)
                        depth_m = float(depth_frame.get_distance(int(cx), int(cy)))

                        # Keep the highest-confidence candidate per class in this frame
                        prev = stable_candidates.get(cls_id)
                        if (prev is None) or (conf > prev["conf"]):
                            stable_candidates[cls_id] = {
                                "conf": conf,
                                "x_norm": x_norm,
                                "y_norm": y_norm,
                                "depth_m": depth_m,
                                "cls_name": cls_name,
                            }

                        # Optional debug draw
                        if self.show_debug:
                            label = f"{cls_id}:{conf:.2f}"
                            cv2.rectangle(color_image,
                                          (int(x1), int(y1)),
                                          (int(x2), int(y2)),
                                          (0, 255, 0),
                                          2)
                            cv2.circle(color_image, (int(cx), int(cy)), 3,
                                       (0, 0, 255), -1)
                            cv2.putText(color_image, label,
                                        (int(x1), int(y1) - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 255, 0), 1)

                # Send ALL stable candidates for this frame as a "scene"
                if stable_candidates:
                    for cid, cand in stable_candidates.items():
                        x_norm = float(cand["x_norm"])
                        y_norm = float(cand["y_norm"])
                        depth_m = float(cand["depth_m"])
                        cls_name = cand["cls_name"]

                        msg = [x_norm, cls_name, y_norm, depth_m, self.frame_index]
                        print(f"[SEND] frame={self.frame_index} cid={cid}({cls_name}) "
                              f"x={x_norm:.3f} y={y_norm:.3f} d={depth_m:.2f}")
                        self.osc_client.send_message("/yolo", msg)

                if self.show_debug:
                    cv2.imshow("RealSense YOLO", color_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break

        except KeyboardInterrupt:
            print("Stopping RealSenseD455YOLOOSC (KeyboardInterrupt)")
        finally:
            self.stop()
            if self.show_debug:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    # Start spatial sound OSC server in a background thread
    ss = SpatialSoundHeadphoneYOLO(verbose=True)
    sound_thread = threading.Thread(target=ss.start, daemon=True)
    sound_thread.start()

    # Start RealSense + YOLO + OSC loop in main thread
    rs_yolo = RealSenseD455YOLOOSC(
        model_path="yolov8n.pt",   # change to your model path
        osc_ip="127.0.0.1",
        osc_port=6969,
        conf_threshold=0.3,
        show_debug=True           # set False if you don't want OpenCV window
    )
    rs_yolo.run()
