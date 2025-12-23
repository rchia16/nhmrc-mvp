from pysofaconventions import SOFAFile
from scipy import signal
from scipy.spatial.distance import cdist
import threading
import soundfile as sf
import sounddevice as sd
from pythonosc import dispatcher, osc_server
from queue import Queue
import numpy as np
import time
from os.path import join

# --- SOFA files ---
sofa_lib = './sofa-lib/'
sofa_glasses_file = join(sofa_lib, "./BRIR_HATS_3degree_for_glasses.sofa")

# --- Audio files ---
sound_lib = './sound-lib/'
arcade_sound_file = join(sound_lib, 'mixkit-arcade-mechanical-bling-210.wav')
retro_sound_file = join(sound_lib, 'mixkit-retro-game-notification-212.wav')
unlock_sound_file = join(sound_lib, 'mixkit-unlock-game-notification-253.wav')
shortz_sound_file = join(sound_lib, 'Shortzz.mp3')
scifi_sound_file = join(sound_lib, 'ui_sci-fi-sound-36061.wav')  # default


class SpatialSoundHeadphoneYOLO:
    """
    Scene-based OSC + BRIR spatialisation for YOLO detections.

    OSC:
        /yolo <x_position> [class_id_or_label, y_norm, depth_m, frame_index]

        x_position:
            - 0..1  (normalised): 0 = far right, 1 = far left
            - 0..image_width (pixels): will be normalised assuming image_width

        class_id_or_label (optional):
            - int: YOLO class index (e.g. 0, 1, 2, ...)
            - str: label (e.g. "person", "door")

        frame_index:
            - integer ID of the camera frame; used to group detections
              into a single "scene".

    Behaviour:
        - Each frame_index defines a SCENE: all detections with that index
          are grouped together.
        - For each scene, we play ALL detected objects sequentially,
          one sound at a time (no overlap).
        - Within a scene, objects are played sorted by azimuth
          (left-to-right in auditory space).
        - After a scene finishes, all queued scenes are flushed and we
          keep only the latest frame that arrived while playing.
        - Sender does stability; this side focuses on:
            * full scene comprehension
            * fairness across classes within a scene (each played once).
    """

    def __init__(self,
                 sofa_file_path=sofa_glasses_file,
                 image_width=640.0,
                 verbose=False):
        # ---- Config ----
        self.image_width = float(image_width)
        self.yolo_counter = 0
        self.message = None
        self.playing = False

        self.verbose = verbose

        # Optional per-class cooldown across scenes (can be small or 0.0)
        self.min_interval_between_plays = 0.0
        self.last_play_time = {}  # cid_or_label -> last time played

        # ---- OSC dispatcher ----
        osc_dispatcher = dispatcher.Dispatcher()
        osc_dispatcher.map("/yolo", self.yolo_handler)

        self.OSCserver = osc_server.ThreadingOSCUDPServer(
            ('0.0.0.0', 6969),
            osc_dispatcher
        )

        # ---- Load SOFA / BRIR ----
        sofa = SOFAFile(sofa_file_path, 'r')
        self.BRIRs = sofa.getDataIR()
        self.BRIR_samplerate = sofa.getSamplingRate()
        self.BRIR_sourcePositions = sofa.getVariableValue('SourcePosition')  # phi, theta, r

        # ---- Audio device / samplerate ----
        # sd.default.device = 'Speakers (Realtek(R) Audio), Windows DirectSound'
        sd.default.samplerate = self.BRIR_samplerate

        # ---- Load YOLO sound sources (different objects → different sounds) ----
        self.setup_audio_sources()

        # ---- Scene management ----
        # frame_id -> {cid: {"spherical": np.array, "sound_key": str}}
        self.pending_scenes = {}
        # Frame IDs in the order they arrived
        self.scene_order = []

        # Current scene playback state
        self.current_scene_id = None
        self.current_scene_objects = []   # list of (cid, obj_dict)
        self.current_scene_index = 0

        # ---- Playback thread ----
        self.read_queue_manager_thread = threading.Thread(
            target=self.read_queue_manager,
            args=(),
            daemon=True
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup_audio_sources(self):
        """
        Load multiple sound files and map them to class IDs / labels.
        You can customise the mapping in self.yolo_sounds.
        """
        self.yolo_sounds = {}  # key -> (audio_array, samplerate)

        def safe_load(path):
            try:
                audio, fs = sf.read(path)
                print(f"Loaded audio: {path}")
                return audio, fs
            except Exception as e:
                print(f"Error loading audio file {path}: {e}")
                return None, None

        # Load all candidate sounds
        audio_default, fs_default = safe_load(scifi_sound_file)
        audio_retro,   fs_retro   = safe_load(retro_sound_file)
        audio_arcade,  fs_arcade  = safe_load(arcade_sound_file)
        audio_unlock,  fs_unlock  = safe_load(unlock_sound_file)
        audio_shortz,  fs_shortz  = safe_load(shortz_sound_file)

        # Default sound (used when no specific class mapping found)
        if audio_default is not None:
            self.yolo_sounds["default"] = (audio_default, fs_default)

        # Example numeric class mappings (adjust to your YOLO classes of interest)
        if audio_retro is not None:
            self.yolo_sounds[0] = (audio_retro, fs_retro)   # class 0
        if audio_arcade is not None:
            self.yolo_sounds[1] = (audio_arcade, fs_arcade)  # class 1
        if audio_unlock is not None:
            self.yolo_sounds[2] = (audio_unlock, fs_unlock)  # class 2
        if audio_shortz is not None:
            self.yolo_sounds[3] = (audio_shortz, fs_shortz)  # class 3

        # Optional label-based mappings; change these to your own labels
        if audio_retro is not None:
            self.yolo_sounds["person"] = (audio_retro, fs_retro)
        if audio_unlock is not None:
            self.yolo_sounds["door"] = (audio_unlock, fs_unlock)
        if audio_arcade is not None:
            self.yolo_sounds["object"] = (audio_arcade, fs_arcade)
        if audio_shortz is not None:
            self.yolo_sounds["chair"] = (audio_shortz, fs_shortz)

        print("YOLO sound mapping keys:", list(self.yolo_sounds.keys()))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def start(self):
        print('Starting OSC server for YOLO detections...')
        self.read_queue_manager_thread.start()
        self.OSCserver.serve_forever()

    # ------------------------------------------------------------------
    # Scene-based playback manager
    # ------------------------------------------------------------------
    def read_queue_manager(self):
        """
        Playback loop:

        - If no current scene, pick the latest frame_id from scene_order,
          freeze its objects as current_scene_objects (and flush older scenes).
        - For the current scene, play each object's sound in turn
          (one at a time), sorted by azimuth.
        - When all objects in a scene are played, move to the next scene,
          again flushing to keep only the latest frame.
        """
        while True:
            if not self.playing:
                # If we don't have an active scene or we've finished the current one,
                # select the next scene if available.
                if self.current_scene_id is None or \
                   self.current_scene_index >= len(self.current_scene_objects):

                    if self.current_scene_id is not None and self.verbose:
                        print(f"[SCENE] Finished scene {self.current_scene_id}")

                    self.current_scene_id = None
                    self.current_scene_objects = []
                    self.current_scene_index = 0

                    if not self.scene_order:
                        # No scenes waiting; idle for a bit
                        time.sleep(0.05)
                        continue

                    # --- CHANGED: pick the latest frame and flush older scenes ---
                    # latest frame_id seen so far
                    latest_scene_id = max(self.scene_order)
                    # keep only this scene in pending_scenes / scene_order
                    for sid in list(self.pending_scenes.keys()):
                        if sid != latest_scene_id:
                            if self.verbose:
                                print(f"[SCENE] Flushing old scene {sid}")
                            self.pending_scenes.pop(sid, None)
                    self.scene_order = [latest_scene_id]

                    next_scene_id = latest_scene_id
                    scene_dict = self.pending_scenes.pop(next_scene_id, {})

                    if not scene_dict:
                        # Empty scene, skip
                        continue

                    # Turn dict (cid -> obj) into a list for iteration
                    items = list(scene_dict.items())

                    # --- CHANGED: sort by azimuth (spherical[0]) rather than cid ---
                    # azimuth is in degrees, [-90, 90] in our mapping
                    items.sort(key=lambda kv: kv[1]["spherical"][0])
                    # ---------------------------------------------------------------

                    self.current_scene_id = next_scene_id
                    self.current_scene_objects = items
                    self.current_scene_index = 0

                    if self.verbose:
                        az_list = [obj["spherical"][0] for _, obj in items]
                        print(f"[SCENE] Starting scene {self.current_scene_id} "
                              f"with {len(items)} objects, azimuths={az_list}")

                # If we now have a scene, attempt to play the next object
                if self.current_scene_id is not None and \
                   self.current_scene_index < len(self.current_scene_objects):

                    cid, obj = self.current_scene_objects[self.current_scene_index]
                    sound_key = obj["sound_key"]
                    spherical = obj["spherical"]

                    if sound_key not in self.yolo_sounds:
                        if self.verbose:
                            print(f"[SCENE] No sound for key {sound_key}, skipping.")
                        self.current_scene_index += 1
                        continue

                    audio_src, fs = self.yolo_sounds[sound_key]

                    # Optional per-class cooldown across scenes (not within scene)
                    now = time.time()
                    last_t = self.last_play_time.get(cid, -1e9)
                    if self.min_interval_between_plays > 0.0 and \
                       now - last_t < self.min_interval_between_plays:
                        if self.verbose:
                            print(f"[SCENE] cid={cid} still in cooldown ({now - last_t:.2f}s), "
                                  f"skipping this occurrence.")
                        self.current_scene_index += 1
                        continue

                    # Prepare and play this object's sound
                    self.message = {
                        "caller": "scene playback",
                        "yolo_spherical": spherical,
                        "yolo_sound_key": sound_key,
                    }

                    self.yolo_counter += 1
                    audio_to_play = self.prepare_sound_yolo(
                        self.yolo_counter,
                        audio_src,
                        is_mono=True
                    )

                    if audio_to_play is not None:
                        if self.verbose:
                            print(f"[SCENE] Playing scene {self.current_scene_id}, "
                                  f"object index {self.current_scene_index}, cid={cid}, "
                                  f"key={sound_key}, az={spherical[0]:.1f}")
                        self.playing = True
                        self.play_sound(audio_to_play)
                        self.playing = False
                        self.last_play_time[cid] = time.time()
                    else:
                        if self.verbose:
                            print(f"[SCENE] prepare_sound_yolo returned None for cid={cid}")

                    # Move to next object in the scene
                    self.current_scene_index += 1
                else:
                    # Nothing to play right now
                    time.sleep(0.02)
            else:
                # Shouldn't really happen because play_sound is blocking,
                # but keep for safety.
                time.sleep(0.01)

    # ------------------------------------------------------------------
    # OSC handler for YOLO x-position + class + frame_id
    # ------------------------------------------------------------------
    def yolo_handler(self, address, *args):
        """
        Handle YOLO x-position + cid + optional y, depth, frame_index.

        /yolo <x_position> [cid, y_norm, depth_m, frame_index]
        """

        if len(args) == 0:
            if self.verbose:
                print("YOLO handler called without x-position argument.")
            return

        raw_x = float(args[0])
        extra = args[1:]

        # Get class key (cid) if available
        if len(extra) >= 1:
            class_info = extra[0]
            if isinstance(class_info, (int, float)):
                cid = int(class_info)
            elif isinstance(class_info, str):
                cid = class_info
            else:
                cid = "default"
        else:
            cid = "default"

        # Try to get frame_id from the last extra field (if present)
        frame_id = None
        if len(extra) >= 4:
            try:
                frame_id = int(extra[3])
            except (ValueError, TypeError):
                frame_id = None

        # If sender didn't supply frame_id, treat each message as its own scene
        if frame_id is None:
            frame_id = -1  # a special value; each such message will overwrite its scene

        if self.verbose:
            print(f"~~~ Received YOLO message: x={raw_x}, cid={cid}, frame={frame_id}, extra={extra}")

        # Normalise x (supports either 0..1 or pixel coords)
        if raw_x > 1.0:
            x_norm = max(0.0, min(1.0, raw_x / self.image_width))
        else:
            x_norm = max(0.0, min(1.0, raw_x))

        # Convert to azimuth
        azimuth_deg = -90.0 + 180.0 * x_norm
        elevation_deg = 0.0
        spherical_coord = np.array([azimuth_deg, elevation_deg, 1.0])

        if self.verbose:
            print(f"~~~ spherical coord (az, el, r) = {spherical_coord}")

        # Resolve sound for this cid
        sound_key = cid
        if sound_key not in self.yolo_sounds:
            if self.verbose:
                print(f"No specific sound for key {sound_key}, falling back to default/first.")
            if "default" in self.yolo_sounds:
                sound_key = "default"
            elif self.yolo_sounds:
                sound_key = next(iter(self.yolo_sounds))
            else:
                if self.verbose:
                    print("No sounds loaded at all – nothing to play.")
                return

        # Register this object into the appropriate scene
        scene = self.pending_scenes.get(frame_id)
        if scene is None:
            scene = {}
            self.pending_scenes[frame_id] = scene
            # New scene arrival -> add to play order
            self.scene_order.append(frame_id)

        # If multiple detections of the same cid in the same frame arrive,
        # we keep only the latest (overwrites previous for that cid).
        scene[cid] = {
            "spherical": spherical_coord,
            "sound_key": sound_key,
        }

    # ------------------------------------------------------------------
    # Playback helpers
    # ------------------------------------------------------------------
    def play_sound(self, signal_to_play):
        sd.play(signal_to_play, mapping=[1, 2])
        sd.wait()
        if self.verbose:
            print('!----------------- END sound')

    # ------------------------------------------------------------------
    # Sound preparation
    # ------------------------------------------------------------------
    def prepare_sound_yolo(self, counter, audio_src, is_mono=True):
        if self.verbose:
            print(f"~~~ prepare YOLO audio {counter}")
        if audio_src is None:
            if self.verbose:
                print("YOLO audio source is None.")
            return None

        # Downmix stereo -> mono for BRIR convolution
        if is_mono and audio_src.ndim == 2:
            audio_src = audio_src.mean(axis=1)

        try:
            spherical = self.message['yolo_spherical']
            audio = self.sound_process(
                spherical,
                audio_src,
                self.BRIRs,
                self.BRIR_sourcePositions
            )
        except Exception as e:
            print(f"Error during YOLO sound processing: {e}")
            return None

        if audio is not None:
            len_audio_src = audio_src.shape[0]
            signal_to_play = audio[:len_audio_src]
            if self.verbose:
                print(f"~~~ END prepare YOLO audio {counter}")
            return signal_to_play

        return None

    # ------------------------------------------------------------------
    # BRIR processing
    # ------------------------------------------------------------------
    def sound_process(self, data, audio, BRIRs, sourcePositions):
        """
        data: [az_deg, el_deg, r]
        audio: 1D mono
        """
        data = np.array(data, dtype=float)

        # Wrap azimuth into [0, 360] to match SOFA convention if needed
        if data[0] < 0:
            data[0] += 360.0
        if data[0] > 360.0:
            data[0] -= 360.0

        distances = cdist(data[np.newaxis, :], sourcePositions, metric='euclidean')
        index_source = np.argmin(distances)
        BRIR_this = BRIRs[index_source, :, :]

        signal_this_left = signal.convolve(10 * audio, BRIR_this[0, :], mode='full')
        signal_this_right = signal.convolve(10 * audio, BRIR_this[1, :], mode='full')
        signal_this = np.stack((signal_this_left, signal_this_right), axis=-1)

        return signal_this


if __name__ == "__main__":
    MySS = SpatialSoundHeadphoneYOLO(verbose=True)
    MySS.start()
