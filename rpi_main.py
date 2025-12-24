#!/usr/bin/env python3
import argparse
import threading
import time
import signal
import sys
import numpy as np

from config import load_config, deep_get
from udp_pcm_stream import PCMStreamPlayerApp, PCMStreamParams
from yolo_sofa import DepthAwareSpatialSound, SpatialSoundHeadphoneYOLO
from pythonosc import dispatcher, osc_server



def _run_rs_sender(cfg, stop_evt: threading.Event):
    import rs_d455_raw_udp_sender as rs_sender

    argv = [
        "rs_d455_raw_udp_sender.py",
        "--pc-ip", deep_get(cfg, "network.pc_ip"),
        "--pc-port", str(deep_get(cfg, "ports.rs_udp")),
        "--color-w", str(deep_get(cfg, "realsense.color_w")),
        "--color-h", str(deep_get(cfg, "realsense.color_h")),
        "--depth-w", str(deep_get(cfg, "realsense.depth_w")),
        "--depth-h", str(deep_get(cfg, "realsense.depth_h")),
        "--fps", str(deep_get(cfg, "realsense.fps")),
        "--mtu-payload", str(deep_get(cfg, "realsense.mtu_payload")),
    ]
    if deep_get(cfg, "realsense.no_depth", False):
        argv.append("--no-depth")

    saved = sys.argv[:]
    try:
        sys.argv = argv
        rs_sender.main()
    finally:
        sys.argv = saved
        stop_evt.set()


def _run_ppg_stream(cfg, stop_evt: threading.Event):
    from ppg_stream import App

    app = App(
        send_ip=deep_get(cfg, "network.pc_ip"),
        send_port=int(deep_get(cfg, "ports.ppg_tcp")),
        plot_window=2000,
        no_plot=True,
        log_dir=deep_get(cfg, "ppg.log_dir"),
        rotate_every_seconds=int(deep_get(cfg, "ppg.rotate_seconds")),
        fsync_every=int(deep_get(cfg, "ppg.fsync_every")),
        rate_print=bool(deep_get(cfg, "ppg.rate_print")),
        force_poll=bool(deep_get(cfg, "ppg.force_poll")),
        no_data_timeout=float(deep_get(cfg, "ppg.no_data_timeout")),
        poll_sleep_ms=float(deep_get(cfg, "ppg.poll_sleep_ms")),
    )
    try:
        app.run()
    finally:
        stop_evt.set()


def _run_audio_bt_receiver_wait(cfg, stop_evt: threading.Event):
    from bth_audio_manager import UDPAudioBluetoothReceiverApp

    app = UDPAudioBluetoothReceiverApp(
        bt_mac=deep_get(cfg, "audio.bt_mac"),
        udp_listen_ip=deep_get(cfg, "network.pi_listen_ip", "0.0.0.0"),
        udp_port=int(deep_get(cfg, "ports.audio_udp")),
        pair=bool(deep_get(cfg, "audio.pair")),
        trust=bool(deep_get(cfg, "audio.trust")),
        reconnect_every_s=float(deep_get(cfg, "audio.bt_reconnect_every")),
        reassembly_timeout_s=float(deep_get(cfg, "audio.recv_reassembly_timeout_s")),
        keep_files=bool(deep_get(cfg, "audio.keep_audio_files")),
        debug_udp=bool(deep_get(cfg, "audio.debug_udp")),
    )
    try:
        app.run_forever()
    finally:
        stop_evt.set()

def _run_depth_ordered_spatialiser(cfg, stop_evt: threading.Event):
    """
    Listen for classified object coordinates (x, y, depth) and spatialise
    their associated sounds using the SOFA BRIR. Objects within a scene are
    played from nearest to farthest.
    """

    sofa_path = deep_get(cfg, "sonification.sofa", "./sofa-lib/BRIR_HATS_3degree_for_glasses.sofa")
    image_width = float(deep_get(cfg, "realsense.color_w", 640))
    osc_port = int(deep_get(cfg, "ports.audio_udp", 40100))

    app = DepthAwareSpatialSound(sofa_file_path=sofa_path,
                                 image_width=image_width, osc_port=osc_port,
                                 verbose=True)

    thread = threading.Thread(target=app.start, daemon=True)
    thread.start()

    try:
        while not stop_evt.is_set():
            time.sleep(0.2)
    finally:
        try:
            app.OSCserver.server_close()
        except Exception:
            pass

def _run_audio_bt_receiver(cfg, stop_evt: threading.Event):
    # Streaming mode (long-term)
    if bool(deep_get(cfg, "audio.stream.enabled", False)):
        p = PCMStreamParams(
            sample_rate=int(deep_get(cfg, "audio.stream.sample_rate", 16000)),
            channels=int(deep_get(cfg, "audio.stream.channels", 2)),
            frame_ms=int(deep_get(cfg, "audio.stream.frame_ms", 20)),
            jitter_ms=int(deep_get(cfg, "audio.stream.jitter_ms", 120)),
            max_jitter_ms=int(deep_get(cfg, "audio.stream.max_jitter_ms", 400)),
            bind_ip=str(deep_get(cfg, "network.pi_bind_ip", "0.0.0.0")),
            mtu_payload=int(deep_get(cfg, "audio.stream.mtu_payload", 1400)),
        )
        app = PCMStreamPlayerApp(
            listen_port=int(deep_get(cfg, "audio.stream.port", 50030)),
            params=p,
            alsa_device=deep_get(cfg, "audio.stream.alsa_device", None),
            debug=bool(deep_get(cfg, "audio.debug_udp", True)),
        )
        app.start()
    else:
        # Legacy file-based playback
        from bth_audio_manager import UDPAudioBluetoothReceiverApp
        app = UDPAudioBluetoothReceiverApp(
            bind_ip=str(deep_get(cfg, "network.pi_bind_ip", "0.0.0.0")),
            port=int(deep_get(cfg, "audio.rx_port", 50020)),
            bt_mac=str(deep_get(cfg, "audio.bt_mac", "")),
            tmp_dir=str(deep_get(cfg, "audio.tmp_dir", "/tmp/udp_audio")),
            keep_files=bool(deep_get(cfg, "audio.keep_audio_files", False)),
            debug=bool(deep_get(cfg, "audio.debug_udp", True)),
        )
        app.start()

    try:
        while not stop_evt.is_set():
            time.sleep(0.2)
    finally:
        try:
            app.stop()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    help="Path to nhmrc_streaming_config.yaml",
                    default="./streaming_config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    stop_evt = threading.Event()

    def _sig_handler(_sig, _frame):
        stop_evt.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    threads = [
        threading.Thread(target=_run_rs_sender, args=(cfg, stop_evt), daemon=True),
        threading.Thread(target=_run_ppg_stream, args=(cfg, stop_evt), daemon=True),
        # threading.Thread(target=_run_audio_bt_receiver, args=(cfg, stop_evt), daemon=True),
        threading.Thread(target=_run_depth_ordered_spatialiser, args=(cfg, stop_evt), daemon=True),
    ]

    for t in threads:
        t.start()

    print("[PI] Running (config-driven): RS->UDP + PPG->TCP + Audio<-UDP->BT")
    try:
        while not stop_evt.is_set():
            time.sleep(0.5)
    finally:
        print("[PI] Stop requested. Exiting.")


if __name__ == "__main__":
    main()

