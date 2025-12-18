#!/usr/bin/env python3
import socket
import struct
import pickle

import numpy as np
import cv2


def recv_exact(sock, n: int) -> bytes:
    """Receive exactly n bytes or raise EOFError."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise EOFError("Socket closed while receiving data")
        data += chunk
    return data


def handle_client(conn: socket.socket):
    print("Client connected.")
    try:
        while True:
            # Read 4-byte length header
            header = recv_exact(conn, 4)
            (length,) = struct.unpack("!I", header)

            payload = recv_exact(conn, length)
            packet = pickle.loads(payload)

            timestamp = packet.get("t")
            jpeg_bytes = packet.get("jpeg")
            imu = packet.get("imu", {})

            # Decode JPEG back to image
            jpg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)

            # Show frame
            if frame is not None:
                text_lines = []
                accel = imu.get("accel")
                gyro = imu.get("gyro")

                if accel is not None:
                    text_lines.append(
                        f"ACC: {accel[0]:+.3f}, {accel[1]:+.3f}, {accel[2]:+.3f}"
                    )
                if gyro is not None:
                    text_lines.append(
                        f"GYR: {gyro[0]:+.3f}, {gyro[1]:+.3f}, {gyro[2]:+.3f}"
                    )

                y = 20
                for line in text_lines:
                    cv2.putText(
                        frame,
                        line,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    y += 20

                cv2.imshow("RealSense D455 stream", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("ESC pressed, closing viewer.")
                    break
            else:
                print("Received frame could not be decoded.")

    except EOFError:
        print("Client disconnected.")
    except Exception as e:
        print(f"Receiver error: {e}")
    finally:
        conn.close()
        cv2.destroyAllWindows()


def main():
    host = "0.0.0.0"  # listen on all interfaces
    # host = "192.168.0.1"
    port = 50000

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)

    print(f"Listening on {host}:{port} ...")
    try:
        while True:
            conn, addr = server_sock.accept()
            print(f"Incoming connection from {addr}")
            handle_client(conn)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: shutting down server.")
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()

