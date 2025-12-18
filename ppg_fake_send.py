import socket, struct, time, math, argparse
PACK_FMT = "!dii"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--hz", type=float, default=60.0)
    args = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t0 = time.time()
    k = 0
    while True:
        ts = time.time()
        # simple fake waveform
        red = int(50000 + 2000 * math.sin(2*math.pi*1.2*(ts - t0)))
        ir  = int(52000 + 2500 * math.sin(2*math.pi*1.2*(ts - t0) + 0.4))
        pkt = struct.pack(PACK_FMT, ts, red, ir)
        s.sendto(pkt, (args.ip, args.port))
        k += 1
        time.sleep(1.0 / args.hz)

if __name__ == "__main__":
    main()

