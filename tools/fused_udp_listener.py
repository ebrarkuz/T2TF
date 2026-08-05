"""UDP 8888 fused mesajlarını doğrular ve okunabilir biçimde gösterir."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime.message_schema import decode_json_packet, validate_fused_message


def main() -> None:
    parser = argparse.ArgumentParser(description="Fused UDP JSON listener")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    print(f"Fused UDP dinleniyor: {args.host}:{args.port} (Ctrl+C ile durdur)")
    try:
        while True:
            payload, address = sock.recvfrom(65508)
            try:
                message = validate_fused_message(decode_json_packet(payload))
                print(address, json.dumps(message, ensure_ascii=False, indent=2))
            except ValueError as exc:
                print(address, f"REJECTED: {exc}")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


if __name__ == "__main__":
    main()
