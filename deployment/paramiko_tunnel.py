#!/usr/bin/env python3
"""Open a local TCP forward using the deployment SSH profile.

The SSH password is read from ``AI_GARMENT_SSH_PASSWORD`` and is never written.
This helper is intended for automated health/UI checks; the interactive
Windows launcher remains the normal user-facing access method.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import socketserver
from pathlib import Path

import paramiko


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("server_connection.local.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    password = os.environ.get("AI_GARMENT_SSH_PASSWORD")
    if not password:
        raise SystemExit("AI_GARMENT_SSH_PASSWORD is required")

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=config["sshHost"],
        port=int(config["sshPort"]),
        username=config.get("sshUser", "root"),
        password=password,
        timeout=20,
    )
    transport = client.get_transport()
    if transport is None:
        raise RuntimeError("SSH transport was not established")

    remote_host = "127.0.0.1"
    remote_port = int(config["remotePort"])

    class ForwardHandler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            channel = transport.open_channel(
                "direct-tcpip",
                (remote_host, remote_port),
                self.request.getpeername(),
            )
            if channel is None:
                return
            try:
                while True:
                    readable, _, _ = select.select([self.request, channel], [], [])
                    if self.request in readable:
                        data = self.request.recv(65536)
                        if not data:
                            break
                        channel.sendall(data)
                    if channel in readable:
                        data = channel.recv(65536)
                        if not data:
                            break
                        self.request.sendall(data)
            finally:
                channel.close()

    class ForwardServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    try:
        with ForwardServer(("127.0.0.1", int(config["localPort"])), ForwardHandler) as server:
            print(f"forwarding http://127.0.0.1:{config['localPort']}", flush=True)
            server.serve_forever()
    finally:
        client.close()


if __name__ == "__main__":
    main()
