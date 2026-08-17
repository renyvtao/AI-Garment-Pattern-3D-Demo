#!/usr/bin/env python3
"""Stream public official Google Drive archives directly to a remote inbox.

The SSH password must be supplied in the CG_REMOTE_PASS environment variable.
No credential is written to disk. Partial uploads use a `.part` suffix and are
resumed when the HTTP server accepts byte-range requests.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Callable

import paramiko
import requests


@dataclass(frozen=True)
class Asset:
    file_id: str
    remote_name: str
    expected_size: int


ASSETS = (
    Asset(
        file_id="1QXezA3J6uXqWHGATmcw3jaYxRXY2Ctte",
        remote_name="chatgarment-cg-assets.zip",
        expected_size=24_493_770,
    ),
    Asset(
        file_id="1NfxAeaC2va8TWMjiO_gbAcVPnZ8BYFPD",
        remote_name="contourcraft-data.zip",
        expected_size=883_651_710,
    ),
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def remote_size(sftp: paramiko.SFTPClient, path: str) -> int:
    try:
        return int(sftp.stat(path).st_size)
    except FileNotFoundError:
        return 0


def transfer_asset(
    asset: Asset,
    remote_inbox: str,
    connect: Callable[[], paramiko.SSHClient],
    max_attempts: int,
    segment_bytes: int,
) -> None:
    target = f"{remote_inbox.rstrip('/')}/{asset.remote_name}"
    partial = f"{target}.part"
    url = (
        "https://drive.usercontent.google.com/download"
        f"?id={asset.file_id}&export=download&confirm=t"
    )

    for attempt in range(1, max_attempts + 1):
        client: paramiko.SSHClient | None = None
        try:
            client = connect()
            with client.open_sftp() as sftp, requests.Session() as session:
                completed_size = remote_size(sftp, target)
                if completed_size:
                    if completed_size != asset.expected_size:
                        raise RuntimeError(
                            f"existing target has wrong size: {target} "
                            f"({completed_size} != {asset.expected_size})"
                        )
                    log(f"skip verified existing {target}")
                    return

                offset = remote_size(sftp, partial)
                if offset > asset.expected_size:
                    raise RuntimeError(
                        f"partial target is larger than expected: {partial}"
                    )
                range_end = min(
                    offset + segment_bytes - 1,
                    asset.expected_size - 1,
                )
                headers = {"Range": f"bytes={offset}-{range_end}"}
                response = session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=(30, 120),
                    allow_redirects=True,
                )
                response.raise_for_status()

                if response.status_code != 206:
                    raise RuntimeError(
                        f"HTTP server did not accept byte range "
                        f"{headers['Range']}: status={response.status_code}"
                    )

                total = asset.expected_size
                mode = "ab" if offset else "wb"
                written = offset
                next_report = offset
                report_step = (
                    max(total // 20, 8 * 1024 * 1024)
                    if total
                    else 32 * 1024 * 1024
                )

                log(
                    f"transfer {asset.remote_name}: attempt={attempt}, "
                    f"range={headers['Range']}, expected_total={total}"
                )
                with sftp.open(partial, mode, bufsize=0) as remote_file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        remote_file.write(chunk)
                        remote_file.flush()
                        written += len(chunk)
                        if written >= next_report:
                            percent = (written / total * 100) if total else 0.0
                            suffix = (
                                f" / {total / 1024**2:.1f} MiB ({percent:.1f}%)"
                                if total
                                else ""
                            )
                            log(
                                f"{asset.remote_name}: "
                                f"{written / 1024**2:.1f} MiB{suffix}"
                            )
                            next_report = written + report_step

                if written > total:
                    raise RuntimeError(
                        f"size mismatch for {asset.remote_name}: "
                        f"wrote {written}, maximum {total}"
                    )
                if written == total:
                    sftp.rename(partial, target)
                    log(f"completed {target} ({written} bytes)")
                    return
                log(
                    f"segment completed for {asset.remote_name}: "
                    f"{written}/{total} bytes"
                )
        except Exception as exc:
            log(
                f"{asset.remote_name}: attempt {attempt}/{max_attempts} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt == max_attempts:
                raise
            time.sleep(3)
        finally:
            if client is not None:
                client.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--remote-inbox", required=True)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--segment-mib", type=int, default=16)
    args = parser.parse_args()

    password = os.environ.get("CG_REMOTE_PASS")
    if not password:
        raise RuntimeError("CG_REMOTE_PASS is not set")

    def connect() -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=args.host,
            port=args.port,
            username=args.user,
            password=password,
            timeout=30,
            banner_timeout=30,
            auth_timeout=30,
        )
        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(10)
        return client

    client = connect()
    try:
        _, stdout, stderr = client.exec_command(
            f"mkdir -p {args.remote_inbox}", timeout=30
        )
        if stdout.channel.recv_exit_status():
            raise RuntimeError(stderr.read().decode(errors="replace"))
    finally:
        client.close()

    for asset in ASSETS:
        transfer_asset(
            asset,
            args.remote_inbox,
            connect=connect,
            max_attempts=args.max_attempts,
            segment_bytes=args.segment_mib * 1024 * 1024,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
