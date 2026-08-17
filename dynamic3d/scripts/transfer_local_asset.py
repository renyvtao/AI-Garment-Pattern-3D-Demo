#!/usr/bin/env python3
"""Upload one local asset to a remote host with resumable SFTP segments."""

from __future__ import annotations

import argparse
import hashlib
import os
import posixpath
import time
from pathlib import Path

import paramiko


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def remote_size(sftp: paramiko.SFTPClient, path: str) -> int:
    try:
        return int(sftp.stat(path).st_size)
    except FileNotFoundError:
        return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-file", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", default="root")
    parser.add_argument("--remote-file", required=True)
    parser.add_argument("--segment-mib", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=200)
    args = parser.parse_args()

    password = os.environ.get("CG_REMOTE_PASS")
    if not password:
        raise RuntimeError("CG_REMOTE_PASS is not set")

    local_file = args.local_file.resolve()
    local_size = local_file.stat().st_size
    local_sha256 = sha256_file(local_file)
    partial = f"{args.remote_file}.part"
    segment_bytes = args.segment_mib * 1024 * 1024

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

    for attempt in range(1, args.max_attempts + 1):
        client: paramiko.SSHClient | None = None
        try:
            client = connect()
            parent = posixpath.dirname(args.remote_file)
            _, stdout, stderr = client.exec_command(
                f"mkdir -p {parent}",
                timeout=30,
            )
            if stdout.channel.recv_exit_status():
                raise RuntimeError(stderr.read().decode(errors="replace"))

            with client.open_sftp() as sftp:
                completed_size = remote_size(sftp, args.remote_file)
                if completed_size:
                    if completed_size != local_size:
                        raise RuntimeError(
                            f"existing target size mismatch: "
                            f"{completed_size} != {local_size}"
                        )
                    log(f"target already has expected size: {args.remote_file}")
                    break

                offset = remote_size(sftp, partial)
                if offset > local_size:
                    raise RuntimeError(
                        f"partial target is larger than local file: {offset}"
                    )
                remaining = min(segment_bytes, local_size - offset)
                if remaining == 0:
                    sftp.rename(partial, args.remote_file)
                    break

                log(
                    f"attempt={attempt} offset={offset} "
                    f"segment={remaining} total={local_size}"
                )
                with (
                    local_file.open("rb") as source,
                    sftp.open(partial, "ab" if offset else "wb", bufsize=0) as target,
                ):
                    source.seek(offset)
                    written = 0
                    while written < remaining:
                        chunk = source.read(min(1024 * 1024, remaining - written))
                        if not chunk:
                            raise RuntimeError("unexpected end of local file")
                        target.write(chunk)
                        target.flush()
                        written += len(chunk)

                new_size = remote_size(sftp, partial)
                if new_size != offset + remaining:
                    raise RuntimeError(
                        f"remote segment size mismatch: "
                        f"{new_size} != {offset + remaining}"
                    )
                percent = new_size / local_size * 100
                log(
                    f"segment complete: {new_size}/{local_size} "
                    f"({percent:.1f}%)"
                )
                if new_size == local_size:
                    sftp.rename(partial, args.remote_file)
                    break
        except Exception as exc:
            log(
                f"attempt {attempt}/{args.max_attempts} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt == args.max_attempts:
                raise
            time.sleep(3)
        finally:
            if client is not None:
                client.close()
    else:
        raise RuntimeError("upload attempts exhausted")

    client = connect()
    try:
        command = f"sha256sum {args.remote_file}"
        _, stdout, stderr = client.exec_command(command, timeout=300)
        code = stdout.channel.recv_exit_status()
        output = stdout.read().decode(errors="replace").strip()
        if code:
            raise RuntimeError(stderr.read().decode(errors="replace"))
        remote_sha256 = output.split()[0]
    finally:
        client.close()

    log(f"local_sha256={local_sha256}")
    log(f"remote_sha256={remote_sha256}")
    if remote_sha256 != local_sha256:
        raise RuntimeError("SHA-256 mismatch after upload")
    log(f"completed and verified: {args.remote_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
