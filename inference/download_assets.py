#!/usr/bin/env python3
"""Download the versioned CORAL checkpoint and CCD files; verify SHA-256."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def matches(path, meta):
    if not path.is_file() or path.stat().st_size != meta["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest() == meta["sha256"]


def download(target, meta):
    if target.exists():
        if matches(target, meta):
            print(f"OK {target.name} (already present)", flush=True)
            return
        raise ValueError(f"Existing file does not match: {target}; move it before retrying")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(3):
        try:
            print(f"Downloading {target.name} ({meta['bytes'] / 1e6:.1f} MB)", flush=True)
            request = urllib.request.Request(meta["url"], headers={"User-Agent": "CORAL-inference/0.1"})
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                while block := response.read(8 << 20):
                    handle.write(block)
                    digest.update(block)
                    size += len(block)
                    if size > meta["bytes"]:
                        raise ValueError(f"Unexpected download size for {target.name}")
            if size != meta["bytes"] or digest.hexdigest() != meta["sha256"]:
                raise ValueError(f"Download checksum mismatch: {target.name}")
            os.replace(partial, target)
            print(f"OK {target.name} (SHA-256 verified)", flush=True)
            return
        except (OSError, ValueError) as error:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise RuntimeError(f"Could not download {target.name}: {error}") from error
            print(f"Retrying {target.name}: {error}", flush=True)
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=ROOT, help="Package root to receive checkpoints/ and data/")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "assets.json").read_text())
    try:
        for name, meta in manifest.items():
            download(args.dest.resolve() / name, meta)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
