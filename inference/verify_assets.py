#!/usr/bin/env python3
"""Verify the exact local checkpoint/CCD files required by this bundle."""
import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
expected=json.loads((r/'assets.json').read_text())
for name,meta in expected.items():
 p=r/name
 if not p.is_file():raise SystemExit(f'Missing asset: {p}')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 if h.hexdigest()!=meta['sha256'] or p.stat().st_size!=meta['bytes']:raise SystemExit(f'Asset mismatch: {p}')
 print(f'OK {name}')
