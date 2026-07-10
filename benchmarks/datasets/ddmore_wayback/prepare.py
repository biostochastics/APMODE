#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""
DDMoRe Model Repository — Wayback Machine fallback fetcher.

The live repository (repository.ddmore.foundation) returned 503 across all
URLs on 2026-07-08. The Internet Archive has a snapshot from 2025-03-27
that is verified reachable. This script pulls the model index + a few
canonical model pages from the Wayback snapshot.

Reference:
  Harnisch L, Matthews I, Chard J, Karlsson MO. (2013)
  Drug and Disease Model Resources: A Consortium to Create Standards and
  Tools to Enhance Model-Based Drug Development.
  CPT: Pharmacometrics & Systems Pharmacology 2:e34.
  doi:10.1038/psp.2013.10

Snapshot URL:
  http://web.archive.org/web/20250327075132/http://repository.ddmore.foundation/models
"""

import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://web.archive.org/web/20250327075132/http://repository.ddmore.foundation"
CANONICAL_IDS = [
    ("DDMODEL00000003", "Hamren 2008 diabetes tesaglitazar (KPD turnover)"),
    ("DDMODEL00000103", "Trefz 2015 Kuvan/PKU turnover-KPD"),
    ("DDMODEL00000130", "Karaiskos 2015 CMS/colistin PK"),
    ("DDMODEL00000238", "Germovsek 2017 gentamicin neonatal IOV"),
    ("DDMODEL00000243", "TTE example"),
    ("DDMODEL00000247", "IRT example"),
    ("DDMODEL00000248", "Preterm-neonatal morphine + phenobarbital"),
]


def fetch(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "APMODE-DDMoRe-Wayback/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            dest.write_bytes(r.read())
        return True
    except Exception as e:
        print(f"  FAILED {url}: {e}")
        return False


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Model index snapshot
    print("Fetching model index snapshot...")
    fetch(f"{BASE}/models", out / "models_index.html")

    # Individual model pages
    print("Fetching canonical model pages...")
    for mid, desc in CANONICAL_IDS:
        print(f"  {mid}: {desc}")
        fetch(f"{BASE}/model/{mid}", out / f"{mid}.html")
        time.sleep(0.5)  # be polite to archive.org

    # Write a README summarising what was retrieved
    (out / "README.md").write_text(
        """# DDMoRe Model Repository — Wayback Snapshot

Live repository (repository.ddmore.foundation) returned 503 across all URLs on
2026-07-08. This directory contains a Wayback Machine snapshot from **2025-03-27**.

## Canonical model IDs
"""
        + "\n".join(f"- `{mid}` — {desc}" for mid, desc in CANONICAL_IDS)
        + f"""

## Source
- Snapshot base: {BASE}
- Live URL (503): http://repository.ddmore.foundation/models
- Foundation home (200): https://www.ddmore.foundation/

## Reference
Harnisch L, Matthews I, Chard J, Karlsson MO. (2013)
Drug and Disease Model Resources.
CPT: Pharmacometrics & Systems Pharmacology 2:e34.
[doi:10.1038/psp.2013.10](https://doi.org/10.1038/psp.2013.10)
"""
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/datasets/ddmore_wayback")
