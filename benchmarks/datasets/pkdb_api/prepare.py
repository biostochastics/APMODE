#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""
PK-DB open-corpus fetcher.

Enumerates all studies at https://pk-db.com/api/v1/studies/, filters to
`licence:open`, and downloads:
  1) index.json           — study-level metadata for all 88 open studies
  2) timecourses/<sid>.json — per-study timecourse data where available

License:
  Code: MIT (github.com/matthiaskoenig/pkdb, verified 2026-07-08)
  Data: per-study "licence" field. This script fetches only licence:open studies.

Reference:
  Grzegorzewski J, Brandhorst J, Green K, Eleftheriadou D, Duport Y, Barthorscht F,
  Köhnlein A, Mosig A, König M. (2020) PK-DB: pharmacokinetics database for
  individualized and stratified computational modeling.
  Nucleic Acids Research 49(D1):D1358-D1364. doi:10.1093/nar/gkaa990
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

API = "https://pk-db.com/api/v1"


def fetch(url: str) -> dict:
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "timecourses").mkdir(exist_ok=True)

    open_studies = []
    page = 1
    while True:
        d = fetch(f"{API}/studies/?format=json&page={page}&page_size=100")
        rows = d["data"]["data"]
        if not rows:
            break
        for s in rows:
            if s.get("licence") == "open":
                open_studies.append(s)
        last_page = d.get("last_page") or d["data"].get("last_page")
        if last_page is None or page >= last_page:
            break
        page += 1

    print(f"Retrieved {len(open_studies)} open-license studies")

    with (out / "index.json").open("w") as f:
        json.dump(open_studies, f, indent=2)

    # Compact metadata table
    with (out / "index_summary.csv").open("w") as f:
        f.write("sid,name,pmid,doi,timecourse_count,individual_count,intervention_count\n")
        for s in open_studies:
            ref = s.get("reference") or {}
            f.write(",".join([
                s.get("sid", ""),
                (s.get("name") or "").replace(",", " "),
                str(ref.get("pmid") or ""),
                (ref.get("doi") or "").replace(",", " "),
                str(s.get("timecourse_count") or 0),
                str(s.get("individual_count") or 0),
                str(s.get("intervention_count") or 0),
            ]) + "\n")
    print(f"index.json + index_summary.csv written")

    # Fetch full study payload (contains dataset/groupset/individualset/
    # interventionset/outputset — all timecourse data lives in these fields).
    # Endpoint is /studies/{sid}/, not /timecourses/ (which doesn't exist).
    with_tc = [s for s in open_studies if (s.get("timecourse_count") or 0) > 0]
    print(f"Fetching full study payloads for {len(with_tc)} timecourse-bearing studies...")
    (out / "studies").mkdir(exist_ok=True)
    ok, fail = 0, 0
    for i, s in enumerate(with_tc, 1):
        sid = s.get("sid")
        try:
            d = fetch(f"{API}/studies/{sid}/?format=json")
            with (out / "studies" / f"{sid}.json").open("w") as f:
                json.dump(d, f)
            ok += 1
        except Exception as e:
            print(f"  [{i:3d}/{len(with_tc)}] {sid}: FAILED {e}")
            fail += 1
        time.sleep(0.05)
    print(f"Retrieved {ok}/{len(with_tc)} full study payloads ({fail} failed)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/datasets/pkdb_api")
