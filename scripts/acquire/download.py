"""Download every raw source file once and record URL, size, SHA-256 and server date.

Appends one JSON line per file to _downloads/MANIFEST.jsonl (read later by write_licenses.py).
Usage: python download.py GROUP  (GROUP in amazon, google, openflights, enron, all)
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

RAW = Path(__file__).resolve().parents[1]
DL = RAW / "_downloads"
AMZ = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
GL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal"
OF_COMMIT = "7d1a611e070295dba776d6afb86e57d0d1aa1cef"
OF = f"https://raw.githubusercontent.com/jpatokal/openflights/{OF_COMMIT}/data"
EN_REV = "cfc06c758093d90993abce1a43668fb7357258a6"
EN = f"https://huggingface.co/datasets/corbt/enron-emails/resolve/{EN_REV}/data"

AMAZON_CATEGORIES = ["Grocery_and_Gourmet_Food", "Pet_Supplies", "Baby_Products", "Appliances", "All_Beauty"]
GOOGLE_STATES = ["District_of_Columbia", "Delaware", "Rhode_Island"]

GROUPS = {
    "amazon": [f"{AMZ}/review_categories/{c}.jsonl.gz" for c in AMAZON_CATEGORIES]
    + [f"{AMZ}/meta_categories/meta_{c}.jsonl.gz" for c in AMAZON_CATEGORIES],
    "google": [f"{GL}/review-{s}_10.json.gz" for s in GOOGLE_STATES]
    + [f"{GL}/meta-{s}.json.gz" for s in GOOGLE_STATES],
    "openflights": [f"{OF}/{n}.dat" for n in ("airports", "airlines", "routes")],
    "enron": [f"{EN}/train-0000{i}-of-00003.parquet" for i in range(3)],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(group: str, url: str) -> None:
    dest = DL / group / url.rsplit("/", 1)[1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["curl", "-sIL", "--max-time", "60", url], capture_output=True, text=True).stdout
    last_mod = [ln.split(":", 1)[1].strip() for ln in head.splitlines() if ln.lower().startswith("last-modified")]
    subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "-C", "-", "-o", str(dest), url], check=True)
    rec = {
        "group": group, "url": url, "file": dest.name, "bytes": dest.stat().st_size, "sha256": sha256(dest),
        "server_last_modified": last_mod[-1] if last_mod else None,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with (DL / "MANIFEST.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    groups = list(GROUPS) if sys.argv[1] == "all" else sys.argv[1:]
    for g in groups:
        for u in GROUPS[g]:
            fetch(g, u)
