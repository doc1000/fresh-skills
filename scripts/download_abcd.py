"""Download public ABCD files into data/raw/."""

from __future__ import annotations

import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from playbook.config import ABCD_GZ, ABCD_JSON, DATA_RAW  # noqa: E402

BASE = "https://raw.githubusercontent.com/asappresearch/abcd/master/data"
FILES = ("ontology.json", "guidelines.json", "kb.json", "abcd_v1.1.json.gz")


def _download(name: str) -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    dest = DATA_RAW / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip existing {dest} ({dest.stat().st_size} bytes)")
        return dest
    url = f"{BASE}/{name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "fresh-skills-abcd-download"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(dest)
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")
    return dest


def _decompress_gz(path: Path) -> Path:
    out = ABCD_JSON if path == ABCD_GZ else path.with_suffix("")
    if out.exists() and out.stat().st_size > 0:
        print(f"skip existing {out} ({out.stat().st_size} bytes)")
        return out
    print(f"decompressing {path}")
    with gzip.open(path, "rb") as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return out


def main() -> None:
    for name in FILES:
        path = _download(name)
        if name.endswith(".gz"):
            _decompress_gz(path)
    print("ABCD download complete")


if __name__ == "__main__":
    main()
