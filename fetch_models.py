"""Copy or download the 11 IFC-Bench models of the exercise into models/.

Usage:
    uv run fetch_models.py                     # local checkout if present, else Hugging Face
    uv run fetch_models.py --source hf         # always download from Hugging Face
    uv run fetch_models.py --local PATH        # copy from an IFC-Bench checkout at PATH

Every file is checked against its SHA-256, so a git LFS pointer or a truncated
download is rejected. Each model's licence is copied next to it.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DEFAULT_LOCAL = Path.home() / "code" / "tum" / "ifc-bench"
HF_REVISION = "9ba471a3e38b353a81a1dc9ecb4aa56d2ce0a7fa"
HF_BASE = f"https://huggingface.co/datasets/sylvainHellin/ifc-bench/resolve/{HF_REVISION}"

# model id -> (IFC-Bench project, IFC-Bench ifc_model, size in bytes, sha256)
MODELS: dict[str, tuple[str, str, int, str]] = {
    "4351": ("4351", "arc", 7288853, "96077863a6734f574fadc33c77199a9b82222fc4b3f4c0da6612eef9347c99a9"),
    "ac20": ("ac20", "arc", 10934237, "cfb2124497b25d9a72101075e84be0feb44ff669cb1bd3251be11efebeea945c"),
    "duplex_arc": ("duplex", "arc", 1656153, "707a032566dee8e969a44d95509be46d646440e2443624709f8d3ae2b45f4656"),
    "duplex_mep": ("duplex", "mep", 11129368, "285f85de4424416ba6bade121588dbe0ae278fb4fdb0f4b782051cfdc7c92972"),
    "fantasy_hotel_1": ("fantasy_hotel_1", "arc", 8455469, "cd976f3e9222a49ad072078ed27934f20dbc11c889271b1ee3c1ab4ce504a03e"),
    "fantasy_office_building_1": ("fantasy_office_building_1", "arc", 13427535, "2b0fcd8f8abc9000043ea575b22a9153f75219edc3437c157348cbfcc4358569"),
    "fantasy_office_building_2": ("fantasy_office_building_2", "arc", 19463969, "944844edfa7a02698b438f2733ccd76aa4a8d471ddd7b02866d77d1b4703bd46"),
    "fantasy_office_building_3": ("fantasy_office_building_3", "arc", 9655829, "bc92fbc42ecacf0c46246eb5c7e1c12f5a63982aba947aefac5fb46e93cc1e74"),
    "fantasy_residential_building_1": ("fantasy_residential_building_1", "arc", 19719618, "dd7000fc4d1d4f651044375e32d48f947ee3aaf6ce8efc634c0e65a4bed48a1c"),
    "fzk_house": ("fzk_house", "arc", 2570803, "70cc8ff245fc0894201d96496c031005a5cbd7a96b22d8a1b87c5a883fb77994"),
    "smiley_west": ("smiley_west", "arc", 6109213, "26734e67bdc0fd2ab30cb560bddc279a0a4f23eb4d28861509524e4bbe201c48"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid(path: Path, size: int, digest: str) -> bool:
    return path.is_file() and path.stat().st_size == size and sha256(path) == digest


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)


def fetch(source: str, local: Path) -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    use_local = source == "local" or (source == "auto" and (local / "projects").is_dir())
    failures = 0
    for model_id, (project, ifc_model, size, digest) in MODELS.items():
        dest = MODELS_DIR / f"{model_id}.ifc"
        lic = MODELS_DIR / f"{model_id}.license.txt"
        if is_valid(dest, size, digest) and lic.is_file():
            print(f"ok       {dest.name}")
            continue
        rel_ifc = f"projects/{project}/{ifc_model}.ifc"
        rel_lic = f"projects/{project}/license.txt"
        try:
            if use_local:
                shutil.copyfile(local / rel_ifc, dest)
                shutil.copyfile(local / rel_lic, lic)
                how = "copied"
            else:
                download(f"{HF_BASE}/{rel_ifc}", dest)
                download(f"{HF_BASE}/{rel_lic}", lic)
                how = "fetched"
        except OSError as exc:
            print(f"FAILED   {dest.name}: {exc}", file=sys.stderr)
            failures += 1
            continue
        if is_valid(dest, size, digest):
            print(f"{how:<8} {dest.name} ({size / 1e6:.1f} MB)")
        else:
            print(f"FAILED   {dest.name}: size or SHA-256 mismatch (git LFS pointer?)", file=sys.stderr)
            failures += 1
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["auto", "local", "hf"], default="auto")
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL, help="IFC-Bench checkout")
    args = parser.parse_args()
    source = "local" if args.source == "auto" and args.local != DEFAULT_LOCAL else args.source
    sys.exit(1 if fetch(source, args.local) else 0)


if __name__ == "__main__":
    main()
