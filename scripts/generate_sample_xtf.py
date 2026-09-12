"""
Utility script to generate sample binary Triton XTF side-scan sonar files for testing and demo.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.sonar_raw_ingestion import generate_synthetic_xtf


def main():
    out_dir = ROOT_DIR / "samples" / "raw_xtf"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "survey_track_alpha.xtf"

    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(f"Generating synthetic Triton XTF file: {out_file} ...")
    generated = generate_synthetic_xtf(
        output_path=out_file,
        num_pings=160,
        samples_per_channel=400,
        base_lat=13.0827,
        base_lon=80.2707,
        inject_target=True
    )
    print(f"[OK] Synthetic XTF file successfully created at: {generated}")
    print(f"  File size: {generated.stat().st_size} bytes")


if __name__ == "__main__":
    main()
