"""
Experiment Comparator and Summary Report Generator.
Aggregates performance metrics across baseline and experimental runs into Markdown and JSON comparison tables.
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_comparison_table(exp_dirs: list, output_md_path: str) -> None:
    results = []

    for edir in exp_dirs:
        edir_path = Path(edir).resolve()
        summary_file = edir_path / "evaluation_summary.json"
        if not summary_file.exists():
            # Check inside runs or subdirs
            candidates = list(edir_path.glob("**/evaluation_summary.json"))
            if candidates:
                summary_file = candidates[0]
            else:
                continue

        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        exp_name = edir_path.name
        overall = data.get("overall", {})
        per_class = data.get("per_class", {})

        row = {
            "Experiment": exp_name,
            "Resolution": data.get("imgsz", 640),
            "Conf": data.get("conf_threshold", 0.25),
            "Precision (%)": f"{overall.get('precision', 0) * 100:.2f}%",
            "Recall (%)": f"{overall.get('recall', 0) * 100:.2f}%",
            "mAP@50 (%)": f"{overall.get('map50', 0) * 100:.2f}%",
            "mAP@50-95 (%)": f"{overall.get('map50_95', 0) * 100:.2f}%",
            "Latency (ms)": f"{overall.get('inference_speed_ms', 0):.2f}",
            "FPS": f"{overall.get('fps', 0):.1f}",
            "Shipwreck mAP50": f"{per_class.get('shipwreck', {}).get('map50', 0) * 100:.1f}%",
            "Airplane mAP50": f"{per_class.get('airplane', {}).get('map50', 0) * 100:.1f}%",
            "Mine mAP50": f"{per_class.get('mine', {}).get('map50', 0) * 100:.1f}%",
            "Human mAP50": f"{per_class.get('human', {}).get('map50', 0) * 100:.1f}%",
        }
        results.append(row)

    if not results:
        print("[Warning] No evaluation summaries found across specified experiment directories.")
        return

    # Build pure Markdown table string
    headers = list(results[0].keys())
    header_str = "| " + " | ".join(headers) + " |"
    sep_str = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows_str = []
    for r in results:
        rows_str.append("| " + " | ".join(str(r[h]) for h in headers) + " |")

    md_table = "\n".join([header_str, sep_str] + rows_str)

    print("\n" + "=" * 90)
    print(" 🏆 YOLO11 UNDERWATER DETECTION EXPERIMENTAL COMPARISON TABLE")
    print("=" * 90)
    print(md_table)
    print("=" * 90 + "\n")


    out_file = Path(output_md_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# YOLO11 Experimental Optimization Results\n\n")
        f.write(md_table)
        f.write("\n")

    print(f"✅ Comparison table saved to: {out_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO11 experimental runs.")
    parser.add_argument("--exp-dir", type=str, default="outputs/experiments", help="Parent directory containing experiment runs")
    parser.add_argument("--output", type=str, default="outputs/experiments/FINAL_COMPARISON_TABLE.md", help="Output Markdown path")
    args = parser.parse_args()

    parent_dir = Path(args.exp_dir).resolve()
    if not parent_dir.exists():
        print(f"[Error] Experiment directory not found: {parent_dir}")
        sys.exit(1)

    exp_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]
    generate_comparison_table(exp_dirs, args.output)


if __name__ == "__main__":
    main()
