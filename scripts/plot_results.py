"""Render publication-style figures from the saved experiment report and training logs."""

import argparse
import ast
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def training_points(path):
    points = []
    for line in Path(path).read_text(errors="replace").splitlines():
        match = re.search(r"Epoch:\[(\d+)/\d+\]\((\d+)/(\d+)\).*metrics: (\{.*\})", line)
        if match:
            epoch, step, total = map(int, match.group(1, 2, 3))
            metrics = ast.literal_eval(match.group(4))
            points.append({"step": (epoch - 1) * total + step, **metrics})
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/experiments.json")
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text())
    rows = {row["id"]: row for row in report["results"]}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9), layout="constrained")
    colors = ["#3075a6", "#de8c37", "#5c9b70"]
    for seed, color in zip((42, 43, 44), colors):
        values = [rows[f"sft_base_seed{seed}"]["response_ppl"], rows[f"sft_neftune_seed{seed}"]["response_ppl"]]
        axes[0].plot([0, 1], values, "o-", color=color, label=f"Seed {seed}", alpha=.85)
    axes[0].set_xticks([0, 1], ["SFT", "SFT + NEFTune"])
    axes[0].set_ylabel("Held-out response PPL (lower is better)")
    axes[0].set_title("SFT repeatability across 3 seeds")
    axes[0].legend(frameon=False)
    names = ["dpo_sum", "dpo_mean", "dpo_sum_nll", "dpo_mean_nll", "dpo_mean_nll_filtered"]
    labels = ["Sum", "Mean", "Sum\n+ NLL", "Mean\n+ NLL", "Mean + NLL\nfiltered"]
    values = [rows[name]["response_ppl"] for name in names]
    bars = axes[1].bar(range(len(names)), values, color=["#b9c8d2", "#b9c8d2", "#547e9a", "#547e9a", "#4a8d70"])
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(names)), labels)
    axes[1].axhline(rows["sft_neftune_seed42"]["response_ppl"], color="#bb5a51", linestyle="--", label="SFT reference")
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width()/2, value*1.05, f"{value:.2g}", ha="center", fontsize=8)
    axes[1].set_title("DPO factorial + data-quality control")
    axes[1].set_ylabel("Held-out response PPL (log scale)")
    axes[1].legend(frameon=False, fontsize=8)
    logs = []
    for name, label, color in [("dpo_mean", "Mean DPO", "#bb5a51"),
                                ("dpo_mean_nll", "Mean DPO + NLL", "#547e9a"),
                                ("dpo_mean_nll_filtered", "Mean DPO + NLL, filtered", "#4a8d70")]:
        points = training_points(f"out/experiments/{name}.log")
        logs.extend({"run": name, **point} for point in points)
        axes[2].plot([point["step"] for point in points], [point["chosen_nll"] for point in points],
                     "o-", label=label, color=color, markersize=3)
    axes[2].set_title("Chosen-response likelihood during training")
    axes[2].set_xlabel("Optimizer step")
    axes[2].set_ylabel("Logged minibatch chosen NLL (nats/token)")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=.15, axis="y")
    Path("images").mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(f"images/medical_experiments.{extension}")
    svg_path = Path("images/medical_experiments.svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)
    if logs:
        with Path("results/training_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=logs[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(logs)
    print("Figures -> images/medical_experiments.png and .svg")


if __name__ == "__main__":
    main()
