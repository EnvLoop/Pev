# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2"]
# ///
"""Figures (PNG + SVG, plus vector PDF for the LaTeX paper) and the TEST tables of the tech report, computed only from JSON result files.

Inputs (paths relative to the repository root):
  docs/tech-report/data/{hidden,dev-rounds}.json   aggregate snapshots written by collect_results.py
  docs/reference-dev.json                          DEV reference comparison (6 predictors)
  docs/TEST_RESULTS.json                           optional; the TEST comparison of amendment 6 (schema below)

    uv run docs/tech-report/make_figures.py            # writes docs/tech-report/figures/*.{png,svg} and paper/figures/*.pdf
    uv run docs/tech-report/make_figures.py tables     # prints the TEST tables (Markdown) for report.md section 5.3

TEST_RESULTS.json (schema "muse-test-comparison/1"); every predictor and pair block may carry the halves
"all", "A" (gpt-6-astra-rendered users) and "B" (Claude-rendered users):
  {"schema", "test_sha256", "commitment_sha256", "adapter": "<predictor name>",
   "n": {"all"|"A"|"B": {"states", "questions", "scorable"}},
   "predictors": {name: {"all"|"A"|"B": {"macro_accuracy", "macro_brier", "macro_ece", "automation_coverage"?,
                                         "families": {family: {"accuracy", "n_scored"}},
                                         "safety_fn_rate": {family: rate}}}},
   "paired_adapter_vs": {"adapter - <name>": {"all"|"A"|"B": {"delta", "ci95", "p_one_sided", "p_holm", "clusters",
                                                             "families": {family: {"delta", "ci95"}}}}}}
A predictor block without halves (the reference-dev.json shape) is read as "all".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "figures"
PAPER_FIG = HERE / "paper" / "figures"  # vector PDFs included by paper/main.tex
FAMILIES = ["apply_memory", "forgotten_violation", "needs_approval", "share_ok", "route", "notify_level", "pick_option"]
CHANCE = {"apply_memory": 0.5, "forgotten_violation": 0.5, "needs_approval": 0.5, "share_ok": 0.5, "route": 1 / 6,
          "notify_level": 0.25, "pick_option": 0.279}  # pick_option: DEV chance (shortcut-baselines, DEV_RECEIPT)
# Fixed identity colors (validated categorical order; the adapter always slot 1, B0 slot 2).
COLORS = {"adapter": "#2a78d6", "b0": "#eb6834", "kev": "#1baf7a", "astra": "#eda100", "jev": "#e87ba4",
          "4b": "#008300", "other": "#4a3aa7"}
# Display names in figure text: the release names instead of the run / predictor names stored in the JSON inputs.
ADAPTER_LABEL, BASE_LABEL = "Pev-27B", "Qwen3.8-27B (base)"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"svg.hashsalt": "muse-tech-report", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def load(rel: str) -> dict | None:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def color_of(name: str) -> str:
    low = name.lower()
    for key, token in (("adapter", "adapter"), ("b0", "b0"), ("3.8-27b", "b0"), ("kev", "kev"), ("astra", "astra"),
                       ("jev", "jev"), ("4b", "4b")):
        if key in low:
            return COLORS[token]
    return COLORS["other"]


def label_of(name: str) -> str:
    low = name.lower()
    if "adapter" in low:
        return ADAPTER_LABEL
    if low.startswith("b0") or "3.8-27b" in low:
        return BASE_LABEL
    return name


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", metadata={"Date": None, "Creator": None})
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG / f"{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None})
    plt.close(fig)
    print(f"figures/{name}.png, .svg; paper/figures/{name}.pdf")


def pts(x: float) -> float:
    return 100 * x


def fig_hidden_accuracy(hidden: dict) -> None:
    agg = hidden["aggregate"]
    rows = FAMILIES + ["macro"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, fam in enumerate(rows):
        block = agg["macro"] if fam == "macro" else agg["families"][fam]
        b, c = block["base"]["accuracy"], block["cand"]["accuracy"]
        ax.plot([b, c], [i, i], color=GRID, lw=2, solid_capstyle="round", zorder=1)
        ax.scatter([b], [i], s=42, color=COLORS["b0"], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=BASE_LABEL if i == 0 else None)
        ax.scatter([c], [i], s=42, color=COLORS["adapter"], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=ADAPTER_LABEL if i == 0 else None)
        if fam in CHANCE:
            ax.plot([CHANCE[fam]] * 2, [i - 0.3, i + 0.3], color=MUTED, lw=1, zorder=2,
                    label="chance" if i == 0 else None)
        ax.annotate(f"{pts(c - b):+.1f}", (max(b, c), i), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=8)
    ax.set_yticks(range(len(rows)), [r.replace("_", " ") for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.1, 1.08)
    ax.set_xlabel("Accuracy on HIDDEN (2,106 scorable questions); label = change in points")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    save(fig, "fig1_hidden_accuracy")


def forest(ax, labels: list[str], series: list[tuple[str, str, list[tuple[float, float, float]]]]) -> None:
    """series: (legend, color, [(delta, lo, hi)] per label); horizontal CIs, offset per series."""
    k = len(series)
    for j, (legend, color, values) in enumerate(series):
        off = (j - (k - 1) / 2) * 0.26
        for i, (d, lo, hi) in enumerate(values):
            ax.plot([pts(lo), pts(hi)], [i + off] * 2, color=color, lw=2, solid_capstyle="round")
            ax.scatter([pts(d)], [i + off], s=36, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=legend if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(labels)), [x.replace("_", " ") for x in labels])
    ax.invert_yaxis()


def fig_hidden_deltas(hidden: dict, rounds: dict) -> None:
    agg, dev = hidden["aggregate"], rounds["r2-mix"]
    rows = FAMILIES + ["macro"]

    def triples(score: dict) -> list[tuple[float, float, float]]:
        out = [(score["families"][f]["accuracy_delta"], *score["families"][f]["accuracy_delta_ci95"]) for f in FAMILIES]
        return out + [(score["bootstrap"]["delta"], *score["bootstrap"]["ci95"])]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    forest(ax, rows, [("DEV (hill-climb test set)", COLORS["other"], triples(dev)),
                      ("HIDDEN (one-shot gate)", COLORS["adapter"], triples(agg))])
    ax.set_xlabel(f"{ADAPTER_LABEL} − base accuracy (points), paired state-cluster bootstrap 95% CI")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig2_dev_hidden_deltas")


def fig_rounds(rounds: dict) -> None:
    names = ["r1-seedA", "r1-seedB", "r2-mix", "r2-7k"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for i, name in enumerate(names):
        r = rounds[name]
        boot = r["bootstrap"]
        color = COLORS["adapter"] if r["decision"] == "advance_to_hidden" else MUTED
        a.plot([pts(boot["ci95"][0]), pts(boot["ci95"][1])], [i, i], color=color, lw=2, solid_capstyle="round")
        a.scatter([pts(boot["delta"])], [i], s=36, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        g = pts(r["guard"]["delta"])
        b.barh(i, g, height=0.45, color=color)
        b.annotate(f"{g:+.2f}", (g, i), xytext=(4 if g >= 0 else -4, 0), textcoords="offset points",
                   ha="left" if g >= 0 else "right", va="center", fontsize=8, color=INK)
    a.axvline(5, color=MUTED, lw=1)
    shown = {"r2-mix": f"r2-mix = {ADAPTER_LABEL}"}  # the round whose adapter is released
    a.set_yticks(range(len(names)),
                 [f"{shown.get(n, n)}\n({rounds[n]['decision'].replace('_', ' ')})" for n in names])
    a.invert_yaxis()
    a.set_xlim(0, 18)
    a.set_xlabel("DEV macro Δ vs base (points, 95% CI); line = +5 gate")
    b.axvline(0, color=MUTED, lw=1)
    b.set_xlim(-1.2, 2.2)
    b.set_xlabel("decision-v7 guard Δ vs base (points)")
    save(fig, "fig3_dev_rounds")


def fig_automation(hidden: dict) -> None:
    agg = hidden["aggregate"]["families"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for key, label, color in (("base", BASE_LABEL, COLORS["b0"]), ("cand", ADAPTER_LABEL, COLORS["adapter"])):
        xs = [agg[f][key]["coverage"] for f in FAMILIES]
        ys = [agg[f][key]["realized_error"] for f in FAMILIES]
        ax.scatter(xs, ys, s=40, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=label)
        for f, x, y in zip(FAMILIES, xs, ys):
            if f == "pick_option" or (key == "base" and f == "notify_level"):
                ax.annotate(f.replace("_", " "), (x, y), xytext=(5, 3), textcoords="offset points", fontsize=7,
                            color=INK)
    ax.axhline(0.05, color=MUTED, lw=1)
    ax.annotate("5% error budget", (0.0, 0.05), xytext=(2, 3), textcoords="offset points", fontsize=7, color=MUTED)
    ax.set_xlabel("Automation coverage on HIDDEN at the VAL-fitted threshold")
    ax.set_ylabel("Realized error on accepted questions")
    ax.set_xlim(-0.03, 1.05)
    ax.legend(loc="center right", fontsize=8)
    save(fig, "fig4_hidden_automation")


def fig_dev_references(ref: dict) -> None:
    preds = sorted(ref["predictors"], key=lambda n: ref["predictors"][n]["macro_accuracy"])
    cols = FAMILIES + ["macro"]
    grid = [[ref["predictors"][p]["macro_accuracy"] if c == "macro" else ref["predictors"][p]["families"][c]["accuracy"]
             for c in cols] for p in preds]
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.imshow(grid, cmap="Blues", vmin=0.2, vmax=1.15, aspect="auto")
    ax.grid(False)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5, color="#ffffff" if v > 0.85 else INK)
    ax.set_xticks(range(len(cols)), [c.replace("_", "\n") for c in cols], fontsize=7.5)
    ax.set_yticks(range(len(preds)), [label_of(p) for p in preds], fontsize=8)
    ax.set_title("DEV accuracy (1,409 scorable questions; pre-pseudonymization text)", fontsize=9, color=INK)
    save(fig, "fig5_dev_references")


def half(block: dict, key: str) -> dict | None:
    if key in block:
        return block[key]
    return block if key == "all" and "macro_accuracy" in block else None


def fig_test(test: dict) -> None:
    order = list(COLORS.values())
    preds = sorted(test["predictors"], key=lambda name: order.index(color_of(name)))
    halves = [h for h in ("all", "A", "B") if any(half(test["predictors"][p], h) for p in preds)]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    width = 0.8 / max(len(preds), 1)
    for j, p in enumerate(preds):
        vals = [(half(test["predictors"][p], h) or {}).get("macro_accuracy") for h in halves]
        xs = [i + (j - (len(preds) - 1) / 2) * width for i in range(len(halves))]
        ax.bar([x for x, v in zip(xs, vals) if v is not None], [v for v in vals if v is not None],
               width=width * 0.9, color=color_of(p), label=label_of(p))
    ax.set_xticks(range(len(halves)), [{"all": "TEST (all)", "A": "A half\n(gpt-6-astra-rendered)",
                                         "B": "B half\n(Claude Opus 5.5-rendered)"}[h] for h in halves])
    ax.set_ylabel("Family-macro accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig6_test_macro")
    pairs = test.get("paired_adapter_vs") or {}
    pairs = dict(sorted(pairs.items(), key=lambda kv: (half(kv[1], "all") or {}).get("delta", 0)))
    if pairs:
        fig, ax = plt.subplots(figsize=(6.4, 0.5 + 0.5 * len(pairs)))
        series = []
        for h, color in (("all", COLORS["adapter"]), ("A", COLORS["astra"]), ("B", COLORS["other"])):
            vals = [half(pairs[k], h) for k in pairs]
            if all(vals):
                series.append((f"TEST {h}", color, [(v["delta"], *v["ci95"]) for v in vals]))
        forest(ax, ["vs " + label_of(k.removeprefix("adapter - ")) for k in pairs], series)
        ax.set_xlabel(f"{ADAPTER_LABEL} − model, family-macro accuracy (points, 95% CI)")
        ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
        save(fig, "fig7_test_deltas")


def fmt(x, digits=3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def ci(v: dict) -> str:
    return f"{pts(v['delta']):+.1f} [{pts(v['ci95'][0]):+.1f}, {pts(v['ci95'][1]):+.1f}]"


def tables(test: dict) -> None:
    """The four TEST tables of report.md section 5.3, rows ordered by overall macro accuracy (ascending)."""
    preds = sorted(test["predictors"], key=lambda p: test["predictors"][p]["all"]["macro_accuracy"])
    sens = test.get("sensitivity_excl_pick_option") or {}
    print("<!-- source: docs/TEST_RESULTS.json (predictors.*.all) -->")
    print("| Model | Macro | " + " | ".join(FAMILIES) + " | Brier | ECE | Coverage@5% |")
    print("|---|---|" + "---|" * len(FAMILIES) + "---|---|---|")
    for p in preds:
        b = test["predictors"][p]["all"]
        fams = " | ".join(fmt(b["families"][f]["accuracy"]) for f in FAMILIES)
        cov = "n/a" if b.get("automation_coverage") is None else fmt(b["automation_coverage"], 2)
        print(f"| {p} | {fmt(b['macro_accuracy'])} | {fams} | {fmt(b.get('macro_brier'))} | "
              f"{fmt(b.get('macro_ece'))} | {cov} |")
    print("\n<!-- source: docs/TEST_RESULTS.json (predictors.*.{all,A,B}; sensitivity_excl_pick_option.predictors) -->")
    print("| Model | All | A half | B half | B − A (points) | All, 6 families | A, 6 | B, 6 |")
    print("|---|---|---|---|---|---|---|---|")
    for p in preds:
        a, b = test["predictors"][p]["A"]["macro_accuracy"], test["predictors"][p]["B"]["macro_accuracy"]
        six = (sens.get("predictors") or {}).get(p, {})
        cells = " | ".join(fmt((six.get(h) or {}).get("macro_accuracy")) for h in ("all", "A", "B"))
        print(f"| {p} | {fmt(test['predictors'][p]['all']['macro_accuracy'])} | {fmt(a)} | {fmt(b)} | "
              f"{pts(b - a):+.1f} | {cells} |")
    print("\n<!-- source: docs/TEST_RESULTS.json (paired_adapter_vs; "
          "sensitivity_excl_pick_option.paired_adapter_vs) -->")
    print("| Adapter − model | All | A half | B half | All, 6 families | A, 6 | B, 6 | p (Holm), every column |")
    print("|---|---|---|---|---|---|---|---|")
    pairs, spairs = test["paired_adapter_vs"], sens.get("paired_adapter_vs") or {}
    for k in sorted(pairs, key=lambda k: pairs[k]["all"]["delta"]):
        cells = [ci(pairs[k][h]) for h in ("all", "A", "B")]
        cells += [ci(spairs[k][h]) if k in spairs else "—" for h in ("all", "A", "B")]
        holms = {v["p_holm"] for h in ("all", "A", "B") for v in (pairs[k][h], spairs.get(k, {}).get(h, pairs[k][h]))}
        holm = ", ".join(f"{x:.1g}" for x in sorted(holms))
        print(f"| {k.removeprefix('adapter - ')} | " + " | ".join(cells) + f" | {holm} |")
    print("\n<!-- source: docs/TEST_RESULTS.json (predictors.*.all.safety_fn_rate) -->")
    print("| Model | forgotten_violation | needs_approval | share_ok |")
    print("|---|---|---|---|")
    for p in preds:
        r = test["predictors"][p]["all"]["safety_fn_rate"]
        print(f"| {p} | " + " | ".join(f"{100 * r[f]:.1f}%" for f in ("forgotten_violation", "needs_approval",
                                                                          "share_ok")) + " |")


def main() -> None:
    test = load("docs/TEST_RESULTS.json")
    if sys.argv[1:] == ["tables"]:
        if test is None:
            raise SystemExit("docs/TEST_RESULTS.json does not exist yet")
        tables(test)
        return
    hidden, rounds = load("docs/tech-report/data/hidden.json"), load("docs/tech-report/data/dev-rounds.json")
    fig_hidden_accuracy(hidden)
    fig_hidden_deltas(hidden, rounds)
    fig_rounds(rounds)
    fig_automation(hidden)
    fig_dev_references(load("docs/reference-dev.json"))
    if test is not None:
        fig_test(test)
    else:
        print("docs/TEST_RESULTS.json not found: TEST figures skipped")


if __name__ == "__main__":
    main()
