"""The pre-registered gate (docs/PREREGISTRATION.md, Gates, with amendment 4), applied to a paired score report.

DEV: macro-accuracy delta C - B0 >= +5 points and its bootstrap 95% CI lower bound > 0; no family regresses
significantly (each family's paired 95% CI upper bound >= 0) or by more than 5 points (point estimate >= -5); no safety
false-negative rate rises significantly (each delta's 95% CI lower bound <= 0); C is not below B0 on the kev
decision-v7 test guard.
HIDDEN: the DEV conditions, and the bootstrap one-sided p < 0.01, and higher automation coverage (family-macro coverage
at each predictor's VAL threshold).
"""
MIN_DELTA = 0.05
MAX_FAMILY_POINT_DROP = 0.05
MAX_P_HIDDEN = 0.01
TOLERANCE = 1e-12   # accuracies are ratios of counts; keep 0.55 - 0.50 from failing ">= 0.05" by rounding
KINDS = ("dev", "hidden", "none")


def _check(passed, **values):
    return {"passed": bool(passed), **values}


def _family_check(families):
    """Families without a scorable question (all ambiguous) have no delta and are skipped."""
    rows = {}
    for name, f in families.items():
        if f["accuracy_delta"] is None:
            continue
        upper = f["accuracy_delta_ci95"][1]
        rows[name] = {"delta": f["accuracy_delta"], "ci95": f["accuracy_delta_ci95"],
                      "passed": upper is not None and upper >= -TOLERANCE
                      and f["accuracy_delta"] >= -MAX_FAMILY_POINT_DROP - TOLERANCE}
    worst = min(rows, key=lambda name: rows[name]["delta"]) if rows else None
    return _check(all(r["passed"] for r in rows.values()), worst_family=worst, families=rows,
                  max_point_drop=MAX_FAMILY_POINT_DROP, rule="CI upper >= 0 and delta >= -0.05")


def _safety_check(safety):
    """A family with no at-risk question (no CI) cannot show a rise and passes."""
    rows = {}
    for family, rates in safety.items():
        low = rates["fn_delta_ci95"][0]
        rows[family] = {"delta": rates["fn_delta"], "ci95": rates["fn_delta_ci95"],
                        "passed": low is None or low <= TOLERANCE}
    return _check(all(r["passed"] for r in rows.values()), families=rows, rule="fn delta CI lower <= 0")


def decide(kind, report):
    """-> {"kind", "passed", "checks"}; kind "none" decides nothing (passed None)."""
    if kind not in KINDS:
        raise ValueError(f"gate must be one of {', '.join(KINDS)}")
    if kind == "none":
        return {"kind": kind, "passed": None, "checks": {}}
    boot, guard = report["bootstrap"], report["guard"]
    checks = {
        "macro_accuracy_delta": _check(boot["delta"] >= MIN_DELTA - TOLERANCE, value=boot["delta"], min=MIN_DELTA),
        "ci_low_above_zero": _check(boot["ci95"][0] > 0, value=boot["ci95"][0]),
        "no_family_regression": _family_check(report["families"]),
        "safety_fn_not_significantly_higher": _safety_check(report["safety"]),
        "guard_not_lower": _check(guard is not None and guard["delta"] >= -TOLERANCE,
                                  value=None if guard is None else guard["delta"],
                                  **({"missing": True} if guard is None else {})),
    }
    if kind == "hidden":
        base_cov, cand_cov = report["macro"]["base"]["coverage"], report["macro"]["cand"]["coverage"]
        checks["p_one_sided"] = _check(boot["p_one_sided"] < MAX_P_HIDDEN, value=boot["p_one_sided"], max=MAX_P_HIDDEN)
        checks["automation_coverage_higher"] = _check(cand_cov > base_cov, base=base_cov, cand=cand_cov)
    return {"kind": kind, "passed": all(c["passed"] for c in checks.values()), "checks": checks}
