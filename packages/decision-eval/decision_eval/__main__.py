"""`python -m decision_eval <command> [args]`: the same commands as the console scripts."""
import sys

from .cli import diagnose, fit, predict, reference, release, score, sft

COMMANDS = {
    "predict-base": predict.predict_base_main,
    "predict-kev": predict.predict_kev_main,
    "predict-kev-ref": predict.predict_kev_ref_main,
    "predict-openai": reference.predict_openai_main,
    "predict-jev": reference.predict_jev_main,
    "to-sft": sft.to_sft_main,
    "sft-rows": sft.to_sft_main,   # the name the post-training runtime calls it by
    "select-template": fit.select_template_main,
    "calibrate": fit.calibrate_main,
    "thresholds": fit.thresholds_main,
    "score": score.score_main,
    "score-reference": score.score_reference_main,
    "noise-floor": diagnose.noise_floor_main,
    "validity": diagnose.validity_main,
    "export-hillclimb": diagnose.export_hillclimb_main,
    "compare-release": release.compare_release_main,
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        raise SystemExit(f"usage: python -m decision_eval {{{','.join(COMMANDS)}}} [args]")
    COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
