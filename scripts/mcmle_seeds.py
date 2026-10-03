"""The Monte Carlo MLE of every dyad-dependent reference model, over several
seeds, against R's (tests/data/r_reference.json): how many converged, the
largest distance from R's estimates in R's standard errors, and the range
of the standard errors' ratios to R's. The numbers of docs/validation.md.

Run from the root of the repository:
    uv run python scripts/mcmle_seeds.py [--seeds 5] [model ...]
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from conftest import REFERENCE, estimated, formula, load, options  # noqa: E402

import ergmx  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default=None, help="write the results as JSON")
    args = parser.parse_args()
    names = args.models or sorted(n for n, m in REFERENCE.items()
                                  if "mle" in m["checks"] and not m["dyad_independent"])
    results = {}
    for name in names:
        model = REFERENCE[name]
        g = load(model["network"])
        terms = estimated(model)
        rows = []
        for seed in range(1, args.seeds + 1):
            start = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fit = ergmx.ergm(g, formula(model), seed=seed, eval_loglik=False, **options(model))
                except Exception as e:  # noqa: BLE001
                    rows.append({"seed": seed, "error": f"{type(e).__name__}: {e}"[:200]})
                    continue
            z = [(fit.coef[t] - model["mle"][t]) / model["se"][t] for t in terms]
            ratio = [fit.stderr[t] / model["se"][t] for t in terms]
            rows.append({"seed": seed, "converged": bool(fit.converged), "iterations": fit.iterations,
                         "max_z": float(np.max(np.abs(z))), "se_ratio": [float(min(ratio)), float(max(ratio))],
                         "seconds": time.time() - start})
        ok = [r for r in rows if "error" not in r]
        summary = {
            "converged": sum(r["converged"] for r in ok), "fits": len(rows),
            "max_z": max((r["max_z"] for r in ok), default=np.nan),
            "se_ratio": [min((r["se_ratio"][0] for r in ok), default=np.nan),
                         max((r["se_ratio"][1] for r in ok), default=np.nan)],
            "iterations": [r["iterations"] for r in ok], "seconds": sum(r.get("seconds", 0) for r in ok),
            "errors": [r["error"] for r in rows if "error" in r],
        }
        results[name] = summary
        print(f"{name:28s} converged {summary['converged']}/{summary['fits']}  max |z| {summary['max_z']:.2f}  "
              f"SE ratio {summary['se_ratio'][0]:.2f}-{summary['se_ratio'][1]:.2f}  "
              f"iterations {summary['iterations']}  {summary['seconds']:.0f} s"
              + (f"  errors {summary['errors']}" if summary["errors"] else ""), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
