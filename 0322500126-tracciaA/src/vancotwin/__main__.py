"""Entry point: ``python -m vancotwin [--quick] [--output DIR]``."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from .config import RunConfig
from .pipeline import run


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Digital Twin PK/PD della vancomicina")
    parser.add_argument("--output", default="outputs", help="cartella dei risultati")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=None, help="pazienti usati per la validazione")
    parser.add_argument("--quick", action="store_true", help="esecuzione rapida (meno pazienti e campioni MCMC)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("pymc", "pytensor", "arviz"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    cfg = RunConfig(seed=args.seed, output_dir=args.output)
    if args.quick:
        cfg = replace(cfg, n_patients=40, n_eval=5, draws=300, tune=300, n_ensemble=150)
    if args.n_eval is not None:
        cfg = replace(cfg, n_eval=args.n_eval)
    run(cfg)


if __name__ == "__main__":
    main()
