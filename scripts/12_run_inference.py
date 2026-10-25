#!/usr/bin/env python3
"""
Retrobiosynthesis CLI Runner

Accepts a target natural product SMILES, runs the under-the-hood 3-Head Reverse-Peeling
GNN + PATRO LSTM + `deterministic_stage` PKS/NRPS Solver (`deterministic_stage/retrobiosynthesis_engine.py`),
prints the step-by-step enzymatic peeling pathway and module/domain table, and optionally
saves a pathway diagram.

Usage:
    python scripts/12_run_inference.py --smiles "<TARGET_SMILES>"
    python scripts/12_run_inference.py --smiles "<TARGET_SMILES>" --plot_out plots/my_pathway.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deterministic_stage.retrobiosynthesis_engine import (
    DEFAULT_CHECKPOINT,
    DEFAULT_PLOT_OUT,
    build_enzymatic_rule_bank,
    format_deterministic_stage_report,
    get_compute_device,
    load_peeling_model,
    render_peeling_pathway_image,
    run_reverse_peeling_search,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end retrobiosynthetic peeling and deterministic PKS/NRPS module readout on an input SMILES."
    )
    parser.add_argument("--smiles", type=str, required=True, help="Input product SMILES to decompose")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to trained 3-head GNN checkpoint")
    parser.add_argument("--max_depth", type=int, default=6, help="Maximum reverse-peeling steps")
    parser.add_argument("--beam_width", type=int, default=8, help="Beam search width")
    parser.add_argument("--plot_out", type=str, default=None, help="Optional PNG path to save the visual pathway diagram")
    args = parser.parse_args()

    device = get_compute_device()
    model = load_peeling_model(args.checkpoint, device)
    rule_bank = build_enzymatic_rule_bank()

    result = run_reverse_peeling_search(
        model=model,
        target_smiles=args.smiles,
        rule_bank=rule_bank,
        device=device,
        max_depth=args.max_depth,
        beam_width=args.beam_width,
    )
    print(format_deterministic_stage_report(result))

    if args.plot_out:
        render_peeling_pathway_image(result, args.plot_out)


if __name__ == "__main__":
    main()
