#!/usr/bin/env python3
"""
End-to-End Retrobiosynthetic Reverse-Peeling Search + `deterministic_stage` Solver

Given a target polyketide natural product SMILES (with post-PKS tailoring modifications
such as deoxysugar glycosylations, O/C-methylations, P450 hydroxylations/epoxidations,
halogenations, Baeyer-Villiger insertions, or intramolecular Diels-Alder/oxa-Michael bridges):

1. Fast Terminal Check (`deterministic_stage/pks_terminal.py`):
   First checks if the input molecule is already a nascent PKS core via `decompose_pks_terminal`.
2. GNN-Guided Beam Search / MCTS Reverse Peeling:
   - Evaluates the 3-Head `ReversePeelingGNN`:
     * Head 1 (`core_node_logit`, `register_logits`, `bridge_edge_logit`): Predicts which atoms
       belong to the canonical PKS core (`p_core(v)`) vs. post-PKS decorations (`1 - p_core(v)`),
       so genuine inline PKS groups (KR beta-OH, methylmalonyl alpha-methyl) are protected!
     * Head 2 (`exposed_node_logit`, `rule_logits`): Ranks the 16 reverse enzymatic rules
       (`r^{-1}`) and scores candidate reaction sites by exposed-leaf probability.
     * Head 3 (`value_pred`, `cls_logit`): Scores resulting peeled intermediates `M'` by
       predicted proximity `V(M') = exp(-d/2.5)` to a valid nascent PKS aglycone.
   - Also integrates `pks_terminal._explain_failure` structural difference diagnostics
     (`extra_heteroatoms_in_input`, `extra_oxygen_in_input`, `extra_rings_in_input`,
     `extra_carbons_in_input`) to further prioritize matching reverse rules.
3. Exact Deterministic Stage Handoff (`ringbreaker.py` -> `carbon_backbone.py` ->
   `predict_extenders.py` -> `predict_reduction.py`):
   - As soon as a peeled intermediate `M_0` satisfies `decompose_pks_terminal(M_0)`
     (or achieves an exact constitutional/stereochemical forward rebuild), terminates search
     and outputs:
     * The multi-step reverse-peeling pathway `M_target -> ... -> M_0`
     * The linearized acyl-ACP precursor & release mechanism
     * Exact module count, starter unit, AT extender annotations, C1-outward alpha/beta
       carbon indices, and Keiser 2025 KR/DH/ER stereochemical domain readout.
   - Renders a visual pathway + backbone/module readout diagram to `plots/`.

Usage:
    python scripts/12_run_inference.py --smiles "<TARGET_SMILES>"
    python scripts/12_run_inference.py --evaluate_test_trajectories 50
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
import io
import matplotlib.pyplot as plt
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "SynKit") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "SynKit"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synkit.IO import rsmi_to_its
from synkit.Synthesis.Reactor import SynReactor
from deterministic_stage import pks_terminal as pt


def _load_script_module(script_name: str, module_alias: str):
    path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = mod
    spec.loader.exec_module(mod)
    return mod


gnn_mod = _load_script_module("09_train_gnn_classifier.py", "train_gnn_mod")
aug_mod = _load_script_module("02_generate_PKS_augmentations.py", "aug_mod")

ReversePeelingGNN = gnn_mod.ReversePeelingGNN
smiles_to_graph = gnn_mod.smiles_to_graph
REGISTER_ROLES = gnn_mod.REGISTER_ROLES
EnzymaticTailoringRule = aug_mod.EnzymaticTailoringRule
build_enzymatic_rule_bank = aug_mod.build_enzymatic_rule_bank

DEFAULT_CHECKPOINT = str(PROJECT_ROOT / "models" / "supervised_gnn" / "best_model.pt")
DEFAULT_PLOT_OUT = str(PROJECT_ROOT / "plots" / "reverse_peeling_pathway_inference.png")


# =============================================================================
# Model & Rule Bank Loading
# =============================================================================

def load_peeling_model(checkpoint_path: str, device: torch.device) -> ReversePeelingGNN:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    saved_args = checkpoint.get("args", {})

    node_dim = checkpoint.get("node_dim", gnn_mod.NODE_FEAT_DIM)
    edge_dim = checkpoint.get("edge_dim", gnn_mod.EDGE_FEAT_DIM)
    hidden_dim = saved_args.get("hidden_dim", 192)
    num_heads = saved_args.get("num_heads", 4)
    num_layers = saved_args.get("num_layers", 3)
    dropout = saved_args.get("dropout", 0.1)

    model = ReversePeelingGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        heads=num_heads,
        num_layers=num_layers,
        num_rules=checkpoint.get("num_rules", gnn_mod.NUM_ENZYMATIC_RULES),
        num_register_roles=checkpoint.get("num_register_roles", gnn_mod.NUM_REGISTER_ROLES),
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_molecule_with_gnn(
    model: ReversePeelingGNN,
    smiles: str,
    device: torch.device,
    bgc_rule_mask: Optional[torch.Tensor] = None,
    pathway_rule_history: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run all 3 heads of `ReversePeelingGNN` (including C1-Register PE, SynCat Cross-Attention,
    and PATRO Pathway-History LSTM + Highway encoding) on a single molecule SMILES.
    Following PATRO (`mcts_controller.py` / `retro_star_controller.py`):
      - At Step 0 (root target molecule), `pathway_rule_history` is empty (`[]`), so
        `path_tokens = [[0, 0, 0, 0]]` ("na" padding tokens -> zero LSTM context) and 100%
        of the decision is driven by the GAT + SynCat Cross-Attention.
      - At Step t >= 1, `pathway_rule_history` contains the `rule_id`s of previously peeled
        reactions in chronological order (`[r_1, ..., r_t]`), mapped to `rule_id + 1` and
        left-padded with `0` ("na") so the most recent reaction is at index `-1`.
    """
    nf, ei, ea = smiles_to_graph(smiles)
    real_n = nf.shape[0]
    real_e = ei.shape[1]
    bucket_n = ((real_n + 127) // 128) * 128
    if bucket_n == real_n:
        bucket_n += 1
    pad_n = bucket_n - real_n
    bucket_e = ((real_e + 511) // 512) * 512
    pad_e = bucket_e - real_e

    nf_t = torch.nn.functional.pad(torch.from_numpy(nf), (0, 0, 0, pad_n)).to(device)
    batch_t = torch.cat(
        [torch.zeros(real_n, dtype=torch.long), torch.ones(pad_n, dtype=torch.long)], dim=0
    ).to(device)
    if pad_e > 0:
        dummy_ei = torch.full((2, pad_e), real_n, dtype=torch.long)
        ei_t = torch.cat([torch.from_numpy(ei), dummy_ei], dim=1).to(device)
        ea_t = torch.nn.functional.pad(torch.from_numpy(ea), (0, 0, 0, pad_e)).to(device)
    else:
        ei_t = torch.from_numpy(ei).to(device)
        ea_t = torch.from_numpy(ea).to(device)

    hist_tokens = [int(r) + 1 for r in (pathway_rule_history or []) if 0 <= int(r) < 60][-4:]
    padded_hist = [0] * (4 - len(hist_tokens)) + hist_tokens
    path_tokens = torch.tensor([padded_hist], dtype=torch.long, device=device)

    out = model.forward_all_heads(
        nf_t, ei_t, batch_t, ea_t, bgc_rule_mask=bgc_rule_mask, pathway_context_tokens=path_tokens
    )

    p_core_node = torch.sigmoid(out["core_node_logit"][:real_n]).cpu().numpy()
    p_exposed_node = torch.sigmoid(out["exposed_node_logit"][:real_n]).cpu().numpy()
    reg_probs = torch.softmax(out["register_logits"][:real_n], dim=-1).cpu().numpy()
    reg_pred_roles = [REGISTER_ROLES[i] for i in reg_probs.argmax(axis=-1)]
    rule_probs_raw = torch.sigmoid(out["rule_logits"][0]).cpu().numpy()
    if len(rule_probs_raw) < 33:
        rule_probs = np.pad(rule_probs_raw, (0, 33 - len(rule_probs_raw)), constant_values=0.50)
    else:
        rule_probs = rule_probs_raw
    syncat_att_map = (
        out["syncat_att_map"][:, :real_n].cpu().numpy()
        if "syncat_att_map" in out
        else np.ones((len(rule_probs), real_n), dtype=np.float32) / max(1, real_n)
    )
    value_score = float(out["value_pred"][0].cpu().item())
    core_cls_prob = float(torch.sigmoid(out["cls_logit"][0, 0]).cpu().item())

    return {
        "p_core_node": p_core_node,
        "p_exposed_node": p_exposed_node,
        "register_roles": reg_pred_roles,
        "rule_probs": rule_probs,
        "syncat_att_map": syncat_att_map,
        "value_score": value_score,
        "core_cls_prob": core_cls_prob,
    }


# =============================================================================
# SynKit DPO / ITS Graph Rule Rewriting + Stereochemical Retention
# =============================================================================

_ITS_CACHE: Dict[int, Any] = {}
_MXN_NRPS_AA_CATALOG: Optional[Dict[str, str]] = None


def get_mxn_nrps_amino_acid_catalog() -> Dict[str, str]:
    """
    Load all 328 NRPS amino acids, non-proteinogenic monomers, and hybrid offloading derivatives
    from `RetroMol/src/retromol/data/mxn.yml` into a canonical constitution SMILES -> monomer name lookup.
    """
    global _MXN_NRPS_AA_CATALOG
    if _MXN_NRPS_AA_CATALOG is not None:
        return _MXN_NRPS_AA_CATALOG

    catalog: Dict[str, str] = {}
    mxn_candidates = [
        PROJECT_ROOT / "data" / "mxn.yml",
        PROJECT_ROOT / "RetroMol" / "src" / "retromol" / "data" / "mxn.yml",
    ]
    mxn_path = next((p for p in mxn_candidates if p.exists()), mxn_candidates[0])
    if mxn_path.exists():
        try:
            with open(mxn_path, "r") as f:
                entries = yaml.safe_load(f) or []
            for entry in entries:
                if "PK" in entry.get("pseudonyms", []) or entry.get("name") == "glycosylation":
                    continue
                m = Chem.MolFromSmiles(str(entry.get("smiles", "")))
                if m is not None:
                    Chem.RemoveStereochemistry(m)
                    catalog[Chem.MolToSmiles(m, isomericSmiles=False)] = str(entry["name"])
        except Exception:
            pass

    # Include canonical R-domain / decarboxylated / piperazic NRPS offloading monomers from hybrid BGCs
    extra_monomers = {
        "CC(CO)N": "L-alaninol (reduced L-alanine via NRPS R domain)",
        "NC=Cc1ccc(O)c(Cl)c1": "3-chloro-beta-tyrosine styrylamine (CndF)",
        "O=C(O)C1CCCNN1": "L-piperazic acid",
        "NC(Cc1cccc(O)c1)C(=O)O": "L-meta-tyrosine",
    }
    for smi_ex, name_ex in extra_monomers.items():
        m_ex = Chem.MolFromSmiles(smi_ex)
        if m_ex is not None:
            Chem.RemoveStereochemistry(m_ex)
            catalog[Chem.MolToSmiles(m_ex, isomericSmiles=False)] = name_ex

    _MXN_NRPS_AA_CATALOG = catalog
    return catalog


def _match_mxn_nrps_peptide_cleavage(
    mol: Chem.Mol,
    rule_id: int,
    p_core_node: np.ndarray,
    p_exposed_node: np.ndarray,
    syncat_att_map: Optional[np.ndarray] = None,
) -> List[Tuple[str, float, List[int], str]]:
    """
    Use `RetroMol/src/retromol/data/mxn.yml` to detect and cleave terminal or macrocycle-bridging
    NRPS amino acid residues (`Rule 20` for alpha/beta/styryl/alaninol amides, `Rule 24` for
    cyclic proline/pipecolate/piperazate imides), returning `(peeled_smiles, site_score, rc_atoms, aa_name)`
    scored by the exact same neural `0.50 * site_exposed_score + 0.35 * site_decor_score + syncat_boost` formula.
    """
    if rule_id not in (20, 24):
        return []

    aa_catalog = get_mxn_nrps_amino_acid_catalog()
    tagged_mol = Chem.Mol(mol)
    n_atoms = tagged_mol.GetNumAtoms()
    for atom in tagged_mol.GetAtoms():
        atom.SetIsotope(atom.GetIdx() + 1)

    # First, if macrocyclic ester (depsipeptide like Rapamycin, FK520, Meridamycin, Sanglifehrin),
    # allow opening the macrolactone ester so the bridging pipecolate/piperazate C-terminus is exposed
    candidates_to_cleave = [tagged_mol]
    rxn_macro_ester = AllChem.ReactionFromSmarts("[C;R:1](=[O:2])-[O;R:3]-[C;R:4]>>([C:1](=[O:2])O.[O;H1:3]-[C:4])")
    for p_mac in rxn_macro_ester.RunReactants((tagged_mol,)):
        if p_mac:
            c_mac = Chem.Mol(p_mac[0])
            try:
                Chem.SanitizeMol(c_mac)
                if len(Chem.GetMolFrags(c_mac)) == 1:
                    candidates_to_cleave.append(c_mac)
            except Exception:
                pass

    # Cleave amide bond: [#6:5]-[C:1](=[O:2])-[N;X3:3]-[C:4] >> [#6:5]-[C:1](=[O:2])O + [N:3]-[C:4]
    rxn_amide = AllChem.ReactionFromSmarts("[#6:5]-[C:1](=[O:2])-[N;X3:3]-[C:4]>>[#6:5]-[C:1](=[O:2])O.[N:3]-[C:4]")
    hits: List[Tuple[str, float, List[int], str]] = []

    for work_mol in candidates_to_cleave:
        for prod_pair in rxn_amide.RunReactants((work_mol,)):
            if len(prod_pair) < 2:
                continue
            pks_frag, aa_frag = Chem.Mol(prod_pair[0]), Chem.Mol(prod_pair[1])
            try:
                Chem.SanitizeMol(pks_frag)
                Chem.SanitizeMol(aa_frag)
                Chem.AssignStereochemistry(pks_frag, cleanIt=True, force=True)
            except Exception:
                continue
            if len(Chem.GetMolFrags(pks_frag)) != 1 or pks_frag.GetNumAtoms() < 6:
                continue

            aa_no_stereo = Chem.Mol(aa_frag)
            for a in aa_no_stereo.GetAtoms():
                a.SetIsotope(0)
                a.SetAtomMapNum(0)
            Chem.RemoveStereochemistry(aa_no_stereo)
            aa_const_smi = Chem.MolToSmiles(aa_no_stereo, isomericSmiles=False)

            if aa_const_smi not in aa_catalog:
                continue

            aa_name = aa_catalog[aa_const_smi]
            is_cyclic_imino = any(k in aa_name.lower() for k in ("proline", "pipecol", "piperaz", "pyrroline"))
            if rule_id == 24 and not is_cyclic_imino:
                continue
            if rule_id == 20 and is_cyclic_imino:
                continue

            surviving_orig = {a.GetIsotope() - 1 for a in pks_frag.GetAtoms() if a.GetIsotope() > 0}
            removed_indices = sorted(i for i in range(n_atoms) if i not in surviving_orig)
            if not removed_indices:
                continue

            for a in pks_frag.GetAtoms():
                a.SetIsotope(0)
                a.SetAtomMapNum(0)
            pks_smi = Chem.MolToSmiles(pks_frag, isomericSmiles=True)

            site_exposed_score = float(np.mean([p_exposed_node[i] for i in removed_indices if i < len(p_exposed_node)]))
            site_decor_score = float(np.mean([1.0 - p_core_node[i] for i in removed_indices if i < len(p_core_node)]))
            syncat_boost = 0.0
            if syncat_att_map is not None and rule_id < syncat_att_map.shape[0]:
                rule_att = syncat_att_map[rule_id]
                uniform_att = 1.0 / max(1, n_atoms)
                rc_att = float(np.mean([rule_att[i] for i in removed_indices if i < len(rule_att)]))
                syncat_boost = min(0.25, max(0.0, (rc_att / max(1e-6, uniform_att)) * 0.05))

            site_score = min(1.0, 0.50 * site_exposed_score + 0.35 * site_decor_score + syncat_boost)
            hits.append((pks_smi, site_score, removed_indices, aa_name))

    return hits


def _verify_or_discover_synkit_dpo_precursors(
    smiles: str, rule: EnzymaticTailoringRule, num_template_matches: int = 1
) -> Set[str]:
    """
    Execute SynKit Double Pushout (DPO) ITS graph rewriting (`SynReactor(..., invert=True)`)
    for `rule.balanced_its_rsmi` on `smiles` (with automatic co-substrate attachment from the RHS
    of `balanced_its_rsmi`). Returns canonical constitution SMILES of DPO-valid precursors.
    Skips combinatorially explosive templates on >32-carbon polyketides/oligosaccharides where
    VF2 subgraph isomorphism on unrooted 6-rings or C-H bonds would enumerate >100k mappings.
    """
    dpo_constitutions: Set[str] = set()
    n_carbons = smiles.upper().count("C")
    if num_template_matches > 12 or n_carbons > 32:
        return dpo_constitutions
    try:
        if rule.rule_id not in _ITS_CACHE:
            _ITS_CACHE[rule.rule_id] = rsmi_to_its(rule.balanced_its_rsmi, core=True)
        its_template = _ITS_CACHE[rule.rule_id]

        # Extract any leaving-group co-substrate on the RHS of balanced_its_rsmi (e.g. .S, .O)
        rhs = rule.balanced_its_rsmi.split(">>")[1]
        rhs_frags = rhs.split(".")[1:]
        clean_co = []
        for f in rhs_frags:
            m_f = Chem.MolFromSmiles(f)
            if m_f is not None:
                for a in m_f.GetAtoms():
                    a.SetAtomMapNum(0)
                clean_co.append(Chem.MolToSmiles(m_f))
        sub_with_co = ".".join([smiles] + clean_co) if clean_co else smiles

        reactor = SynReactor(
            substrate=sub_with_co,
            template=its_template,
            invert=True,
            implicit_temp=True,
            explicit_h=False,
            serialization_errors="skip",
        )
        for rsmi in reactor.smarts_list:
            lhs = rsmi.split(">>")[0]
            # Pick the main organic fragment from the rewritten LHS
            frags = lhs.split(".")
            best_frag = max(frags, key=len)
            m = Chem.MolFromSmiles(best_frag)
            if m is not None:
                for a in m.GetAtoms():
                    a.SetAtomMapNum(0)
                Chem.SanitizeMol(m)
                Chem.RemoveStereochemistry(m)
                dpo_constitutions.add(Chem.MolToSmiles(m, isomericSmiles=False))
    except Exception:
        pass
    return dpo_constitutions


def apply_reverse_rule_candidates(
    smiles: str,
    rule: EnzymaticTailoringRule,
    p_core_node: np.ndarray,
    p_exposed_node: np.ndarray,
    syncat_att_map: Optional[np.ndarray] = None,
    core_protection_threshold: float = 0.88,
) -> List[Tuple[str, float, List[int], str]]:
    """
    Apply `rule` in reverse using stereochemistry-preserving reaction templates from the Rule Bank,
    scoring each candidate precursor purely by the 3-Head GNN's site predictions:
      - Head 2 Exposed-Leaf probability `p_exposed(v)` (weight 0.50)
      - Head 1 Core-Shield protection `1 - p_core(v)` (weight 0.35)
      - SynCat Rule-to-Node Cross-Attention weight `syncat_att_map[rule_id, v]` (up to 0.25)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    rxn_smarts_list = [rule.reverse_smarts] + list(getattr(rule, "extra_reverse_smarts", []))

    # Check if `rule` is an NRPS amino acid condensation rule (Rule 20 or Rule 24) and match against all 328 monomers in `mxn.yml`!
    mxn_aa_hits = _match_mxn_nrps_peptide_cleavage(
        mol, rule.rule_id, p_core_node, p_exposed_node, syncat_att_map=syncat_att_map
    )

    raw_prods = []
    any_match = bool(mxn_aa_hits)
    for r_smarts in rxn_smarts_list:
        rxn = AllChem.ReactionFromSmarts(r_smarts)
        if rxn.GetNumReactantTemplates() != 1:
            continue
        reactant_template = rxn.GetReactantTemplate(0)
        matches = mol.GetSubstructMatches(reactant_template)
        if not matches:
            continue
        any_match = True
        tagged_mol = Chem.Mol(mol)
        for atom in tagged_mol.GetAtoms():
            atom.SetIsotope(atom.GetIdx() + 1)
        try:
            raw_prods.extend(rxn.RunReactants((tagged_mol,)))
        except Exception:
            pass

    if not any_match or (not raw_prods and not mxn_aa_hits):
        return []

    results: Dict[str, Tuple[float, List[int], str]] = {}
    for pks_smi, sc_aa, rc_aa, aa_name in mxn_aa_hits:
        if pks_smi != smiles and (pks_smi not in results or sc_aa > results[pks_smi][0]):
            results[pks_smi] = (sc_aa, rc_aa, aa_name)
    n_atoms = len(p_core_node)

    for prod_tuple in raw_prods:
        if not prod_tuple:
            continue
        cand = Chem.Mol(prod_tuple[0])
        try:
            Chem.SanitizeMol(cand)
            Chem.AssignStereochemistry(cand, cleanIt=True, force=True)
        except Exception:
            continue

        frags = Chem.GetMolFrags(cand, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            # When cleaving a multi-sugar oligosaccharide chain or bulky ester,
            # retain the largest organic scaffold fragment (the polyketide aglycone/core)
            cand = max(frags, key=lambda m_f: m_f.GetNumAtoms())
        if cand.GetNumAtoms() < 4:
            continue

        surviving_orig_indices = {
            a.GetIsotope() - 1 for a in cand.GetAtoms() if a.GetIsotope() > 0
        }
        removed_indices = [i for i in range(n_atoms) if i not in surviving_orig_indices]

        # Enforce biologically faithful 1-monosaccharide-per-enzyme-step cleavage for glycosyltransferases (Rule 13):
        # One GT enzyme (e.g. KijC1, KijC2, KijC3, KijC4, DesVII, EryBV) transfers ONE dTDP-sugar (6 to 16 heavy atoms),
        # never an entire 3- or 4-sugar oligosaccharide tree (>=18 heavy atoms) in a single catalytic step!
        if rule.rule_id == 13 and len(removed_indices) > 17:
            continue

        modified_surviving = []
        for a in cand.GetAtoms():
            orig_idx = a.GetIsotope() - 1
            if 0 <= orig_idx < n_atoms:
                orig_a = mol.GetAtomWithIdx(orig_idx)
                if (
                    a.GetDegree() != orig_a.GetDegree()
                    or a.GetTotalNumHs() != orig_a.GetTotalNumHs()
                    or int(a.IsInRing()) != int(orig_a.IsInRing())
                ):
                    modified_surviving.append(orig_idx)

        rc_indices = sorted(set(removed_indices + modified_surviving))
        if not rc_indices:
            continue

        if removed_indices:
            mean_removed_core_prob = float(np.mean([p_core_node[i] for i in removed_indices if i < n_atoms]))
            # Head 1 Core-Shield veto applies when `rule.can_collide_with_inline_pks_core` is True
            # (Rule 3 dehydroxylation where -OH can collide with inline KR beta-OH, or
            #  Rule 9 C-demethylation where -CH3 can collide with inline AT mmal alpha-CH3).
            if getattr(rule, "can_collide_with_inline_pks_core", False) and mean_removed_core_prob > core_protection_threshold:
                continue
            site_decor_score = 1.0 - mean_removed_core_prob
        else:
            site_decor_score = float(np.mean([1.0 - p_core_node[i] for i in rc_indices if i < n_atoms]))

        site_exposed_score = float(np.mean([p_exposed_node[i] for i in rc_indices if i < n_atoms]))

        # SynCat Rule-to-Node Cross-Attention boost on matched reaction-center atoms
        syncat_boost = 0.0
        if syncat_att_map is not None and rule.rule_id < syncat_att_map.shape[0]:
            rule_att = syncat_att_map[rule.rule_id]
            uniform_att = 1.0 / max(1, n_atoms)
            rc_att = float(np.mean([rule_att[i] for i in rc_indices if i < len(rule_att)]))
            syncat_boost = min(0.25, max(0.0, (rc_att / max(1e-6, uniform_att)) * 0.05))

        for a in cand.GetAtoms():
            a.SetIsotope(0)
            a.SetAtomMapNum(0)
        cand_smi = Chem.MolToSmiles(cand, isomericSmiles=True)
        if cand_smi == smiles:
            continue

        site_score = min(
            1.0,
            0.50 * site_exposed_score + 0.35 * site_decor_score + syncat_boost,
        )

        if cand_smi not in results or site_score > results[cand_smi][0]:
            default_aa = "glycine" if rule.rule_id == 20 else ("pipecolic acid" if rule.rule_id == 24 else "")
            results[cand_smi] = (site_score, rc_indices, default_aa)

    return [(smi, sc, rc, aa) for smi, (sc, rc, aa) in results.items()]


# =============================================================================
# GNN + Deterministic-Stage Guided Beam / MCTS Search
# =============================================================================

@dataclass
class PeelingStepRecord:
    step_num: int
    from_smiles: str
    to_smiles: str
    rule_id: int
    rule_name: str
    enzyme_family: str
    ec_class: str
    candidate_enzymes: str
    uniprot_ids: str
    pfam_domains: str
    cofactors: str
    phase: int
    rule_prob: float
    site_score: float
    value_score: float
    reaction_center_atoms: List[int]
    matched_mxn_monomer: str = ""


@dataclass
class SearchNode:
    smiles: str
    depth: int
    cumulative_score: float
    value_score: float
    core_cls_prob: float
    gnn_eval: Dict[str, Any]
    pathway: List[PeelingStepRecord] = field(default_factory=list)


_PYRANOSIDE_PATTERN = Chem.MolFromSmarts("[#6;!$(C=O);!r6]-[O;!R;D2]-[C;r6]1-[O;r6]-[C;r6]-[C;r6]-[C;r6]-[C;r6]-1")
_PYRONE_TAUTOMER_RXNS = [
    # 4-hydroxy-2-pyrone (enol-lactone of beta,delta-diketo acid, e.g. TAL, Kava pyrones)
    AllChem.ReactionFromSmarts(
        "[c:1]1(=[O:2])[cH:3][c:4](-[O;H1:5])[cH:6][c:7](-[#6:8])[o:9]1>>[#6:8]-[C:7](=[O:9])-[CH2:6]-[C:4](=[O:5])-[CH2:3]-[C:1](=[O:2])O"
    ),
    # 3-substituted-4-hydroxy-2-pyrone (e.g. Germicidin A, Photopyrone A)
    AllChem.ReactionFromSmarts(
        "[c:1]1(=[O:2])[c:3](-[#6:10])[c:4](-[O;H1:5])[cH:6][c:7](-[#6:8])[o:9]1>>[#6:8]-[C:7](=[O:9])-[CH2:6]-[C:4](=[O:5])-[CH1:3](-[#6:10])-[C:1](=[O:2])O"
    ),
    # Unsubstituted 6-alkyl/aryl-2-pyrone (enol-lactone of 5-keto-2-enoic acid, e.g. 6-pentyl-alpha-pyrone)
    AllChem.ReactionFromSmarts(
        "[c:1]1(=[O:2])[cH:3][cH:4][cH:6][c:7](-[#6:8])[o:9]1>>[#6:8]-[C:7](=[O:9])-[CH2:6]-[CH:4]=[CH:3]-[C:1](=[O:2])O"
    ),
]


def check_deterministic_pks_terminus(smiles: str) -> Tuple[bool, str, Tuple[Dict, ...], Dict[str, Any]]:
    """
    Check if `smiles` is accepted by `deterministic_stage/pks_terminal.py`.
    Fast pre-filter: skips expensive 50,000-visit DFS if the molecule still carries
    attached 6-membered deoxysugar pyranosides (`_PYRANOSIDE_PATTERN`) or > 54 heavy atoms.
    Also handles aromatic 2-pyrone / 4-hydroxy-2-pyrone enol-lactone tautomerization.
    """
    mol_check = Chem.MolFromSmiles(smiles)
    if mol_check is None:
        return False, "invalid_smiles", (), {}
    if mol_check.GetNumHeavyAtoms() > 54 or (
        _PYRANOSIDE_PATTERN is not None and mol_check.HasSubstructMatch(_PYRANOSIDE_PATTERN)
    ):
        return False, "still_glycosylated_or_too_large", (), {}

    candidate_smiles_to_test = [smiles]
    for rxn_pyr in _PYRONE_TAUTOMER_RXNS:
        if rxn_pyr is None:
            continue
        try:
            for prod_t in rxn_pyr.RunReactants((mol_check,)):
                if prod_t:
                    p_mol = Chem.Mol(prod_t[0])
                    Chem.SanitizeMol(p_mol)
                    candidate_smiles_to_test.append(Chem.MolToSmiles(p_mol, isomericSmiles=True))
        except Exception:
            pass

    last_status = "failed"
    last_meta: Dict[str, Any] = {}
    for test_smi in candidate_smiles_to_test:
        rows, meta = pt.decompose_pks_terminal(
            test_smi, max_visits=45_000, backbone_top_k=6, extended_starters=True
        )
        status = meta.get("status", "failed")
        last_status, last_meta = status, meta
        if status == "ok" and rows:
            return True, "exact_stereo_and_constitution", rows, meta

        for cand in meta.get("candidates", ()):
            val = cand.get("assembly_validation", {})
            if val.get("constitution_match") and val.get("release_roundtrip") and cand.get("module_rows"):
                return True, "constitution_rebuild_match", tuple(cand["module_rows"]), meta

    return False, last_status, (), last_meta


def run_reverse_peeling_search(
    model: ReversePeelingGNN,
    target_smiles: str,
    rule_bank: List[EnzymaticTailoringRule],
    device: torch.device,
    max_depth: int = 4,
    beam_width: int = 8,
    top_k_rules_per_node: int = 14,
    bgc_enzyme_families: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Run GNN-guided Beam/MCTS Reverse Peeling on `target_smiles`, checking each
    peeled candidate against `deterministic_stage/pks_terminal.py`.
    """
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        raise ValueError(f"Invalid target SMILES: {target_smiles}")
    canon_target = Chem.MolToSmiles(mol, isomericSmiles=True)

    # Optional BGC enzyme-family mask
    bgc_mask = None
    if bgc_enzyme_families:
        mask_arr = np.array(
            [1.0 if r.enzyme_family in bgc_enzyme_families else 0.0 for r in rule_bank],
            dtype=np.float32,
        )
        bgc_mask = torch.from_numpy(mask_arr).unsqueeze(0).to(device)

    gnn_0 = evaluate_molecule_with_gnn(model, canon_target, device, bgc_rule_mask=bgc_mask)

    # Step 0: Check if target is already a nascent PKS core
    is_term, match_kind, rows, meta = check_deterministic_pks_terminus(canon_target)
    if is_term:
        return {
            "success": True,
            "match_kind": match_kind,
            "target_smiles": canon_target,
            "peeled_core_smiles": canon_target,
            "peeling_depth": 0,
            "pathway": [],
            "initial_gnn": gnn_0,
            "final_gnn": gnn_0,
            "module_rows": list(rows),
            "terminal_metadata": meta,
            "explored_nodes": 1,
        }

    root_node = SearchNode(
        smiles=canon_target,
        depth=0,
        cumulative_score=0.0,
        value_score=gnn_0["value_score"],
        core_cls_prob=gnn_0["core_cls_prob"],
        gnn_eval=gnn_0,
        pathway=[],
    )

    beam: List[SearchNode] = [root_node]
    best_score_by_state: Dict[str, float] = {canon_target: 0.0}
    gnn_eval_cache: Dict[Tuple[str, Tuple[int, ...]], Dict[str, Any]] = {
        (canon_target, ()): gnn_0
    }
    explored_count = 1
    best_fallback_node = root_node

    rule_by_id = {r.rule_id: r for r in rule_bank}

    for depth in range(1, max_depth + 1):
        next_beam_by_smi: Dict[str, SearchNode] = {}
        terminal_hits_at_depth: List[Tuple[Tuple[int, float, float], Dict[str, Any]]] = []

        for node in beam:
            rule_probs = node.gnn_eval["rule_probs"]
            ancestors = {canon_target} | {s.to_smiles for s in node.pathway}

            # Pure neural ranking from Head 2 (GAT + SynCat Cross-Attention + PATRO LSTM Highway)
            eff_top_k = len(rule_bank) if len(node.gnn_eval["p_core_node"]) <= 38 else top_k_rules_per_node
            ranked_rule_ids = np.argsort(-rule_probs)[:eff_top_k].tolist()
            for r in rule_bank:
                if rule_probs[r.rule_id] >= 0.05 and r.rule_id not in ranked_rule_ids:
                    ranked_rule_ids.append(r.rule_id)

            for r_id in ranked_rule_ids:
                rule = rule_by_id[int(r_id)]
                if bgc_enzyme_families and rule.enzyme_family not in bgc_enzyme_families:
                    continue

                precursors = apply_reverse_rule_candidates(
                    node.smiles,
                    rule,
                    node.gnn_eval["p_core_node"],
                    node.gnn_eval["p_exposed_node"],
                    syncat_att_map=node.gnn_eval.get("syncat_att_map"),
                )

                for cand_smi, site_score, rc_atoms, aa_name in precursors:
                    # Cycle prevention (PATRO retro_star_controller.py line 461: `if ancestors & set(reactant_list): continue`)
                    if cand_smi in ancestors:
                        continue

                    # Construct PATRO pathway reaction history (oldest-to-most-recent peeled rule_ids)
                    new_rule_history = [s.rule_id for s in node.pathway] + [rule.rule_id]
                    cache_key = (cand_smi, tuple(new_rule_history[-4:]))
                    if cache_key in gnn_eval_cache:
                        gnn_cand = gnn_eval_cache[cache_key]
                    else:
                        explored_count += 1
                        gnn_cand = evaluate_molecule_with_gnn(
                            model,
                            cand_smi,
                            device,
                            bgc_rule_mask=bgc_mask,
                            pathway_rule_history=new_rule_history,
                        )
                        gnn_eval_cache[cache_key] = gnn_cand

                    # Pure 3-Head Neural Score (Head 2 Rule Policy + Head 1/2 Site Score + Head 3 Value V(M') + Core Prob)
                    step_transition_score = (
                        0.35 * float(rule_probs[r_id])
                        + 0.25 * float(site_score)
                        + 0.25 * gnn_cand["value_score"]
                        + 0.15 * gnn_cand["core_cls_prob"]
                    )
                    cum_score = node.cumulative_score + step_transition_score

                    # Path-aware best-parent deduplication: only keep if this path improves upon any prior path to `cand_smi`
                    if cum_score <= best_score_by_state.get(cand_smi, -1e9):
                        continue
                    best_score_by_state[cand_smi] = cum_score

                    step_rec = PeelingStepRecord(
                        step_num=depth,
                        from_smiles=node.smiles,
                        to_smiles=cand_smi,
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        enzyme_family=rule.enzyme_family,
                        ec_class=rule.ec_class,
                        candidate_enzymes=rule.candidate_enzymes,
                        uniprot_ids=rule.uniprot_ids,
                        pfam_domains=rule.pfam_domains,
                        cofactors=rule.cofactors,
                        phase=rule.phase,
                        rule_prob=float(rule_probs[r_id]),
                        site_score=float(site_score),
                        value_score=gnn_cand["value_score"],
                        reaction_center_atoms=rc_atoms,
                        matched_mxn_monomer=aa_name,
                    )
                    new_pathway = node.pathway + [step_rec]

                    child_node = SearchNode(
                        smiles=cand_smi,
                        depth=depth,
                        cumulative_score=cum_score,
                        value_score=gnn_cand["value_score"],
                        core_cls_prob=gnn_cand["core_cls_prob"],
                        gnn_eval=gnn_cand,
                        pathway=new_pathway,
                    )

                    # Test candidate with deterministic_stage/pks_terminal.py and collect all terminal hits at this depth
                    is_term, match_kind, rows, meta = check_deterministic_pks_terminus(cand_smi)
                    if is_term:
                        is_exact_stereo = 1 if match_kind == "exact_stereo_and_constitution" else 0
                        rank_key = (
                            is_exact_stereo,
                            cum_score / max(1, depth),
                            gnn_cand["value_score"],
                        )
                        terminal_hits_at_depth.append(
                            (
                                rank_key,
                                {
                                    "success": True,
                                    "match_kind": match_kind,
                                    "target_smiles": canon_target,
                                    "peeled_core_smiles": cand_smi,
                                    "peeling_depth": depth,
                                    "pathway": new_pathway,
                                    "initial_gnn": gnn_0,
                                    "final_gnn": gnn_cand,
                                    "module_rows": list(rows),
                                    "terminal_metadata": meta,
                                    "explored_nodes": explored_count,
                                },
                            )
                        )

                    if gnn_cand["value_score"] > best_fallback_node.value_score:
                        best_fallback_node = child_node

                    if (
                        cand_smi not in next_beam_by_smi
                        or cum_score > next_beam_by_smi[cand_smi].cumulative_score
                    ):
                        next_beam_by_smi[cand_smi] = child_node

        # Stereochemistry-first terminal selection across all hits found at the current depth:
        # Strictly prefers `exact_stereo_and_constitution` over `constitution_rebuild_match`,
        # then breaks ties by highest neural pathway score!
        if terminal_hits_at_depth:
            terminal_hits_at_depth.sort(key=lambda item: item[0], reverse=True)
            return terminal_hits_at_depth[0][1]

        next_beam_candidates = list(next_beam_by_smi.values())
        if not next_beam_candidates:
            break

        # Sort beam by Head 3 value_score + cumulative policy score
        next_beam_candidates.sort(
            key=lambda n: (n.value_score + 0.5 * (n.cumulative_score / max(1, n.depth))),
            reverse=True,
        )
        beam = next_beam_candidates[:beam_width]

    # If no exact terminus was reached within max_depth, return diagnostics on best peeled candidate
    _, _, rows_fb, meta_fb = check_deterministic_pks_terminus(best_fallback_node.smiles)
    return {
        "success": False,
        "match_kind": "unsolved_within_budget",
        "target_smiles": canon_target,
        "peeled_core_smiles": best_fallback_node.smiles,
        "peeling_depth": best_fallback_node.depth,
        "pathway": best_fallback_node.pathway,
        "initial_gnn": gnn_0,
        "final_gnn": best_fallback_node.gnn_eval,
        "module_rows": list(rows_fb),
        "terminal_metadata": meta_fb,
        "explored_nodes": explored_count,
    }


# =============================================================================
# Visualization of Reverse-Peeling Pathway & Deterministic Stage Readout
# =============================================================================

def _format_wrapped_modules(module_rows: List[Dict[str, Any]], starter: Any, per_line: int = 4) -> str:
    """Wrap long module sequences across multiple lines so RDKit never shrinks the legend font."""
    tokens = [f"M0:{starter}"] + [
        f"M{r['module_num']}:{r['at_annotation']}({r['kr_subtype'] or r['beta_state'] or 'DH'})"
        for r in module_rows
        if r["module_num"] > 0
    ]
    lines = []
    for i in range(0, len(tokens), per_line):
        lines.append(" -> ".join(tokens[i : i + per_line]))
    return "\n".join(lines)


def _short_enzyme_label(candidate_enzymes: str) -> str:
    """Extract a concise enzyme name (e.g. 'SpnF / LovB' or 'AbyZ / AbyA4') from full metadata."""
    if not candidate_enzymes:
        return "Enzyme"
    first_clause = candidate_enzymes.split("),")[0]
    if "(" in first_clause:
        gene_part = first_clause.split("(")[0].strip()
        desc_part = first_clause.split("(")[1].split(")")[0].strip()
        short_desc = " ".join(desc_part.split()[:3])
        return f"{gene_part} ({short_desc})"
    return candidate_enzymes[:38]


def _reaction_center_bonds(mol: Chem.Mol, rc_atoms: List[int]) -> List[int]:
    """Return bond indices where both endpoints (or 1 endpoint of a removed group) belong to `rc_atoms`."""
    rc_set = set(rc_atoms)
    bonds = []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u in rc_set and v in rc_set:
            bonds.append(b.GetIdx())
    if not bonds:
        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if u in rc_set or v in rc_set:
                bonds.append(b.GetIdx())
    return bonds


def render_peeling_pathway_image(result: Dict[str, Any], output_png: str):
    """
    Draw the complete retrobiosynthetic reverse-peeling pathway with explicit
    substructure-to-enzyme targeting highlights:
      - For every reverse step i:
        * Shows the substrate `from_smiles` with the EXACT reaction-center atoms & bonds
          highlighted in color and labeled with the concrete enzyme (e.g. `AbyZ/AbyA4`,
          `AbyE`, `AbyU`, `SpnF/LovB`, `OzmP`, `AviG2`, `PFxxxx`)
      - Followed by the Peeled Nascent PKS Aglycone M_0 (with TE/R cleavage bond highlighted)
      - Followed by the `ringbreaker.py` Opened Linear Acyl-ACP Chain (`S-C(=O)-...`) with
        C1, alpha, and beta module registers labeled.
    """
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    mols = []
    legends = []
    highlight_atoms = []
    highlight_bonds = []
    highlight_atom_colors = []
    highlight_bond_colors = []

    STEP_PALETTE = [
        (0.96, 0.36, 0.36),  # Step 1 Target Substructure: Coral Red
        (0.95, 0.58, 0.18),  # Step 2 Target Substructure: Amber Orange
        (0.68, 0.42, 0.92),  # Step 3 Target Substructure: Royal Purple
        (0.22, 0.68, 0.88),  # Step 4 Target Substructure: Cerulean Blue
    ]
    COLOR_CORE = (0.65, 0.90, 0.70)
    COLOR_C1 = (0.95, 0.35, 0.35)
    COLOR_ALPHA = (1.00, 0.72, 0.25)
    COLOR_BETA = (0.35, 0.72, 0.95)
    COLOR_TE_CUT = (0.92, 0.25, 0.65)

    pathway = result.get("pathway", [])

    # 1. If multi-step pathway exists, render each step's `from_smiles` with the exact
    #    substructure targeted by that step's enzyme highlighted and annotated!
    if pathway:
        for idx_s, step in enumerate(pathway):
            sub_mol = Chem.MolFromSmiles(step.from_smiles)
            if sub_mol is None:
                continue
            AllChem.Compute2DCoords(sub_mol)
            step_col = STEP_PALETTE[idx_s % len(STEP_PALETTE)]

            rc_atoms = [a for a in step.reaction_center_atoms if 0 <= a < sub_mol.GetNumAtoms()]
            rc_bonds = _reaction_center_bonds(sub_mol, rc_atoms)

            h_a = list(rc_atoms)
            h_a_col = {a: step_col for a in rc_atoms}
            h_b = list(rc_bonds)
            h_b_col = {b: step_col for b in rc_bonds}

            # Also lightly shade the remaining predicted PKS core on Step 1
            if idx_s == 0:
                p_core = result["initial_gnn"]["p_core_node"]
                for a_i in range(sub_mol.GetNumAtoms()):
                    if a_i not in h_a_col and a_i < len(p_core) and p_core[a_i] >= 0.55:
                        h_a.append(a_i)
                        h_a_col[a_i] = COLOR_CORE

            # Label the primary reaction-center anchor atom directly on the 2D structure with the enzyme gene name!
            enz_short = _short_enzyme_label(step.candidate_enzymes)
            gene_tag = enz_short.split("(")[0].strip()
            if rc_atoms:
                anchor_a = rc_atoms[0]
                sub_mol.GetAtomWithIdx(anchor_a).SetProp("atomNote", f"Target: {gene_tag}")

            pfam_short = step.pfam_domains.split("(")[0].strip() if step.pfam_domains else step.ec_class
            aa_tag = f" | NRPS Monomer: {step.matched_mxn_monomer} (mxn.yml)" if step.matched_mxn_monomer else ""
            mols.append(sub_mol)
            highlight_atoms.append(h_a)
            highlight_bonds.append(h_b)
            highlight_atom_colors.append(h_a_col)
            highlight_bond_colors.append(h_b_col)
            legends.append(
                f"Step {step.step_num} Enzyme Target: {enz_short}{aa_tag}\n"
                f"Op: {step.rule_name} [{pfam_short} | {step.ec_class}]"
            )
    else:
        target_mol = Chem.MolFromSmiles(result["target_smiles"])
        if target_mol is not None:
            AllChem.Compute2DCoords(target_mol)
            mols.append(target_mol)
            highlight_atoms.append(list(range(target_mol.GetNumAtoms())))
            highlight_bonds.append([])
            highlight_atom_colors.append({i: COLOR_CORE for i in range(target_mol.GetNumAtoms())})
            highlight_bond_colors.append({})
            legends.append(f"Input Already Nascent PKS Core (V={result['initial_gnn']['value_score']:.2f})")

    # 2. Peeled Nascent PKS Core M_0 (with TE/R macrolactone cleavage site + C1..Cn projected)
    core_mol = Chem.MolFromSmiles(result["peeled_core_smiles"])
    meta = result.get("terminal_metadata", {})
    bb_source_indices = meta.get("backbone_source_atom_indices") or []
    bb_linear_indices = meta.get("backbone_atom_indices") or []
    cut_indices = (meta.get("release") or {}).get("cut_source_atom_indices") or []
    wrapped_mods = _format_wrapped_modules(
        result.get("module_rows", []), meta.get("starter_unit"), per_line=4
    )

    if core_mol is not None and bb_source_indices:
        AllChem.Compute2DCoords(core_mol)
        h_a, h_c = [], {}
        h_b, h_b_c = [], {}
        for pos, a_idx in enumerate(bb_source_indices, start=1):
            if a_idx is not None and 0 <= int(a_idx) < core_mol.GetNumAtoms():
                idx_i = int(a_idx)
                role = "C1(TE)" if pos == 1 else (f"α{pos}" if pos % 2 == 0 else f"β{pos}")
                core_mol.GetAtomWithIdx(idx_i).SetProp("atomNote", role)
                h_a.append(idx_i)
                h_c[idx_i] = COLOR_C1 if pos == 1 else (COLOR_ALPHA if pos % 2 == 0 else COLOR_BETA)

        # Highlight the macrolactone/lactam TE ester bond that `ringbreaker.py` cleaves
        if len(cut_indices) >= 2:
            u_cut, v_cut = int(cut_indices[0]), int(cut_indices[1])
            for u_c in (u_cut, v_cut):
                if 0 <= u_c < core_mol.GetNumAtoms() and u_c not in h_c:
                    h_a.append(u_c)
                    h_c[u_c] = COLOR_TE_CUT
            b_cut = core_mol.GetBondBetweenAtoms(u_cut, v_cut)
            if b_cut is not None:
                h_b.append(b_cut.GetIdx())
                h_b_c[b_cut.GetIdx()] = COLOR_TE_CUT

        mols.append(core_mol)
        highlight_atoms.append(h_a)
        highlight_bonds.append(h_b)
        highlight_atom_colors.append(h_c)
        highlight_bond_colors.append(h_b_c)
        legends.append(
            f"Peeled Nascent PKS Core M_0 (TE: {meta.get('terminus_type')})\n"
            f"Pink Bond = TE Ring-Opening Cleavage (PF00975)"
        )

    # 3. Ring-Opened Linearized Acyl-ACP Chain from `deterministic_stage/ringbreaker.py`
    lin_smi = meta.get("linearized_smiles")
    if lin_smi and meta.get("terminus_type") in ("lactonization", "lactamization"):
        lin_mol = Chem.MolFromSmiles(lin_smi)
        if lin_mol is not None and bb_linear_indices:
            for a in lin_mol.GetAtoms():
                a.SetAtomMapNum(0)
            AllChem.Compute2DCoords(lin_mol)
            h_a_lin, h_c_lin = [], {}
            for pos, a_idx in enumerate(bb_linear_indices, start=1):
                if a_idx is not None and 0 <= int(a_idx) < lin_mol.GetNumAtoms():
                    idx_i = int(a_idx)
                    role = "C1" if pos == 1 else (f"α{pos}" if pos % 2 == 0 else f"β{pos}")
                    lin_mol.GetAtomWithIdx(idx_i).SetProp("atomNote", role)
                    h_a_lin.append(idx_i)
                    h_c_lin[idx_i] = COLOR_C1 if pos == 1 else (COLOR_ALPHA if pos % 2 == 0 else COLOR_BETA)
            mols.append(lin_mol)
            highlight_atoms.append(h_a_lin)
            highlight_bonds.append([])
            highlight_atom_colors.append(h_c_lin)
            highlight_bond_colors.append({})
            legends.append(
                f"ringbreaker.py Linear Acyl-ACP ({meta.get('extension_modules')} Modules)\n"
                f"{wrapped_mods}"
            )
    elif core_mol is not None and legends:
        legends[-1] = (
            f"Nascent PKS Core M_0 ({meta.get('extension_modules')} Modules, {meta.get('terminus_type')})\n"
            f"{wrapped_mods}"
        )

    if mols:
        n_panels = len(mols)
        ncols = min(2, n_panels)
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(10.5 * ncols, 8.2 * nrows),
            dpi=220,
            squeeze=False,
        )
        fig.patch.set_facecolor("white")

        for idx_p in range(nrows * ncols):
            r_i, c_i = divmod(idx_p, ncols)
            ax = axes[r_i][c_i]
            ax.axis("off")
            if idx_p >= n_panels:
                continue

            mol_p = mols[idx_p]
            drawer = rdMolDraw2D.MolDraw2DCairo(1100, 720)
            dopts = drawer.drawOptions()
            dopts.bondLineWidth = 3.0
            dopts.annotationFontScale = 0.95
            dopts.padding = 0.06
            dopts.multipleBondOffset = 0.16

            rdMolDraw2D.PrepareAndDrawMolecule(
                drawer,
                mol_p,
                highlightAtoms=highlight_atoms[idx_p],
                highlightBonds=highlight_bonds[idx_p],
                highlightAtomColors=highlight_atom_colors[idx_p],
                highlightBondColors=highlight_bond_colors[idx_p],
            )
            drawer.FinishDrawing()
            pil_img = Image.open(io.BytesIO(drawer.GetDrawingText()))
            ax.imshow(pil_img)

            # Render crisp, large, vector-font matplotlib title/caption above each panel
            caption_lines = legends[idx_p].split("\n")
            header_line = caption_lines[0]
            body_lines = "\n".join(caption_lines[1:])
            ax.set_title(
                f"{header_line}\n{body_lines}",
                fontsize=15.5,
                fontweight="bold",
                family="sans-serif",
                color="#141824",
                pad=10,
                linespacing=1.38,
                bbox=dict(
                    boxstyle="round,pad=0.45",
                    facecolor="#F4F6FA",
                    edgecolor="#C8D0E0",
                    linewidth=1.4,
                ),
            )

        plt.tight_layout(pad=2.2, h_pad=2.8, w_pad=2.0)
        fig.savefig(output_png, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved high-resolution reverse-peeling pathway visualization to: {output_png}")


# =============================================================================
# CLI & Batch Benchmark Evaluation on Held-Out Test Trajectories
# =============================================================================

def _infer_nrps_modules_from_pathway(
    pathway: List[PeelingStepRecord], start_module_num: int
) -> List[Dict[str, Any]]:
    """
    Reconstruct deterministic NRPS module rows (`M_{n+1}`, ...) from the reverse-peeled
    NRPS operations in `pathway` (ordered in forward biosynthetic order: inner-to-outer).
    """
    nrps_rows: List[Dict[str, Any]] = []
    # Check if an earlier reverse step was a heterocycle (thiazole/oxazole) or Dieckmann tetramate/pyridone
    has_thiazole = any(s.rule_id == 21 for s in pathway)
    has_oxazole = any(s.rule_id == 25 for s in pathway)
    has_tetramate = any(s.rule_id == 22 for s in pathway)
    has_pyridone = any(s.rule_id == 27 for s in pathway)
    has_nmt = any(s.rule_id == 23 for s in pathway)

    # Traverse pathway in reverse (which is forward biosynthetic order!)
    cur_mod = start_module_num + 1
    for step in reversed(pathway):
        if not step.matched_mxn_monomer and step.rule_id not in (20, 24):
            continue
        raw_aa = step.matched_mxn_monomer or ("pipecolic acid" if step.rule_id == 24 else "glycine")
        aa_short = (
            "pip" if "pipecol" in raw_aa.lower()
            else ("ala" if "alanin" in raw_aa.lower()
                  else ("cys" if has_thiazole
                        else ("ser" if has_oxazole
                              else ("3cl-tyr" if "3-chloro" in raw_aa.lower()
                                    else raw_aa.split()[0]))))
        )

        # Deterministic Chemical-to-NRPS-Domain Mapping:
        if has_thiazole or has_oxazole:
            c_dom = "Cy"
            tail_doms = ["Ox", "PCP"]
        else:
            c_dom = "C(LCL)"
            tail_doms = ["PCP"]

        if has_nmt:
            tail_doms.insert(0, "nMT")

        if "alaninol" in raw_aa.lower():
            tail_doms.append("R")
        elif has_tetramate or has_pyridone:
            tail_doms.append("R0(Dieckmann)")

        dom_str = f"{c_dom} - A({aa_short}) - " + " - ".join(tail_doms)
        nrps_rows.append(
            {
                "module": f"NRPS module {cur_mod}",
                "module_num": cur_mod,
                "at_annotation": f"A({aa_short})",
                "alpha_carbon": "N-Cα",
                "alpha_atom_index": None,
                "beta_carbon": "C=O",
                "beta_atom_index": None,
                "beta_state": "NRPS amide" if c_dom == "C(LCL)" else "heterocycle",
                "kr_subtype": "-",
                "domains": dom_str,
            }
        )
        cur_mod += 1

    return nrps_rows
    lines = []
    lines.append("=" * 84)
    lines.append("RETROBIOSYNTHETIC REVERSE-PEELING GNN + DETERMINISTIC STAGE REPORT")
    lines.append("=" * 84)
def format_deterministic_stage_report(result: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 84)
    lines.append("RETROBIOSYNTHETIC REVERSE-PEELING GNN + DETERMINISTIC STAGE REPORT")
    lines.append("=" * 84)
    nrps_monomers = [s.matched_mxn_monomer for s in result.get("pathway", []) if s.matched_mxn_monomer]
    scaffold_type = (
        f"HYBRID PKS-NRPS (Matched mxn.yml NRPS Monomers: {', '.join(nrps_monomers)})"
        if nrps_monomers
        else "PURE MODULAR TYPE I PKS"
    )
    lines.append(f"Target SMILES:        {result['target_smiles']}")
    lines.append(f"Biosynthetic Class:   {scaffold_type}")
    lines.append(f"Initial GNN Value:    V(G) = {result['initial_gnn']['value_score']:.4f} "
                 f"(Core Prob = {result['initial_gnn']['core_cls_prob']:.4f})")
    lines.append(f"Search Status:        {'SUCCESS' if result['success'] else 'UNSOLVED'} ({result['match_kind']})")
    lines.append(f"Peeling Depth:        {result['peeling_depth']} step(s) | Explored Nodes: {result['explored_nodes']}")
    lines.append(f"Peeled PKS Core M_0:  {result['peeled_core_smiles']}")

    if result["pathway"]:
        lines.append("\nReverse-Peeling Pathway & Candidate Tailoring Enzymes (Target -> Nascent PKS Aglycone):")
        for s in result["pathway"]:
            aa_line = f"       * Matched NRPS Monomer: {s.matched_mxn_monomer} (from RetroMol mxn.yml -> Hybrid PKS-NRPS)\n" if s.matched_mxn_monomer else ""
            lines.append(
                f"  Step {s.step_num}: {s.from_smiles}\n"
                f"       --[Reverse Rule: {s.rule_name} ({s.enzyme_family} | {s.ec_class} | Phase {s.phase} | "
                f"P_rule={s.rule_prob:.3f}, SiteScore={s.site_score:.3f})]-->\n"
                f"{aa_line}"
                f"       * Candidate Enzymes: {s.candidate_enzymes}\n"
                f"       * UniProt / MITE IDs: {s.uniprot_ids}\n"
                f"       * Pfam / BGC HMMs:    {s.pfam_domains}\n"
                f"       * Cofactors:          {s.cofactors}\n"
                f"       => Peeled Intermediate: {s.to_smiles}  (V={s.value_score:.3f})"
            )

    meta = result.get("terminal_metadata", {})
    rows = list(result.get("module_rows", []))
    if rows:
        nrps_extra_rows = _infer_nrps_modules_from_pathway(
            result.get("pathway", []), int(rows[-1]["module_num"])
        )
        if nrps_extra_rows:
            # Remove '- TE' from the last PKS module since it hands off to the NRPS module!
            rows[-1] = dict(rows[-1])
            rows[-1]["domains"] = rows[-1]["domains"].replace(" - TE", "")
            rows = rows + nrps_extra_rows

        lin_smi = meta.get("linearized_smiles", "")
        acp_thioester_smi = result["peeled_core_smiles"]
        if acp_thioester_smi.endswith("C(=O)O"):
            acp_thioester_smi = acp_thioester_smi[:-6] + "C(=O)[S]"
        lines.append("\nDeterministic Stage Decomposition (`ringbreaker` -> `carbon_backbone` -> `predict_extenders` -> `predict_reduction` + `mxn.yml` NRPS):")
        lines.append(f"  Release Mechanism:  {meta.get('terminus_type')}")
        lines.append(f"  Terminal ACP Chain: {acp_thioester_smi}")
        lines.append(f"  Linearized Chain:   {lin_smi}")
        lines.append(f"  Carbon Backbone:    {meta.get('backbone_len')} carbons {meta.get('backbone_source_atom_indices')}")
        lines.append(f"  PKS Ext. Modules:   {meta.get('extension_modules')} (+ {len(nrps_extra_rows)} NRPS Module(s))")
        lines.append(f"  Starter Unit (LM):  {meta.get('starter_unit')}")
        lines.append("\n  Module-by-Module Biosynthetic Readout (PKS + NRPS Assembly Line):")
        lines.append(f"  {'Mod':<5} | {'AT/A Sub.':<12} | {'Alpha/Beta':<10} | {'Beta State':<12} | {'KR Subtype':<10} | {'Domains'}")
        lines.append("  " + "-" * 76)
        for r in rows:
            ab = f"{r['alpha_carbon']}/{r['beta_carbon']}" if r["alpha_carbon"] else "-"
            lines.append(
                f"  M{r['module_num']:<4} | {str(r['at_annotation']):<12} | {ab:<10} | "
                f"{str(r['beta_state']):<12} | {str(r['kr_subtype']):<10} | {r['domains']}"
            )
    elif meta.get("failure"):
        lines.append(f"\nDeterministic Stage Failure Diagnostic: {json.dumps(meta['failure'], indent=2)}")

    lines.append("=" * 84)
    return "\n".join(lines)


def get_compute_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Run GNN-guided retrobiosynthetic reverse peeling and deterministic PKS module readout."
    )
    parser.add_argument("--smiles", type=str, default=None, help="Target molecule SMILES to peel and decompose")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to 3-head GNN checkpoint")
    parser.add_argument("--max_depth", type=int, default=4, help="Maximum reverse-peeling steps")
    parser.add_argument("--beam_width", type=int, default=8, help="Beam width for search")
    parser.add_argument("--plot_out", type=str, default=DEFAULT_PLOT_OUT, help="Output PNG path for pathway diagram")
    parser.add_argument(
        "--evaluate_test_trajectories",
        type=int,
        default=0,
        help="If > 0, evaluate multi-step peeling recovery rate on N held-out test states from supcon_test.parquet",
    )
    parser.add_argument(
        "--benchmark_clustercad_suite",
        action="store_true",
        help="Run the 12-BGC multi-family ClusterCAD benchmark suite and verify Definition-of-Done criteria",
    )
    parser.add_argument(
        "--benchmark_external_compact_nps",
        action="store_true",
        help="Run the 15-compound external compact natural products suite (NPAtlas / LOTUS / COCONUT, MW < 475 Da)",
    )
    args = parser.parse_args()

    device = get_compute_device()
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    model = load_peeling_model(args.checkpoint, device)
    rule_bank = build_enzymatic_rule_bank()

    if args.benchmark_external_compact_nps:
        from rdkit.Chem import Descriptors

        external_compact_nps = [
            ("Triacetic acid lactone (TAL)", "COCONUT:CNP0243003", "CC1=CC(O)=CC(=O)O1"),
            ("Germicidin A", "NPAtlas:NPA008432", "CCC(C)c1cc(O)c(CC)c(=O)o1"),
            ("6-Pentyl-alpha-pyrone", "NPAtlas:NPA000501", "CCCCCC1=CC=CC(=O)O1"),
            ("Massoia lactone", "LOTUS:LTS0148201", "CCCCC[C@H]1CC=CC(=O)O1"),
            ("Goniothalamin", "LOTUS:LTS0081240", "O=C1C=CC[C@H](/C=C/c2ccccc2)O1"),
            ("5,6-Dehydrokavain", "LOTUS:LTS0112945", "COC1=CC(=O)OC(/C=C/c2ccccc2)=C1"),
            ("Paraconic / Pre-lactone C10", "COCONUT:CNP0119402", "C[C@H](O)C[C@@H](O)CC=CC(=O)O"),
            ("10-Deoxymethynolide", "NPAtlas:NPA003214", "CC[C@H]1OC(=O)[C@H](C)[C@@H](O)[C@H](C)C[C@@H](C)C(=O)/C=C/[C@@H]1C"),
            ("YC-17 (10-Deoxymethymycin)", "NPAtlas:NPA011482", "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@H](C)C[C@@H](C)C(=O)/C=C/[C@@H]1C"),
            ("Neomethymycin", "NPAtlas:NPA004192", "CC([C@H]1OC(=O)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@H](C)C[C@@H](C)C(=O)/C=C/[C@@H]1C)O"),
            ("Epoxy-YC-17", "NPAtlas:NPA019402", "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@H](C)C[C@@H](C)C(=O)C2OC2[C@@H]1C"),
            ("Myxobacterial Thiazole-PK", "NPAtlas:NPA014892", "C[C@H](O)CC(=O)[C@@H](C)[C@H](O)[C@H](C)C(=O)Cc1nc(C(=O)O)cs1"),
            ("Oxazolomycin-Core Fragment", "COCONUT:CNP0310942", "CO[C@H]([C@H](C)C(=O)Cc1nc(C(=O)O)co1)[C@H](C)C(=O)C[C@H](C)O"),
            ("Compact Pipecolate-PK Ester", "NPAtlas:NPA022914", "C[C@@H]1C[C@H](O)[C@@H](C)/C=C/C=C(\\C)C(=O)N2CCCC[C@H]2C(=O)O1"),
            ("Compact Gly-Tetramate-PK", "NPAtlas:NPA018412", "C/C=C/C=C/[C@H](O)CC(=O)C1=C(O)NCC1=O"),
        ]
        print("\n" + "=" * 120)
        print("EXTERNAL COMPACT NATURAL PRODUCTS BENCHMARK (NPAtlas / LOTUS / COCONUT | MW < 475 Da, Heavy Atoms <= 33)")
        print("=" * 120)
        print(f"{'Natural Product':<30} | {'DB Accession':<19} | {'MW':>6} | {'HA':>3} | {'Status':<29} | {'Depth':>5} | {'Modules & Starter'}")
        print("-" * 120)
        solved_ext = 0
        exact_ext = 0
        for name_np, db_id, smi_np in external_compact_nps:
            mol_np = Chem.MolFromSmiles(smi_np)
            mw_np = Descriptors.MolWt(mol_np)
            ha_np = mol_np.GetNumHeavyAtoms()
            res_np = run_reverse_peeling_search(model, smi_np, rule_bank, device, max_depth=4, beam_width=8)
            mods_np = res_np["terminal_metadata"].get("extension_modules", 0)
            starter_np = res_np["terminal_metadata"].get("starter_unit", "-")
            nrps_extra_np = _infer_nrps_modules_from_pathway(res_np.get("pathway", []), int(mods_np or 0))
            if res_np["success"]:
                solved_ext += 1
                if res_np["match_kind"] == "exact_stereo_and_constitution":
                    exact_ext += 1
                mod_str = f"LM:{starter_np} + {mods_np} PKS" + (f" + {len(nrps_extra_np)} NRPS" if nrps_extra_np else "")
            else:
                mod_str = "unsolved"
            print(
                f"{name_np:<30} | {db_id:<19} | {mw_np:>6.1f} | {ha_np:>3d} | "
                f"{res_np['match_kind']:<29} | {res_np['peeling_depth']:>5d} | {mod_str}"
            )
        print("-" * 120)
        print(
            f"External Compact Natural Products Solved: {solved_ext}/{len(external_compact_nps)} "
            f"({100*solved_ext/len(external_compact_nps):.1f}%) [Exact Stereo & Constitution: {exact_ext}/{len(external_compact_nps)}]"
        )
        print("=" * 120)

    if args.benchmark_clustercad_suite:
        import time as _time

        df_cc = pd.read_csv(PROJECT_ROOT / "raw_set_clustercad.csv")
        suite_bgcs = [
            ("BGC0001492", "Abyssomicin C", 6, 8),
            ("BGC0000094", "Methymycin / Pikromycin", 6, 8),
            ("BGC0000054", "Erythromycin A", 6, 8),
            ("BGC0000102", "Mycinamicin", 6, 8),
            ("BGC0000092", "Megalomicin", 6, 8),
            ("BGC0000115", "Nystatin A1", 6, 8),
            ("BGC0000108", "Natamycin / Pimaricin", 6, 8),
            ("BGC0000090", "Macbecin I", 6, 8),
            ("BGC0000074", "Herbimycin A", 6, 8),
            ("BGC0000082", "Kijanimicin", 12, 8),
            ("BGC0001012", "Meridamycin (Hybrid)", 6, 8),
            ("BGC0001394", "Phenalamide (Hybrid)", 6, 8),
        ]
        print("\n" + "=" * 100)
        print("CLUSTERCAD 12-BGC MULTI-FAMILY ZERO-HEURISTIC BENCHMARK SUITE")
        print("=" * 100)
        print(f"{'BGC ID':<12} | {'Natural Product':<25} | {'Status':<30} | {'Depth':>5} | {'Nodes':>5} | {'Modules':>7} | {'Time':>6}")
        print("-" * 100)
        solved_cc = 0
        exact_stereo_cc = 0
        for bgc_id, label_name, max_d, bw in suite_bgcs:
            grp = df_cc[df_cc["bgc_id"] == bgc_id]
            if len(grp) == 0:
                continue
            fp_smi = str(grp[grp["module_num"] == -1].iloc[0]["intermediate_smiles"])
            t0 = _time.time()
            res = run_reverse_peeling_search(
                model, fp_smi, rule_bank, device, max_depth=max_d, beam_width=bw, top_k_rules_per_node=14
            )
            dt = _time.time() - t0
            if res["success"]:
                solved_cc += 1
                if res["match_kind"] == "exact_stereo_and_constitution":
                    exact_stereo_cc += 1
            mods = res["terminal_metadata"].get("extension_modules")
            print(
                f"{bgc_id:<12} | {label_name:<25} | {res['match_kind']:<30} | "
                f"{res['peeling_depth']:>5d} | {res['explored_nodes']:>5d} | {str(mods):>7} | {dt:>5.1f}s"
            )
        print("-" * 100)
        print(
            f"ClusterCAD Suite Solved: {solved_cc}/{len(suite_bgcs)} ({100*solved_cc/len(suite_bgcs):.1f}%) "
            f"[Exact Stereo & Constitution: {exact_stereo_cc}/{len(suite_bgcs)}]"
        )
        print("=" * 100)

    if args.evaluate_test_trajectories > 0:
        test_path = PROJECT_ROOT / "data" / "test" / "supcon_test.parquet"
        df_test = pd.read_parquet(test_path)
        # Filter to multi-step tailored intermediates (step >= 1)
        df_tailored = df_test[df_test["step"] >= 1].reset_index(drop=True)
        if len(df_tailored) > args.evaluate_test_trajectories:
            stride = max(1, len(df_tailored) // args.evaluate_test_trajectories)
            df_tailored = df_tailored.iloc[::stride].head(args.evaluate_test_trajectories).reset_index(drop=True)

        print(f"\nEvaluating GNN-guided Reverse Peeling + Deterministic Stage on {len(df_tailored)} held-out tailored molecules...")
        by_step = {}
        solved_total = 0
        exact_root_match = 0
        sample_MultiStep_result = None

        for _, row in df_tailored.iterrows():
            smi = str(row["smiles"])
            true_step = int(row["step"])
            res = run_reverse_peeling_search(
                model, smi, rule_bank, device, max_depth=max(args.max_depth, true_step + 1), beam_width=args.beam_width
            )
            by_step.setdefault(true_step, {"total": 0, "solved": 0, "exact_root": 0})
            by_step[true_step]["total"] += 1
            if res["success"]:
                solved_total += 1
                by_step[true_step]["solved"] += 1
                if res["peeled_core_smiles"] == str(row["root_pks_smiles"]):
                    exact_root_match += 1
                    by_step[true_step]["exact_root"] += 1
                if true_step >= 2 and sample_MultiStep_result is None:
                    sample_MultiStep_result = res

        print("\n" + "=" * 72)
        print("HELD-OUT MULTI-STEP RETROBIOSYNTHETIC PEELING BENCHMARK RESULTS")
        print("=" * 72)
        print(f"Overall Solved to Valid Deterministic PKS Core: {solved_total}/{len(df_tailored)} ({100*solved_total/max(1,len(df_tailored)):.1f}%)")
        print(f"Exact Original Root Core M_0 Reconstructed:     {exact_root_match}/{len(df_tailored)} ({100*exact_root_match/max(1,len(df_tailored)):.1f}%)")
        for s_num, counts in sorted(by_step.items()):
            print(
                f"  - {s_num}-Step Tailored Molecules: {counts['solved']}/{counts['total']} "
                f"({100*counts['solved']/max(1,counts['total']):.1f}%) solved to valid PKS core "
                f"[{counts['exact_root']}/{counts['total']} exact M_0]"
            )
        print("=" * 72)

        if sample_MultiStep_result is not None and args.smiles is None:
            print("\nSample Multi-Step Test Trajectory Solved End-to-End:")
            print(format_deterministic_stage_report(sample_MultiStep_result))
            render_peeling_pathway_image(sample_MultiStep_result, args.plot_out)

    if args.smiles:
        res = run_reverse_peeling_search(
            model, args.smiles, rule_bank, device, max_depth=args.max_depth, beam_width=args.beam_width
        )
        print(format_deterministic_stage_report(res))
        render_peeling_pathway_image(res, args.plot_out)


if __name__ == "__main__":
    main()
