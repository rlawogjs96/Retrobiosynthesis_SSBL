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
    nf_t = torch.from_numpy(nf).to(device)
    batch_t = torch.zeros(real_n, dtype=torch.long, device=device)
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
# RetroMol `mxn.yml` NRPS Monomer Catalog & Stereochemical Rule Rewriting
# =============================================================================

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


_ETHER_NEXT_TO_HYDROXYL = Chem.MolFromSmarts("[O;H0;D2;+0]-[C]-[C]-[O;H1;D1;+0]")
_TETRA = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)


def _swap_ligand_stereo(mol: Chem.Mol, center: int, removed: int, added: int) -> Chem.Mol:
    """Backside SN2 substitution from `PKS_core/pks_training_workflow/execution_rules.py`."""
    neighbors = [a.GetIdx() for a in mol.GetAtomWithIdx(center).GetNeighbors()]
    rw = Chem.RWMol(mol)
    rw.RemoveBond(center, removed)
    rw.AddBond(center, added, Chem.BondType.SINGLE)
    if mol.GetAtomWithIdx(center).GetChiralTag() in _TETRA:
        if (len(neighbors) - 1 - neighbors.index(removed)) % 2 == 0:
            rw.GetAtomWithIdx(center).InvertChirality()
    for index, hcount in ((removed, 1), (added, 0)):
        atom = rw.GetAtomWithIdx(index)
        atom.SetNumExplicitHs(hcount)
        atom.SetNoImplicit(True)
    result = rw.GetMol()
    Chem.SanitizeMol(result)
    Chem.AssignStereochemistry(result, cleanIt=True, force=True)
    return result


def _match_polyether_typed_graph_operators(
    mol: Chem.Mol,
    rule_id: int,
    p_core_node: np.ndarray,
    p_exposed_node: np.ndarray,
    syncat_att_map: Optional[np.ndarray] = None,
) -> List[Tuple[str, float, List[int], str]]:
    """
    Execute the typed-graph polyether operators from `PKS_core/pks_training_workflow/execution_rules.py`:
      - Rule 19 (`OPENING` / `literature_RDChiral_Lsd19_etherification_5exo` & `6endo`):
        Inverts 5- or 6-membered cyclic ether adjacent to -OH back to the hydroxy-epoxide.
      - Rule 32 (`POLYETHER_RING` composite operator):
        Inverts 5- or 6-membered cyclic ether adjacent to -OH -> hydroxy-epoxide -> E/Z alkene + alcohol.
    Scored with the exact same `0.50 * site_exposed_score + 0.35 * site_decor_score + syncat_boost` formula.
    """
    if rule_id not in (19, 32):
        return []

    n_atoms = mol.GetNumAtoms()
    hits: List[Tuple[str, float, List[int], str]] = []
    for nuc, attacked, other, oxygen in mol.GetSubstructMatches(_ETHER_NEXT_TO_HYDROXYL):
        # Exclude ketals/hemiketals: neither the attacked carbon nor the other carbon
        # of the ring ether may already be bonded to two oxygens (which is a ketal/hemiketal,
        # handled by Rule 2 `hemiketal_spiroketal_cyclization`).
        if sum(1 for nb in mol.GetAtomWithIdx(attacked).GetNeighbors() if nb.GetAtomicNum() == 8) >= 2:
            continue
        other_side_c = [nb.GetIdx() for nb in mol.GetAtomWithIdx(nuc).GetNeighbors() if nb.GetIdx() != attacked]
        if any(sum(1 for nb in mol.GetAtomWithIdx(c_s).GetNeighbors() if nb.GetAtomicNum() == 8) >= 2 for c_s in other_side_c):
            continue

        edge = mol.GetBondBetweenAtoms(nuc, attacked)
        if edge is None or not any(edge.IsInRingSize(size) for size in (5, 6)):
            continue
        if mol.GetBondBetweenAtoms(oxygen, attacked) is not None:
            continue
        rw_check = Chem.RWMol(mol)
        rw_check.RemoveBond(nuc, attacked)
        path = Chem.GetShortestPath(rw_check.GetMol(), nuc, attacked)
        if len(path) not in (5, 6):
            continue
        try:
            epoxide_cand = _swap_ligand_stereo(mol, attacked, nuc, oxygen)
        except Exception:
            continue

        rc_indices = sorted({int(nuc), int(attacked), int(other), int(oxygen)})
        site_exposed_score = float(np.mean([p_exposed_node[i] for i in rc_indices if i < len(p_exposed_node)]))
        site_decor_score = float(np.mean([1.0 - p_core_node[i] for i in rc_indices if i < len(p_core_node)]))
        syncat_boost = 0.0
        if syncat_att_map is not None and rule_id < syncat_att_map.shape[0]:
            rule_att = syncat_att_map[rule_id]
            uniform_att = 1.0 / max(1, n_atoms)
            rc_att = float(np.mean([rule_att[i] for i in rc_indices if i < len(rule_att)]))
            syncat_boost = min(0.25, max(0.0, (rc_att / max(1e-6, uniform_att)) * 0.05))
        site_score = min(1.0, 0.50 * site_exposed_score + 0.35 * site_decor_score + syncat_boost)

        if rule_id == 19:
            hits.append((Chem.MolToSmiles(epoxide_cand, isomericSmiles=True), site_score, rc_indices, ""))
        elif rule_id == 32:
            # Step 2 of `POLYETHER_RING`: remove epoxide oxygen -> E/Z alkene
            try:
                rw_alk = Chem.RWMol(epoxide_cand)
                rw_alk.RemoveAtom(oxygen)
                aa = attacked - int(attacked > oxygen)
                bb = other - int(other > oxygen)
                rw_alk.GetBondBetweenAtoms(aa, bb).SetBondType(Chem.BondType.DOUBLE)
                for idx_c in (aa, bb):
                    rw_alk.GetAtomWithIdx(idx_c).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                skeleton = rw_alk.GetMol()
                Chem.SanitizeMol(skeleton)
                bond = skeleton.GetBondBetweenAtoms(aa, bb)
                if bond.IsInRing():
                    continue
                refs = [
                    next((n.GetIdx() for n in skeleton.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j), None)
                    for i, j in ((aa, bb), (bb, aa))
                ]
                if None not in refs:
                    for geom in (Chem.BondStereo.STEREOTRANS, Chem.BondStereo.STEREOCIS):
                        cand_g = Chem.Mol(skeleton)
                        b_g = cand_g.GetBondBetweenAtoms(aa, bb)
                        r_pair = list(refs)
                        if b_g.GetBeginAtomIdx() != aa:
                            r_pair.reverse()
                        b_g.SetStereoAtoms(*r_pair)
                        b_g.SetStereo(geom)
                        Chem.SetDoubleBondNeighborDirections(cand_g)
                        Chem.AssignStereochemistry(cand_g, cleanIt=True, force=True)
                        hits.append((Chem.MolToSmiles(cand_g, isomericSmiles=True), site_score, rc_indices, ""))
                else:
                    hits.append((Chem.MolToSmiles(skeleton, isomericSmiles=True), site_score, rc_indices, ""))
            except Exception:
                continue

    return hits


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
    polyether_hits = _match_polyether_typed_graph_operators(
        mol, rule.rule_id, p_core_node, p_exposed_node, syncat_att_map=syncat_att_map
    )

    raw_prods = []
    any_match = bool(mxn_aa_hits) or bool(polyether_hits)
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

    if not any_match or (not raw_prods and not mxn_aa_hits and not polyether_hits):
        return []

    results: Dict[str, Tuple[float, List[int], str]] = {}
    for pks_smi, sc_aa, rc_aa, aa_name in mxn_aa_hits + polyether_hits:
        if pks_smi != smiles and (pks_smi not in results or sc_aa > results[pks_smi][0]):
            results[pks_smi] = (sc_aa, rc_aa, aa_name)
    n_atoms = len(p_core_node)

    for prod_tuple in raw_prods:
        if not prod_tuple:
            continue
        cand = Chem.Mol(prod_tuple[0])
        try:
            # RetroMol valence.py hydrogen adjustment for sp3 -> sp2 transitions (e.g. retro-carbocyclization)
            pt = Chem.GetPeriodicTable()
            for atom in cand.GetAtoms():
                if atom.GetIsAromatic() or atom.GetFormalCharge() != 0 or atom.GetAtomicNum() in (15, 16):
                    continue
                vb = int(sum(b.GetValenceContrib(atom) for b in atom.GetBonds()))
                dv = pt.GetDefaultValence(atom.GetAtomicNum())
                if dv - vb < atom.GetNumExplicitHs():
                    if dv - vb < 0:
                        raise ValueError("Invalid valence")
                    atom.SetNumExplicitHs(dv - vb)
                    if any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds()):
                        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            Chem.SanitizeMol(cand)
            Chem.AssignStereochemistry(cand, cleanIt=True, force=True)
        except Exception:
            continue

        frags = Chem.GetMolFrags(cand, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            if rule.is_bridge_forming:
                # Intramolecular ring-opening rules (Diels-Alder, oxa-Michael, spiroketal, polyether)
                # must produce a single connected molecule without losing fragments (RetroMol rules.py:109)
                continue
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
    Fast pre-filter: skips terminal decomposition if the molecule still carries
    attached 6-membered deoxysugar pyranosides (`_PYRANOSIDE_PATTERN`).
    Also handles aromatic 2-pyrone / 4-hydroxy-2-pyrone enol-lactone tautomerization.
    """
    mol_check = Chem.MolFromSmiles(smiles)
    if mol_check is None:
        return False, "invalid_smiles", (), {}
    if _PYRANOSIDE_PATTERN is not None and mol_check.HasSubstructMatch(_PYRANOSIDE_PATTERN):
        return False, "still_glycosylated", (), {}

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
    max_depth: int = 12,
    beam_width: int = 8,
    top_k_rules_per_node: int = 33,
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
            ranked_rule_ids = np.argsort(-rule_probs)[:top_k_rules_per_node].tolist()
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


def _find_step_broken_bonds_and_detached_monomer(
    from_smi: str, to_smi: str, rule_id: int, matched_aa: str = ""
) -> Tuple[List[int], str, Optional[str], Optional[str]]:
    """
    Identify the EXACT bond(s) in `from_smi` attacked by `rule_id` to form `to_smi`,
    their color role ('blue' for intermolecular ester/glycoside/amide/O-Me cleavage,
    'orange' for intramolecular ring-opening / carbocyclization / C-H oxidation),
    and any detached monomer SMILES + name.
    """
    mol = Chem.MolFromSmiles(from_smi)
    to_mol = Chem.MolFromSmiles(to_smi)
    if mol is None or to_mol is None:
        return [], "orange", None, None

    target_canon = Chem.MolToSmiles(to_mol, isomericSmiles=False)
    rules_dict = {r.rule_id: r for r in build_enzymatic_rule_bank()}
    rule = rules_dict.get(rule_id)

    # Known monomer names for common detached groups
    KNOWN_DETACHED_NAMES = {
        "CCC(C)C(=O)O": "2-methylbutanoic acid",
        "CC(=O)O": "acetic acid",
        "NCC(=O)O": "glycine",
        "OC(=O)C1CCCCN1": "L-pipecolic acid",
        "OC(=O)C1CCCN1": "L-proline",
    }

    if rule is not None:
        tagged = Chem.Mol(mol)
        for a in tagged.GetAtoms():
            a.SetIsotope(a.GetIdx() + 1)

        smarts_list = [rule.reverse_smarts] + list(getattr(rule, "extra_reverse_smarts", []))
        pt = Chem.GetPeriodicTable()
        for sm in smarts_list:
            try:
                rxn = AllChem.ReactionFromSmarts(sm)
                prods = rxn.RunReactants((tagged,))
            except Exception:
                continue
            for tup in prods:
                if not tup:
                    continue
                cand = Chem.Mol(tup[0])
                try:
                    for atom in cand.GetAtoms():
                        if atom.GetIsAromatic() or atom.GetFormalCharge() != 0 or atom.GetAtomicNum() in (15, 16):
                            continue
                        vb = int(sum(b.GetValenceContrib(atom) for b in atom.GetBonds()))
                        dv = pt.GetDefaultValence(atom.GetAtomicNum())
                        if dv - vb < atom.GetNumExplicitHs():
                            if dv - vb < 0:
                                raise ValueError()
                            atom.SetNumExplicitHs(dv - vb)
                            if any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds()):
                                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                    Chem.SanitizeMol(cand)
                except Exception:
                    continue
                frags = Chem.GetMolFrags(cand, asMols=True, sanitizeFrags=True)
                if len(frags) > 1 and rule.is_bridge_forming:
                    continue
                main_f = max(frags, key=lambda m: m.GetNumAtoms()) if len(frags) > 1 else cand
                clean_f = Chem.Mol(main_f)
                for a in clean_f.GetAtoms():
                    a.SetAtomMapNum(0)
                    a.SetIsotope(0)
                if Chem.MolToSmiles(clean_f, isomericSmiles=False) == target_canon:
                    surv = {a.GetIsotope() - 1 for a in main_f.GetAtoms() if a.GetIsotope() > 0}
                    rem = set(range(mol.GetNumAtoms())) - surv
                    if rem:
                        broken_bonds = [
                            b.GetIdx()
                            for b in mol.GetBonds()
                            if (b.GetBeginAtomIdx() in surv and b.GetEndAtomIdx() in rem)
                            or (b.GetBeginAtomIdx() in rem and b.GetEndAtomIdx() in surv)
                        ]
                        color_role = "blue" if rule_id in (8, 11, 13, 15, 20, 24, 26) else "orange"
                        detached_smi, detached_name = None, None
                        if len(rem) >= 2:
                            rw = Chem.RWMol(mol)
                            for idx in sorted(surv, reverse=True):
                                rw.RemoveAtom(idx)
                            det_m = rw.GetMol()
                            for a in det_m.GetAtoms():
                                if a.GetAtomicNum() == 6 and a.GetDegree() == 2 and any(
                                    nb.GetAtomicNum() == 8
                                    and det_m.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx()).GetBondType()
                                    == Chem.BondType.DOUBLE
                                    for nb in a.GetNeighbors()
                                ):
                                    rw2 = Chem.RWMol(det_m)
                                    o_idx = rw2.AddAtom(Chem.Atom(8))
                                    rw2.AddBond(a.GetIdx(), o_idx, Chem.BondType.SINGLE)
                                    det_m = rw2.GetMol()
                                    break
                            try:
                                Chem.SanitizeMol(det_m)
                                detached_smi = Chem.MolToSmiles(det_m, isomericSmiles=True)
                                ns_det = Chem.MolToSmiles(det_m, isomericSmiles=False)
                                detached_name = (
                                    matched_aa
                                    or KNOWN_DETACHED_NAMES.get(ns_det)
                                    or ("deoxysugar" if rule_id == 13 else Chem.rdMolDescriptors.CalcMolFormula(det_m))
                                )
                            except Exception:
                                pass
                        return broken_bonds, color_role, detached_smi, detached_name
                    else:
                        cand_pairs = set()
                        for b in main_f.GetBonds():
                            u, v = b.GetBeginAtom().GetIsotope() - 1, b.GetEndAtom().GetIsotope() - 1
                            cand_pairs.add((min(u, v), max(u, v)))
                        broken_bonds = []
                        for b in mol.GetBonds():
                            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                            if (min(u, v), max(u, v)) not in cand_pairs:
                                broken_bonds.append(b.GetIdx())
                        return broken_bonds, "orange", None, None

    # Fallback for typed-graph polyether or mxn NRPS cleavage
    if matched_aa:
        aa_cat = get_mxn_nrps_amino_acid_catalog()
        det_smi = aa_cat.get(matched_aa, "NCC(=O)O")
        # Find amide C(=O)-N bond in mol
        amide_bonds = []
        for b in mol.GetBonds():
            u, v = b.GetBeginAtom(), b.GetEndAtom()
            if {u.GetAtomicNum(), v.GetAtomicNum()} == {6, 7} and b.GetBondType() == Chem.BondType.SINGLE:
                c_at = u if u.GetAtomicNum() == 6 else v
                if any(
                    nb.GetAtomicNum() == 8
                    and mol.GetBondBetweenAtoms(c_at.GetIdx(), nb.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                    for nb in c_at.GetNeighbors()
                ):
                    amide_bonds.append(b.GetIdx())
        return amide_bonds[:1], "blue", det_smi, matched_aa

    # Fallback for polyether ring C-O bond
    ring_co_bonds = [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.IsInRing() and {b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum()} == {6, 8}
    ]
    return ring_co_bonds[:1], "orange", None, None


def _draw_bond_highlighted_mol(
    smiles: str,
    broken_bonds: List[int],
    color_role: str = "orange",
    size: Tuple[int, int] = (760, 480),
) -> Image.Image:
    """Render an RDKit molecule on a pure white background with ONLY `broken_bonds` highlighted."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Image.new("RGB", size, "white")
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
        a.SetIsotope(0)
    try:
        Chem.rdDepictor.SetPreferCoordGen(True)
        Chem.rdDepictor.Compute2DCoords(mol, canonOrient=True)
    except Exception:
        rdMolDraw2D.PrepareMolForDrawing(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    dopts = drawer.drawOptions()
    dopts.bondLineWidth = 2.1
    dopts.padding = 0.12
    dopts.continuousHighlight = False
    dopts.fillHighlights = True
    dopts.highlightBondWidthMultiplier = 6

    col = (0.22, 0.52, 0.86) if color_role == "blue" else (0.93, 0.58, 0.14)
    valid_bonds = [b for b in broken_bonds if 0 <= b < mol.GetNumBonds()]
    b_cols = {b: col for b in valid_bonds}
    # Only highlight terminal single-atom leaving groups (like -OH in Rule 3 or -CH3 in Rule 9) if bond has degree-1 endpoint
    a_list = []
    a_cols = {}
    a_rads = {}
    for b_idx in valid_bonds:
        b = mol.GetBondWithIdx(b_idx)
        for end_a in (b.GetBeginAtom(), b.GetEndAtom()):
            if end_a.GetDegree() == 1:
                a_list.append(end_a.GetIdx())
                a_cols[end_a.GetIdx()] = col
                a_rads[end_a.GetIdx()] = 0.22

    drawer.DrawMolecule(
        mol,
        highlightAtoms=a_list,
        highlightBonds=valid_bonds,
        highlightAtomColors=a_cols,
        highlightBondColors=b_cols,
        highlightAtomRadii=a_rads,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")



_RULE_DISPLAY_OPERATION: Dict[int, Tuple[str, str, str]] = {
    0: ("SpnF / LovB", "reverse Carbocyclization", "retro-[4+2] Diels-Alder"),
    1: ("SalBIII / AmbDH3", "Pyran ring opening", "reverse oxa-Michael"),
    2: ("AveC / RevG", "Ketal ring opening", "reverse hemiketal/spiroketal"),
    3: ("EryF / PikC", "C–H deoxygenation", "reverse P450 hydroxylation"),
    4: ("AbyE / EpoK", "Epoxide deoxygenation", "reverse P450 epoxidation"),
    5: ("MtmOIV / PtlE", "Lactone excision", "reverse Baeyer-Villiger"),
    6: ("CmlS / RebH", "Dechlorination", "reverse FADH2 halogenation"),
    7: ("Bmp2 / PltA", "Debromination", "reverse FADH2 bromination"),
    8: ("EryG / TylF", "O-demethylation", "reverse SAM O-MT"),
    9: ("NovO / CouO", "C-demethylation", "reverse SAM C-MT"),
    10: ("EryBIV / TylD", "Ketone reduction", "reverse alcohol oxidation"),
    11: ("LovD / MdmB", "Ester cleavage", "reverse O-acyltransferase"),
    12: ("MonBI / EpoH", "Diol → Epoxide", "reverse epoxide hydrolase"),
    13: ("DesVII / EryBV", "Glycosidic cleavage", "reverse O-glycosyltransferase"),
    14: ("NphB / CloQ", "Deprenylation", "reverse prenyltransferase"),
    15: ("Asm21 / NovN", "Carbamoyl cleavage", "reverse O-carbamoyltransferase"),
    16: ("AbyA1–A3 / ChlM", "Tetronate cleavage", "reverse glyceryl offloading"),
    17: ("AbyU / PyrI4", "reverse Spiro-cyclization", "retro-[4+2] spirotetronate"),
    18: ("AbyZ / AbyA4", "Ether bridge opening", "reverse epoxide-enol opening"),
    19: ("Lsd19 / MonBI", "Cyclic ether opening", "reverse epoxide-zipper"),
    20: ("TmcA / RapP", "Amide bond cleavage", "reverse NRPS condensation"),
    21: ("EpoB / BlmIII", "Thiazole ring opening", "reverse NRPS heterocyclization"),
    22: ("TmcA / IkaA", "Tetramate ring opening", "reverse Dieckmann cyclization"),
    23: ("RapN / SnbDE", "N-demethylation", "reverse SAM N-MT"),
    24: ("RapP / FkbP", "Amide bond cleavage", "reverse NRPS pipecolate"),
    25: ("OzmP / VirG", "Oxazole ring opening", "reverse NRPS heterocyclization"),
    26: ("CurM /StfA", "Desulfonation", "reverse PAPS sulfotransferase"),
    27: ("KirB / EfTu", "Pyridone ring opening", "reverse Dieckmann cyclization"),
    28: ("PldB / BorI", "Acetal cleavage", "reverse dioxolane formation"),
    29: ("BorJ / CmmG", "Nitrile reversal", "reverse oxidative nitrile"),
    30: ("AmphN / NysN", "Carboxyl reduction", "reverse P450 methyl oxidation"),
    31: ("GdmM / Rif19", "Quinone reduction", "reverse AHBA monooxygenase"),
    32: ("MonBI / Lsd19", "Cyclic ether opening", "reverse polyether cyclization"),
}


def render_reaction_row_on_axes(
    fig: plt.Figure,
    y_top: float,
    row_height: float,
    steps_spec: List[Dict[str, Any]],
    final_smi: str,
    final_title: str,
    final_subtitle: str,
    badge_color: str = "#28B446",
    row_header: Optional[str] = None,
):
    """
    Draw a single horizontal reaction-scheme row in the exact visual style of
    `Pravastatin C23H36O7 -> C18H28O6 -> PKS monomer`:
      - Clean white canvas, bond-only highlights (blue or orange) on each reactant molecule
      - Horizontal black reaction arrow with numbered circle (`1`, `2`, ...) in `badge_color`,
        bold serif enzyme label above the arrow, 2-line reaction label below the arrow
      - Curved downward arrow + 2D chemical structure for any `detached_smi`
      - Bold serif compound name + molecular formula below each main molecule
    """
    n_steps = len(steps_spec)
    n_mols = n_steps + 1
    total_units = n_mols * 1.15 + n_steps * 0.72
    left_margin = 0.02
    usable_w = 0.96
    unit_w = usable_w / total_units
    mol_w = 1.15 * unit_w
    arr_w = 0.72 * unit_w

    if row_header:
        fig.text(
            left_margin,
            y_top + 0.015,
            row_header,
            ha="left",
            va="bottom",
            fontsize=12.5,
            fontweight="bold",
            family="serif",
            color="#141824",
        )

    main_mol_h = row_height * 0.50
    main_y_bottom = y_top - main_mol_h - 0.02

    x_cursor = left_margin
    for idx in range(n_mols):
        if idx < n_steps:
            st = steps_spec[idx]
            smi = st["from_smi"]
            b_bonds = st.get("broken_bonds", [])
            c_role = st.get("color_role", "orange")
            m_obj = Chem.MolFromSmiles(smi)
            formula = Chem.rdMolDescriptors.CalcMolFormula(m_obj) if m_obj else ""
            mol_title = st.get("mol_title") or (f"{st.get('mol_name', '')}\n{formula}".strip() if st.get("mol_name") else formula)
            mol_sub = st.get("mol_subtitle", "")
        else:
            smi = final_smi
            b_bonds = []
            c_role = "orange"
            mol_title = final_title
            mol_sub = final_subtitle

        ax_m = fig.add_axes([x_cursor, main_y_bottom, mol_w, main_mol_h])
        ax_m.axis("off")
        img_m = _draw_bond_highlighted_mol(smi, b_bonds, c_role, size=(820, 500))
        ax_m.imshow(img_m)

        # Compound name & formula in bold serif directly underneath the molecule
        fig.text(
            x_cursor + mol_w * 0.5,
            main_y_bottom - 0.005,
            mol_title,
            ha="center",
            va="top",
            fontsize=11.5,
            fontweight="bold",
            family="serif",
            color="#111111",
            linespacing=1.18,
        )
        if mol_sub:
            n_title_lines = mol_title.count("\n") + 1
            fig.text(
                x_cursor + mol_w * 0.5,
                main_y_bottom - 0.005 - 0.032 * n_title_lines,
                mol_sub,
                ha="center",
                va="top",
                fontsize=9.2,
                family="serif",
                color="#222222",
                linespacing=1.22,
            )

        x_cursor += mol_w

        # Draw reaction arrow column if not the last molecule
        if idx < n_steps:
            st = steps_spec[idx]
            ax_a = fig.add_axes([x_cursor, y_top - row_height, arr_w, row_height])
            ax_a.set_xlim(0, 1)
            ax_a.set_ylim(0, 1)
            ax_a.axis("off")

            y_arrow = 0.73
            # Numbered circle badge (`1`, `2`, ...)
            ax_a.scatter(
                [0.5],
                [y_arrow + 0.16],
                s=520,
                color=badge_color,
                zorder=5,
                clip_on=False,
            )
            ax_a.text(
                0.5,
                y_arrow + 0.16,
                str(idx + 1),
                ha="center",
                va="center",
                fontsize=12.0,
                fontweight="bold",
                family="sans-serif",
                color="white",
                zorder=6,
            )

            # Bold serif enzyme name above arrow
            ax_a.text(
                0.5,
                y_arrow + 0.035,
                st.get("enzyme_label", "Enzyme"),
                ha="center",
                va="bottom",
                fontsize=11.0,
                fontweight="bold",
                family="serif",
                color="#111111",
            )

            # Horizontal black reaction arrow
            ax_a.annotate(
                "",
                xy=(0.92, y_arrow),
                xytext=(0.08, y_arrow),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8, mutation_scale=13),
            )

            # 2-line operation label below arrow
            op_line1 = st.get("op_line1", "")
            op_line2 = st.get("op_line2", "")
            ax_a.text(
                0.5,
                y_arrow - 0.04,
                f"{op_line1}\n\n{op_line2}" if op_line2 else op_line1,
                ha="center",
                va="top",
                fontsize=9.8,
                family="serif",
                color="#111111",
                linespacing=0.95,
            )

            # If a monomer is detached in this step, draw curved branch arrow + detached monomer 2D structure!
            det_smi = st.get("detached_smi")
            if det_smi:
                ax_a.annotate(
                    "",
                    xy=(0.55, 0.36),
                    xytext=(0.45, y_arrow - 0.15),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        connectionstyle="arc3,rad=-0.28",
                        color="black",
                        lw=1.6,
                        mutation_scale=12,
                    ),
                )
                det_w = arr_w * 1.05
                det_h = row_height * 0.24
                det_x = x_cursor - arr_w * 0.02
                det_y = (y_top - row_height) + row_height * 0.09
                ax_det = fig.add_axes([det_x, det_y, det_w, det_h])
                ax_det.axis("off")
                img_det = _draw_bond_highlighted_mol(det_smi, [], "orange", size=(420, 260))
                ax_det.imshow(img_det)
                det_name = st.get("detached_name", "monomer")
                fig.text(
                    det_x + det_w * 0.5,
                    det_y - 0.002,
                    f"detached\nmonomer\n{det_name}",
                    ha="center",
                    va="top",
                    fontsize=9.2,
                    family="serif",
                    color="#111111",
                    linespacing=1.10,
                )

            x_cursor += arr_w


def build_steps_spec_from_engine_result(
    result: Dict[str, Any], target_name: str = "Target"
) -> Tuple[List[Dict[str, Any]], str, str, str]:
    """Convert `run_reverse_peeling_search` result into `steps_spec` for `render_reaction_row_on_axes`."""
    pathway = result.get("pathway", [])
    steps_spec: List[Dict[str, Any]] = []
    for i, step in enumerate(pathway):
        b_bonds, c_role, det_smi, det_name = _find_step_broken_bonds_and_detached_monomer(
            step.from_smiles, step.to_smiles, step.rule_id, step.matched_mxn_monomer
        )
        enz_lbl, op1, op2 = _RULE_DISPLAY_OPERATION.get(
            step.rule_id,
            (_short_enzyme_label(step.candidate_enzymes).split("(")[0].strip(), step.rule_name, f"reverse {step.enzyme_family}"),
        )
        m_from = Chem.MolFromSmiles(step.from_smiles)
        f_from = Chem.rdMolDescriptors.CalcMolFormula(m_from) if m_from else ""
        m_title = f"{target_name}\n{f_from}" if i == 0 else f_from
        steps_spec.append({
            "from_smi": step.from_smiles,
            "broken_bonds": b_bonds,
            "color_role": c_role,
            "mol_title": m_title,
            "mol_subtitle": "",
            "enzyme_label": enz_lbl,
            "op_line1": op1,
            "op_line2": op2,
            "detached_smi": det_smi,
            "detached_name": det_name,
        })

    core_smi = result["peeled_core_smiles"]
    m_core = Chem.MolFromSmiles(core_smi)
    f_core = Chem.rdMolDescriptors.CalcMolFormula(m_core) if m_core else ""
    meta = result.get("terminal_metadata", {})
    mod_rows = result.get("module_rows", [])
    gene_plan = search_inhouse_wetlab_genes(mod_rows, max_gene_kb=9.0, strictly_inhouse_strains=True)
    mod_tokens = [
        f"M{r['module_num']}:{r['at_annotation']}"
        + (f"({r['kr_subtype']})" if r.get("kr_subtype") and r.get("kr_subtype") != "U" else "")
        for r in mod_rows
        if int(r.get("module_num", -1)) >= 0
    ]
    mod_line = " → ".join(mod_tokens[:5]) + ("\n→ " + " → ".join(mod_tokens[5:]) if len(mod_tokens) > 5 else "")
    gene_line = ""
    if gene_plan.get("feasible"):
        g_toks = [
            f"[M{s['start_mod']}"
            + (f"..M{s['end_mod']}" if s["start_mod"] != s["end_mod"] else "")
            + f"]:{s['gene_id'].split('::')[-1]}({s['est_kb']}kb)"
            for s in gene_plan["best_assembly"]
        ]
        gene_line = "\nIn-House Genes: " + ", ".join(g_toks[:3]) + ("\n" + ", ".join(g_toks[3:]) if len(g_toks) > 3 else "")
    final_title = f"PKS monomer\n{f_core}"
    final_subtitle = f"Modular PKS ({meta.get('extension_modules', 0)} Ext. Mods):\n{mod_line}{gene_line}"
    return steps_spec, core_smi, final_title, final_subtitle


def _draw_aligned_acyl_acp(
    smiles: str,
    highlight_new_unit: bool = True,
    fixed_bond_px: int = 34,
    is_released_acid: bool = False,
) -> Image.Image:
    """
    Render a growing acyl-S-ACP intermediate (`...-C(=O)[S]`) or released free
    carboxylic acid (`...-C(=O)O`) with:
      - The carbon backbone oriented HORIZONTALLY (`tail` on the left `-x`, `C1(=O)S/OH` on the right `+x`),
      - `C1(=O)` carbonyl oxygen pointing UP (`+y`) and `S•` / `OH` on the right (`+x`),
      - Translucent green highlight on the newly added/modified C1–C2(α)–C3(β) unit (or terminal COOH upon release),
      - Fixed bond length (`fixed_bond_px`) and tight auto-cropped bounding box.
    """
    from PIL import ImageChops
    import math as _math

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Image.new("RGB", (200, 120), "white")
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
        a.SetIsotope(0)
    Chem.rdDepictor.SetPreferCoordGen(False)
    Chem.rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()

    s_idx, c1_idx, o1_idx, c2_idx = None, None, None, None
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 16 and a.GetDegree() == 1:
            s_idx = a.GetIdx()
            c1 = a.GetNeighbors()[0]
            c1_idx = c1.GetIdx()
            for nb in c1.GetNeighbors():
                if nb.GetIdx() == s_idx:
                    continue
                b = mol.GetBondBetweenAtoms(c1_idx, nb.GetIdx())
                if nb.GetAtomicNum() == 8 and b.GetBondType() == Chem.BondType.DOUBLE:
                    o1_idx = nb.GetIdx()
                elif nb.GetAtomicNum() == 6:
                    c2_idx = nb.GetIdx()
            break

    if c1_idx is None:
        # Match terminal carboxylic acid -C1(=O1)-OH
        cooh_matches = mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3:1](=[OX1:2])[OX2H1:3]"))
        if cooh_matches:
            c1_idx, o1_idx, s_idx = cooh_matches[-1]
            c1 = mol.GetAtomWithIdx(c1_idx)
            for nb in c1.GetNeighbors():
                if nb.GetIdx() not in (o1_idx, s_idx) and nb.GetAtomicNum() == 6:
                    c2_idx = nb.GetIdx()

    if c1_idx is not None and o1_idx is not None and s_idx is not None:
        coords = np.array(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())]
        )
        coords -= coords[c1_idx]
        if mol.GetNumAtoms() >= 6:
            centered = coords - coords.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            axis_vec = vt[0]
            if np.dot(axis_vec, coords[c1_idx] - coords.mean(axis=0)) < 0:
                axis_vec = -axis_vec
            angle = -_math.atan2(axis_vec[1], axis_vec[0])
        else:
            v_o = coords[o1_idx]
            angle = _math.atan2(v_o[0], v_o[1])
        c_a, s_a = _math.cos(angle), _math.sin(angle)
        rot = np.array([[c_a, -s_a], [s_a, c_a]])
        coords = coords @ rot.T
        if coords[o1_idx][1] < 0:
            coords[:, 1] *= -1
        if c2_idx is not None and coords[s_idx][0] < coords[c2_idx][0]:
            coords[:, 0] *= -1
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (float(coords[i, 0]), float(coords[i, 1]), 0.0))

    hi_atoms, hi_bonds = [], []
    if is_released_acid and c1_idx is not None and o1_idx is not None and s_idx is not None:
        unit_set = {c1_idx, o1_idx, s_idx}
        hi_atoms = sorted(unit_set)
        for b in mol.GetBonds():
            if b.GetBeginAtomIdx() in unit_set and b.GetEndAtomIdx() in unit_set:
                hi_bonds.append(b.GetIdx())
    elif highlight_new_unit and c1_idx is not None and c2_idx is not None:
        unit_set = {c1_idx, c2_idx}
        c2_atom = mol.GetAtomWithIdx(c2_idx)
        c2_c_nbs = [nb for nb in c2_atom.GetNeighbors() if nb.GetIdx() != c1_idx]
        best_c3, best_sz = None, -1
        for nb in c2_c_nbs:
            visited = {c2_idx, nb.GetIdx()}
            q = [nb.GetIdx()]
            while q:
                cur = q.pop()
                for n2 in mol.GetAtomWithIdx(cur).GetNeighbors():
                    if n2.GetIdx() not in visited:
                        visited.add(n2.GetIdx())
                        q.append(n2.GetIdx())
            if len(visited) > best_sz:
                best_sz = len(visited)
                best_c3 = nb.GetIdx()
        for nb in c2_c_nbs:
            unit_set.add(nb.GetIdx())
            if nb.GetIdx() != best_c3:
                for n2 in nb.GetNeighbors():
                    if n2.GetIdx() != c2_idx:
                        unit_set.add(n2.GetIdx())
        if best_c3 is not None:
            c3_atom = mol.GetAtomWithIdx(best_c3)
            for nb in c3_atom.GetNeighbors():
                if nb.GetIdx() != c2_idx and nb.GetAtomicNum() == 8 and nb.GetDegree() == 1:
                    unit_set.add(nb.GetIdx())
        hi_atoms = sorted(unit_set)
        for b in mol.GetBonds():
            if b.GetBeginAtomIdx() in unit_set and b.GetEndAtomIdx() in unit_set:
                hi_bonds.append(b.GetIdx())

    drawer = rdMolDraw2D.MolDraw2DCairo(1800, 600)
    dopts = drawer.drawOptions()
    dopts.fixedBondLength = fixed_bond_px
    dopts.bondLineWidth = 4.6
    dopts.minFontSize = 30
    dopts.maxFontSize = 38
    dopts.padding = 0.04
    dopts.continuousHighlight = True
    dopts.fillHighlights = True
    dopts.highlightBondWidthMultiplier = 10
    green = (0.54, 0.88, 0.56)
    a_cols = {a: green for a in hi_atoms}
    b_cols = {b: green for b in hi_bonds}
    a_rads = {a: 0.36 for a in hi_atoms}
    drawer.DrawMolecule(
        mol,
        highlightAtoms=hi_atoms,
        highlightBonds=hi_bonds,
        highlightAtomColors=a_cols,
        highlightBondColors=b_cols,
        highlightAtomRadii=a_rads,
    )
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        l_b, t_b, r_b, b_b = bbox
        img = img.crop((max(0, l_b - 14), max(0, t_b - 14), min(img.width, r_b + 14), min(img.height, b_b + 14)))
    return img


def render_iterative_vs_modular_pks_assembly_figure(output_png: str):
    """
    Render the side-by-side comparison of:
      - Left Panel (A): Fungal Iterative hrPKS (`LovB + trans-ER LovC`) looping 8× over the
        same `[KS - AT - DH - MT - ER - KR - ACP]` domain set (with inactive domains ghosted
        on each cycle) to form $4a,5$-dihydromonacolin L (`C19H30O4`).
      - Right Panel (B): Our Model's Bacterial Modular Type I PKS (`M0` to `M8`) for the
        `C18H28O6` PKS monomer (`CC=CC=CC=C(O)C[C@H](O)C=CCC[C@@H](O)C[C@@H](O)CC(=O)O`),
        using touching domain spheres (`dx_dom = 2.02 * R`), large 22pt domain labels
        (`KS`, `AT`, `DH`, `ER`, `KR`, `TE`), large horizontally aligned structures with zero
        overlapping text, and a strictly 2-tone Navy/Slate palette (zero rainbow colors).
    """
    import matplotlib.patches as patches
    if os.path.dirname(output_png):
        os.makedirs(os.path.dirname(output_png), exist_ok=True)

    iterative_rows = [
        ("C0", "CC(=O)[S]", False, {"KS", "AT"}),
        ("C1", "C/C=C/C(=O)[S]", True, {"KS", "AT", "DH", "KR"}),
        ("C2", "C/C=C/C=C/C(=O)[S]", True, {"KS", "AT", "DH", "KR"}),
        ("C3 (MT)", "C/C=C/C=C/C[C@H](C)C(=O)[S]", True, {"KS", "AT", "DH", "MT", "ER", "KR"}),
        ("C4", "C/C=C/C=C/C[C@H](C)CCC(=O)[S]", True, {"KS", "AT", "DH", "ER", "KR"}),
        ("C5 (DA)", "C[C@H]1C=CC2C[C@H](C)CC[C@H]2[C@H]1C(=O)[S]", True, {"KS", "AT", "DH", "KR"}),
        ("C6", "C[C@H]1C=CC2C[C@H](C)CC[C@H]2[C@H]1CCC(=O)[S]", True, {"KS", "AT", "DH", "ER", "KR"}),
        ("C7", "C[C@H]1C=CC2C[C@H](C)CC[C@H]2[C@H]1CC[C@@H](O)CC(=O)[S]", True, {"KS", "AT", "KR"}),
        ("C8\n(ACP-bound\nintermediate\nbefore LovG\nrelease)", "C[C@H]1C=CC2C[C@H](C)CC[C@H]2[C@H]1CC[C@@H](O)C[C@@H](O)CC(=O)[S]", True, {"KS", "AT", "KR"}),
    ]
    iter_released_acid_smi = "C[C@H]1C=CC2C[C@H](C)CC[C@H]2[C@H]1CC[C@@H](O)C[C@@H](O)CC(=O)O"

    modular_rows = [
        ("M0", "CC(=O)[S]", False, ["KS", "AT"], False),
        ("M1", "C/C=C/C(=O)[S]", True, ["KS", "AT", "DH", "KR"], False),
        ("M2", "C/C=C/C=C/C(=O)[S]", True, ["KS", "AT", "DH", "KR"], False),
        ("M3 (hmal)", "C/C=C/C=C/C=C(\\O)C(=O)[S]", True, ["KS", "AT", "DH", "KR"], False),
        ("M4 (hmal)", "C/C=C/C=C/C=C(\\O)C[C@H](O)C(=O)[S]", True, ["KS", "AT", "DH", "ER", "KR"], False),
        ("M5", "C/C=C/C=C/C=C(\\O)C[C@H](O)/C=C/C(=O)[S]", True, ["KS", "AT", "DH", "KR"], False),
        ("M6", "C/C=C/C=C/C=C(\\O)C[C@H](O)/C=C/CCC(=O)[S]", True, ["KS", "AT", "DH", "ER", "KR"], False),
        ("M7", "C/C=C/C=C/C=C(\\O)C[C@H](O)/C=C/CC[C@@H](O)CC(=O)[S]", True, ["KS", "AT", "KR"], False),
        ("M8\n(ACP-bound\nintermediate\nbefore TE-mediated\nrelease)", "C/C=C/C=C/C=C(\\O)C[C@H](O)/C=C/CC[C@@H](O)C[C@@H](O)CC(=O)[S]", True, ["KS", "AT", "KR", "TE"], True),
    ]
    mod_released_acid_smi = "C/C=C/C=C/C=C(\\O)C[C@H](O)/C=C/CC[C@@H](O)C[C@@H](O)CC(=O)O"

    # Strictly 2-tone Navy/Steel-Slate alternation (zero rainbow colors!)
    MOD_COLORS = [
        "#1B365D",  # M0: Deep Navy
        "#4B6B94",  # M1: Steel Slate
        "#1B365D",  # M2: Deep Navy
        "#4B6B94",  # M3: Steel Slate
        "#1B365D",  # M4: Deep Navy
        "#4B6B94",  # M5: Steel Slate
        "#1B365D",  # M6: Deep Navy
        "#4B6B94",  # M7: Steel Slate
        "#1B365D",  # M8: Deep Navy
    ]

    fig_w, fig_h = 25.4, 18.2
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=220, facecolor="white")
    ax_full = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax_full.set_xlim(0, fig_w)
    ax_full.set_ylim(0, fig_h)
    ax_full.axis("off")

    # Compact, high-contrast headers
    ax_full.text(
        6.0, fig_h - 0.20,
        "(A) Native Fungal Iterative hrPKS — Lovastatin Pathway (LovB + LovC)",
        ha="center", va="top", fontsize=17.5, fontweight="bold", color="#7A1814",
    )
    ax_full.text(
        6.0, fig_h - 0.60,
        "1 enzyme loops 8× (faded = inactive in cycle; MT active at C3 tetraketide)\n"
        "→ LovG hydrolytic release yields dihydromonacolin L acid (C19H32O4)",
        ha="center", va="top", fontsize=11.2, fontweight="bold", color="#475467", linespacing=1.25,
    )

    ax_full.text(
        18.8, fig_h - 0.20,
        "(B) Proposed Bacterial Modular Type I PKS Assembly",
        ha="center", va="top", fontsize=17.5, fontweight="bold", color="#1B365D",
    )
    ax_full.text(
        18.8, fig_h - 0.60,
        "Proposed installation of hydroxyl groups during chain assembly using hmal at M3 and M4.\n"
        "hmal = (2R)-hydroxymalonyl-ACP (carrier-bound extender for hmal-specific ATs); M3 (DH–KR) retains C12 enol-OH,\n"
        "M4 (DH–ER–KR) retains C10 (2R)-α-OH (triene/enol unsaturation formed via M1–M3 DH; C4a=C5 P450 requirement noted)",
        ha="center", va="top", fontsize=10.2, fontweight="bold", color="#475467", linespacing=1.22,
    )

    ax_full.plot([12.25, 12.25], [0.75, fig_h - 0.18], color="#D0D5DD", lw=2.2, ls="--")

    n_rows = 9
    R = 0.42               # Large domain circle radius
    dx_dom = 2.02 * R      # Circles touch horizontally just like the BioPKS diagram!
    r_acp = 0.225          # Large tucked ACP circle radius
    row_step = 1.50        # Vertical spacing so ACP bridges KS_k and KS_{k+1}
    top_y = fig_h - 1.88
    PX_PER_INCH = 165.0    # Crisp high-DPI structure scaling

    # =========================================================================
    # Draw Left Panel (A): Fungal Iterative hrPKS (LovB + LovC)
    # =========================================================================
    iter_domains = ["KS", "AT", "DH", "MT", "ER", "KR"]
    x_iter_dom0 = 6.35
    iter_color = "#7A1814"

    for r_idx, (cyc_lbl, smi, hi_flag, active_set) in enumerate(iterative_rows):
        cy = top_y - r_idx * row_step

        img_mol = _draw_aligned_acyl_acp(smi, highlight_new_unit=hi_flag, fixed_bond_px=52)
        w_in = img_mol.width / PX_PER_INCH
        h_in = img_mol.height / PX_PER_INCH
        max_w = 5.65
        max_h = 1.38
        scale_f = min(1.0, max_w / w_in, max_h / h_in)
        w_in *= scale_f
        h_in *= scale_f
        x_right = x_iter_dom0 - R - 0.16
        ax_full.imshow(
            img_mol,
            extent=[x_right - w_in, x_right, cy - 0.48 * h_in, cy + 0.52 * h_in],
            aspect="auto",
            zorder=3,
        )

        kr_x = x_iter_dom0 + 5 * dx_dom
        ax_full.text(
            kr_x + R + 0.12, cy,
            cyc_lbl,
            ha="left", va="center",
            fontsize=10.5 if "\n" in cyc_lbl else 13.5,
            fontweight="bold", color="#7A1814", linespacing=1.08,
        )

        acp_x = x_iter_dom0 + 0.5 * dx_dom
        acp_y = cy - 0.50
        ax_full.plot([kr_x, kr_x, acp_x + r_acp], [cy - R, acp_y, acp_y], color="#111111", lw=2.6, zorder=2)
        if r_idx < n_rows - 1:
            ax_full.plot(
                [acp_x - r_acp, x_iter_dom0, x_iter_dom0],
                [acp_y, acp_y, cy - row_step + R],
                color="#7A1814", lw=2.2, ls=":", zorder=2,
            )

        for j, d_key in enumerate(iter_domains):
            cx = x_iter_dom0 + j * dx_dom
            is_act = d_key in active_set
            circ = patches.Circle(
                (cx, cy), R,
                facecolor=iter_color if is_act else "#F2F4F7",
                edgecolor=iter_color if is_act else "#98A2B3",
                linewidth=1.6 if is_act else 1.3,
                linestyle="-" if is_act else "--",
                alpha=0.98 if is_act else 0.55,
                zorder=4,
            )
            ax_full.add_patch(circ)
            ax_full.text(
                cx, cy, d_key,
                ha="center", va="center",
                fontsize=21.0,
                fontweight="bold",
                color="white" if is_act else "#667085",
                zorder=5,
            )

        c_acp = patches.Circle((acp_x, acp_y), r_acp, facecolor=iter_color, edgecolor="none", zorder=5)
        ax_full.add_patch(c_acp)
        ax_full.text(acp_x, acp_y, "ACP", ha="center", va="center", fontsize=12.0, fontweight="bold", color="white", zorder=6)

    # Separate hydrolytic release arrow + free carboxylic acid product below C8 (Item 7 & Item 2)
    cy_c8 = top_y - (n_rows - 1) * row_step
    cy_rel = cy_c8 - 1.62
    acp_x_iter = x_iter_dom0 + 0.5 * dx_dom
    ax_full.annotate(
        "",
        xy=(x_iter_dom0 - R - 0.16, cy_rel + 0.08),
        xytext=(acp_x_iter, cy_c8 - 0.50 - r_acp),
        arrowprops=dict(arrowstyle="-|>", color="#7A1814", lw=2.4, mutation_scale=15),
    )
    img_iter_rel = _draw_aligned_acyl_acp(iter_released_acid_smi, highlight_new_unit=False, fixed_bond_px=52, is_released_acid=True)
    w_in = img_iter_rel.width / PX_PER_INCH
    h_in = img_iter_rel.height / PX_PER_INCH
    scale_f = min(1.0, 5.65 / w_in, 1.38 / h_in)
    w_in *= scale_f
    h_in *= scale_f
    x_right_iter = x_iter_dom0 - R - 0.16
    ax_full.imshow(
        img_iter_rel,
        extent=[x_right_iter - w_in, x_right_iter, cy_rel - 0.48 * h_in, cy_rel + 0.52 * h_in],
        aspect="auto",
        zorder=3,
    )
    ax_full.text(
        x_iter_dom0, cy_rel + 0.04,
        "LovG hydrolytic release (–ACP–SH, +H2O)\n"
        "Released product: Dihydromonacolin L acid (C19H32O4)",
        ha="left", va="center", fontsize=12.0, fontweight="bold", color="#7A1814",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FEF3F2", edgecolor="#FECDCA", lw=1.4),
    )

    # =========================================================================
    # Draw Right Panel (B): Bacterial Modular Type I PKS (M0..M8)
    # =========================================================================
    x_mod_dom0 = 19.65
    for r_idx, (mod_lbl, smi, hi_flag, dom_list, has_te) in enumerate(modular_rows):
        cy = top_y - r_idx * row_step
        row_col = MOD_COLORS[r_idx % len(MOD_COLORS)]
        prev_col = MOD_COLORS[(r_idx - 1) % len(MOD_COLORS)] if r_idx > 0 else row_col

        img_mol = _draw_aligned_acyl_acp(smi, highlight_new_unit=hi_flag, fixed_bond_px=52)
        w_in = img_mol.width / PX_PER_INCH
        h_in = img_mol.height / PX_PER_INCH
        max_w = 6.65
        max_h = 1.38
        scale_f = min(1.0, max_w / w_in, max_h / h_in)
        w_in *= scale_f
        h_in *= scale_f
        x_right = x_mod_dom0 - R - 0.16
        ax_full.imshow(
            img_mol,
            extent=[x_right - w_in, x_right, cy - 0.48 * h_in, cy + 0.52 * h_in],
            aspect="auto",
            zorder=3,
        )

        acp_x = x_mod_dom0 + 0.5 * dx_dom
        acp_y = cy - 0.50
        last_cat_j = len(dom_list) - 2 if has_te else len(dom_list) - 1
        last_cat_x = x_mod_dom0 + last_cat_j * dx_dom
        if last_cat_j >= 1:
            ax_full.plot(
                [last_cat_x, last_cat_x, acp_x + r_acp],
                [cy - R, acp_y, acp_y],
                color="#111111", lw=2.8, zorder=2,
            )
        if r_idx < n_rows - 1:
            ax_full.plot(
                [acp_x - r_acp, x_mod_dom0, x_mod_dom0],
                [acp_y, acp_y, cy - row_step + R],
                color="#111111", lw=2.8, zorder=2,
            )

        for j, d_disp in enumerate(dom_list):
            cx = x_mod_dom0 + j * dx_dom
            c_fill = prev_col if (j == 0 and r_idx > 0) else row_col
            circ = patches.Circle((cx, cy), R, facecolor=c_fill, edgecolor="none", zorder=4)
            ax_full.add_patch(circ)
            ax_full.text(
                cx, cy, d_disp,
                ha="center", va="center",
                fontsize=21.0,
                fontweight="bold",
                color="white",
                zorder=5,
            )

        rightmost_x = x_mod_dom0 + (len(dom_list) - 1) * dx_dom + R + 0.14
        ax_full.text(
            rightmost_x, cy,
            mod_lbl,
            ha="left", va="center",
            fontsize=10.5 if "\n" in mod_lbl else 13.5,
            fontweight="bold", color=row_col, linespacing=1.08,
        )

        c_acp = patches.Circle((acp_x, acp_y), r_acp, facecolor=row_col, edgecolor="none", zorder=5)
        ax_full.add_patch(c_acp)
        ax_full.text(acp_x, acp_y, "ACP", ha="center", va="center", fontsize=12.0, fontweight="bold", color="white", zorder=6)

    # Separate TE-mediated hydrolytic release arrow + free carboxylic acid product below M8 (Items 4 & 7)
    te_x_mod = x_mod_dom0 + 3 * dx_dom
    ax_full.annotate(
        "",
        xy=(x_mod_dom0 - R - 0.16, cy_rel + 0.08),
        xytext=(te_x_mod, cy_c8 - R),
        arrowprops=dict(arrowstyle="-|>", color="#1B365D", lw=2.4, mutation_scale=15),
    )
    img_mod_rel = _draw_aligned_acyl_acp(mod_released_acid_smi, highlight_new_unit=False, fixed_bond_px=52, is_released_acid=True)
    w_in = img_mod_rel.width / PX_PER_INCH
    h_in = img_mod_rel.height / PX_PER_INCH
    scale_f = min(1.0, 6.65 / w_in, 1.38 / h_in)
    w_in *= scale_f
    h_in *= scale_f
    x_right_mod = x_mod_dom0 - R - 0.16
    ax_full.imshow(
        img_mod_rel,
        extent=[x_right_mod - w_in, x_right_mod, cy_rel - 0.48 * h_in, cy_rel + 0.52 * h_in],
        aspect="auto",
        zorder=3,
    )
    ax_full.text(
        x_mod_dom0 - 0.25, cy_rel + 0.04,
        "TE-mediated hydrolytic release (–ACP–SH, +H2O)\n"
        "Released product: Proposed linear nonaketide precursor (C18H28O6)",
        ha="left", va="center", fontsize=11.6, fontweight="bold", color="#1B365D",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#EFF8FF", edgecolor="#B2DDFF", lw=1.4),
    )

    # Consistent downstream footers (Items 2, 3, 4, 6)
    ax_full.text(
        6.0, 0.14,
        "Native Lovastatin Pathway: Dihydromonacolin L acid (C19H32O4) → LovA (P450 C4a,C5-alkene & C8-OH)\n"
        "→ LovD + LovF ((2S)-2-methylbutyryl-ACP acyl donor) → Lovastatin",
        ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#7A1814", linespacing=1.22,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF3F2", edgecolor="#FECDCA", lw=1.6),
    )
    ax_full.text(
        18.8, 0.14,
        "Downstream from Proposed Linear Nonaketide Precursor (C18H28O6): Proposed cyclization and acylation; enzyme compatibility to be validated.\n"
        "(SpnF: candidate for testing; activity on the proposed substrate is unverified.)",
        ha="center", va="bottom", fontsize=11.2, fontweight="bold", color="#1B365D", linespacing=1.22,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#EFF8FF", edgecolor="#B2DDFF", lw=1.6),
    )

    fig.savefig(output_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved Iterative vs. Modular PKS assembly-line comparison to: {output_png}")


def render_modular_pks_assembly_line_for_target(
    result: Dict[str, Any],
    target_name: str = "Target",
    output_png: Optional[str] = None,
):
    """
    Render the BioPKS/RetroTide-style vertical modular assembly-line diagram
    (growing acyl-S-ACP intermediates with green highlights on the newly added
    C1–C2(α)–C3(β) unit + circular KS/AT/DH/ER/KR/TE domain bubbles + tucked ACP
    + black linker lines) for any solved target `result`.
    """
    import matplotlib.patches as patches
    from deterministic_stage import pks_terminal as pt

    core_smi = result.get("peeled_core_smiles", "")
    rows, meta = pt.decompose_pks_terminal(core_smi)
    if not rows or "assembly_validation" not in meta:
        return
    fwd = meta["assembly_validation"]["forward_program"]
    chemistry, starters, fragments, templates = pt._assembly_resources()
    starter_smi, inters = chemistry._simulate_bgc_intermediates(
        fwd["starter"], fwd["modules"], starters, fragments, templates
    )

    MOD_COLORS = [
        "#1B365D", "#4B6B94", "#1B365D", "#4B6B94", "#1B365D",
        "#4B6B94", "#1B365D", "#4B6B94", "#1B365D",
    ]

    mod_specs = []
    for idx, r in enumerate(rows):
        m_num = int(r["module_num"])
        if m_num == 0:
            smi = starter_smi
            doms = ["KS", "AT"]
            has_te = False
        else:
            smi = inters.get(m_num, "")
            state = r.get("beta_state", "")
            if state == "saturated":
                doms = ["KS", "AT", "DH", "ER", "KR"]
            elif state == "enoyl":
                doms = ["KS", "AT", "DH", "KR"]
            elif state == "β hydroxyl":
                doms = ["KS", "AT", "KR"]
            else:
                doms = ["KS", "AT"]
            has_te = idx == len(rows) - 1
            if has_te:
                doms.append("TE")
        mod_specs.append((f"M{m_num}", smi, m_num > 0, doms, has_te))

    n_rows = len(mod_specs)
    row_step = 1.32
    fig_h = max(5.2, n_rows * row_step + 1.4)
    fig = plt.figure(figsize=(10.2, fig_h), dpi=200, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    gene_plan = search_inhouse_wetlab_genes(list(rows), max_gene_kb=9.0, strictly_inhouse_strains=True)
    g_str = ""
    if gene_plan.get("feasible"):
        g_str = " | In-House Genes (<=9kb): " + ", ".join(
            f"[M{s['start_mod']}" + (f"..M{s['end_mod']}" if s["start_mod"] != s["end_mod"] else "") + f"]:{s['gene_id'].split('::')[-1]}({s['est_kb']}kb)"
            for s in gene_plan["best_assembly"]
        )
    ax.text(
        5.1, fig_h - 0.22,
        f"{target_name} — Proposed Modular Type I PKS ({n_rows - 1} Ext. Modules)",
        ha="center", va="top", fontsize=14.0, fontweight="bold", color="#1E3A5F",
    )
    if g_str:
        ax.text(
            5.1, fig_h - 0.56, g_str.lstrip(" | "),
            ha="center", va="top", fontsize=9.8, fontweight="bold", color="#344054",
        )

    top_y = fig_h - 1.35
    R = 0.365
    dx_dom = 2.02 * R
    r_acp = 0.195
    x_dom0 = 5.05
    PX_PER_INCH = 112.0

    for r_idx, (m_lbl, smi, hi_flag, dom_list, has_te) in enumerate(mod_specs):
        cy = top_y - r_idx * row_step
        row_col = MOD_COLORS[r_idx % len(MOD_COLORS)]
        prev_col = MOD_COLORS[(r_idx - 1) % len(MOD_COLORS)] if r_idx > 0 else row_col

        img_mol = _draw_aligned_acyl_acp(smi, highlight_new_unit=hi_flag, fixed_bond_px=38)
        w_in = img_mol.width / PX_PER_INCH
        h_in = img_mol.height / PX_PER_INCH
        max_w = 4.35
        if w_in > max_w:
            scale_f = max_w / w_in
            w_in *= scale_f
            h_in *= scale_f
        x_right = x_dom0 - R - 0.14
        ax.imshow(
            img_mol,
            extent=[x_right - w_in, x_right, cy - 0.48 * h_in, cy + 0.52 * h_in],
            aspect="auto",
            zorder=3,
        )

        acp_x = x_dom0 + 0.5 * dx_dom
        acp_y = cy - 0.44
        last_cat_j = len(dom_list) - 2 if has_te else len(dom_list) - 1
        last_cat_x = x_dom0 + last_cat_j * dx_dom
        if last_cat_j >= 1:
            ax.plot([last_cat_x, last_cat_x, acp_x + r_acp], [cy - R, acp_y, acp_y], color="#111111", lw=2.5, zorder=2)
        if r_idx < n_rows - 1:
            ax.plot([acp_x - r_acp, x_dom0, x_dom0], [acp_y, acp_y, cy - row_step + R], color="#111111", lw=2.5, zorder=2)

        for j, d_disp in enumerate(dom_list):
            cx = x_dom0 + j * dx_dom
            c_fill = prev_col if (j == 0 and r_idx > 0) else row_col
            circ = patches.Circle((cx, cy), R, facecolor=c_fill, edgecolor="none", zorder=4)
            ax.add_patch(circ)
            ax.text(
                cx, cy, d_disp,
                ha="center", va="center",
                fontsize=21.0,
                fontweight="bold", color="white", zorder=5,
            )

        rightmost_x = x_dom0 + (len(dom_list) - 1) * dx_dom + R + 0.12
        ax.text(
            rightmost_x, cy,
            m_lbl,
            ha="left", va="center", fontsize=13.5, fontweight="bold", color=row_col,
        )

        c_acp = patches.Circle((acp_x, acp_y), r_acp, facecolor=row_col, edgecolor="none", zorder=5)
        ax.add_patch(c_acp)
        ax.text(acp_x, acp_y, "ACP", ha="center", va="center", fontsize=12.0, fontweight="bold", color="white", zorder=6)

    if output_png:
        fig.savefig(output_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def render_peeling_pathway_image(result: Dict[str, Any], output_png: str, target_name: str = "Target"):
    """Render the reaction-scheme pathway diagram with bond-only highlights and detached monomers."""
    if os.path.dirname(output_png):
        os.makedirs(os.path.dirname(output_png), exist_ok=True)
    steps_spec, core_smi, final_title, final_subtitle = build_steps_spec_from_engine_result(result, target_name=target_name)
    n_cols = len(steps_spec) + 1
    fig_w = max(12.0, 4.4 * n_cols + 2.2 * len(steps_spec))
    fig = plt.figure(figsize=(fig_w, 6.2), dpi=220, facecolor="white")
    render_reaction_row_on_axes(
        fig,
        y_top=0.92,
        row_height=0.84,
        steps_spec=steps_spec,
        final_smi=core_smi,
        final_title=final_title,
        final_subtitle=final_subtitle,
        badge_color="#28B446",
    )
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


DEFAULT_INHOUSE_GENES_TSV = str(Path(__file__).resolve().parent / "inhouse_antismash_t1pks_genes.tsv")
DEFAULT_GENE_SEARCH_DIR = str(Path(__file__).resolve().parent)


def _estimate_gene_kb(gene_rec: Any) -> float:
    d_kb = {
        "KS": 1.35, "AT": 1.05, "DH": 0.55, "ER": 0.95,
        "KR": 0.55, "ACP": 0.25, "PP": 0.25, "TE": 0.80,
    }
    return round(
        sum(0.30 + sum(d_kb.get(d.type, 0.30) for d in m.domains) for m in gene_rec.modules),
        1,
    )


def search_inhouse_wetlab_genes(
    pks_module_rows: List[Dict[str, Any]],
    tsv_path: str = DEFAULT_INHOUSE_GENES_TSV,
    max_gene_kb: float = 9.0,
    strictly_inhouse_strains: bool = True,
) -> Dict[str, Any]:
    """
    Query the in-house antiSMASH Type I PKS gene inventory (`inhouse_antismash_t1pks_genes.tsv`)
    via dynamic programming (`gene_search.py`) to find minimal whole-gene combinations that
    reconstitute the predicted PKS module sequence while respecting the wet-lab Q5 PCR + pJET1.2
    blunt-cloning size ceiling (`max_gene_kb <= 9.0 kb`, eliminating 14 kb 3-module genes).
    """
    if not pks_module_rows or not Path(tsv_path).exists():
        return {"feasible": False, "reason": "no_pks_modules_or_tsv"}

    sub_map = {
        "mal": "malonyl coa",
        "mmal": "methylmalonyl coa",
        "emal": "ethylmalonyl coa",
        "hmal": "malonyl coa or methylmalonyl coa",
        "mxmal": "malonyl coa or methylmalonyl coa",
    }
    rows = [r for r in pks_module_rows if int(r.get("module_num", -1)) >= 0]
    for r in rows:
        if r.get("at_annotation", "") not in sub_map:
            return {
                "feasible": False,
                "reason": f"AT substrate '{r.get('at_annotation')}' not in in-house gene inventory",
            }

    import types as _types
    if "openpyxl" not in sys.modules:
        sys.modules["openpyxl"] = _types.SimpleNamespace(Workbook=object)
    if DEFAULT_GENE_SEARCH_DIR not in sys.path:
        sys.path.insert(0, DEFAULT_GENE_SEARCH_DIR)
    from gene_search import (
        _build_gene_records,
        _build_position_edges,
        _format_gene_module,
        parse_module_list,
    )

    all_genes = _build_gene_records(tsv_path)
    filtered_genes = [
        g for g in all_genes
        if _estimate_gene_kb(g) <= max_gene_kb
        and (not strictly_inhouse_strains or not g.bgc_id.startswith("GCF_"))
    ]

    def _build_query(relax_kr: bool, include_te: bool) -> List[str]:
        q: List[str] = []
        for idx, r in enumerate(rows):
            at_str = f"AT({sub_map[r['at_annotation']]})"
            if int(r["module_num"]) == 0:
                q.append(f"(KS) - {at_str} - ACP")
                continue
            state = r.get("beta_state", "")
            kr = r.get("kr_subtype", "U")
            kr_pat = kr[0] if (relax_kr and kr in ("A1", "A2", "B1", "B2")) else kr
            if state == "saturated":
                m_str = f"KS - {at_str} - DH - ER - KR - ACP"
            elif state == "enoyl":
                m_str = f"KS - {at_str} - DH - KR - ACP"
            elif state == "β hydroxyl":
                m_str = (
                    f"KS - {at_str} - KR({kr_pat}) - ACP"
                    if (kr_pat and kr_pat != "U")
                    else f"KS - {at_str} - KR - ACP"
                )
            else:
                m_str = f"KS - {at_str} - ACP"
            if include_te and idx == len(rows) - 1:
                m_str += " - TE"
            q.append(m_str)
        return q

    for relax_kr in (False, True):
        for inc_te in (True, False):
            query = _build_query(relax_kr=relax_kr, include_te=inc_te)
            pats = parse_module_list(query)
            edges = _build_position_edges(pats, filtered_genes, allow_gene_splitting=False)
            n = len(pats)
            dp = [10**9] * (n + 1)
            dp[0] = 0
            parent: Dict[int, Tuple[int, Any]] = {}
            for i in range(n):
                if dp[i] == 10**9:
                    continue
                for end, seg in edges[i]:
                    if dp[i] + 1 < dp[end]:
                        dp[end] = dp[i] + 1
                        parent[end] = (i, seg)
            if dp[n] < 10**9:
                segments = []
                cur = n
                while cur > 0:
                    prev, seg = parent[cur]
                    segments.append({
                        "start_mod": prev,
                        "end_mod": cur - 1,
                        "gene_id": seg.gene.gene_id,
                        "est_kb": _estimate_gene_kb(seg.gene),
                        "modules": [_format_gene_module(gm) for gm in seg.modules],
                    })
                    cur = prev
                segments.reverse()
                return {
                    "feasible": True,
                    "num_genes": len(segments),
                    "max_gene_kb_limit": max_gene_kb,
                    "exact_kr_subtypes": not relax_kr,
                    "terminal_te_included": inc_te,
                    "query_modules": query,
                    "best_assembly": segments,
                }
    return {"feasible": False, "reason": "no_contiguous_whole_gene_match_within_kb_limit"}


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

        gene_plan = search_inhouse_wetlab_genes(result.get("module_rows", []))
        lines.append("\n  Wet-Lab In-House Gene Assembly Plan (`gene_search.py` DP over 1,683 In-House T1PKS Genes):")
        if gene_plan.get("feasible"):
            lines.append(
                f"  Status: FEASIBLE ({gene_plan['num_genes']} intact in-house gene(s), "
                f"KR match={'exact subtype' if gene_plan['exact_kr_subtypes'] else 'A/B stereoclass'}, "
                f"TE={'included' if gene_plan['terminal_te_included'] else 'external'})"
            )
            for idx, seg in enumerate(gene_plan["best_assembly"], start=1):
                mod_range = (
                    f"M{seg['start_mod']}"
                    if seg["start_mod"] == seg["end_mod"]
                    else f"M{seg['start_mod']}..M{seg['end_mod']}"
                )
                lines.append(
                    f"    Gene {idx} [{mod_range:<6} | ~{seg.get('est_kb', 0.0):>3.1f} kb] -> {seg['gene_id']:<34} | "
                    + " // ".join(seg["modules"])
                )
        else:
            lines.append(f"  Status: NOT DIRECTLY ASSEMBLABLE ({gene_plan.get('reason')})")
    elif meta.get("failure"):
        lines.append(f"\nDeterministic Stage Failure Diagnostic: {json.dumps(meta['failure'], indent=2)}")

    lines.append("=" * 84)
    return "\n".join(lines)


def get_compute_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Run GNN-guided retrobiosynthetic reverse peeling and deterministic PKS module readout."
    )
    parser.add_argument("--smiles", type=str, required=True, help="Target molecule SMILES to peel and decompose")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to 3-head GNN checkpoint")
    parser.add_argument("--max_depth", type=int, default=12, help="Maximum reverse-peeling steps")
    parser.add_argument("--beam_width", type=int, default=8, help="Beam width for search")
    parser.add_argument("--plot_out", type=str, default=None, help="Optional PNG path for pathway diagram")
    args = parser.parse_args()

    device = get_compute_device()
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    model = load_peeling_model(args.checkpoint, device)
    rule_bank = build_enzymatic_rule_bank()

    res = run_reverse_peeling_search(
        model, args.smiles, rule_bank, device, max_depth=args.max_depth, beam_width=args.beam_width
    )
    print(format_deterministic_stage_report(res))
    if args.plot_out:
        render_peeling_pathway_image(res, args.plot_out)


if __name__ == "__main__":
    main()