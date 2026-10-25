#!/usr/bin/env python3
"""
3-Head Reverse-Peeling GNN with C1-Register PE, SynCat Cross-Attention & PATRO Pathway Highway

Implements the complete specification in `retrobiosynthesis.txt` and `build_guidelines.txt`,
integrating architectural modules from `SynCat` and `PATRO`:

1. C1-Register & Topological Positional Encodings (`build_guidelines.txt` Section 5.1):
   - Identifies the C1 carbonyl carbon v_{C1} (thioester, macrolactone/lactam ester, or carboxylic acid).
   - Computes shortest path distance d_{C1}(v) from C1 to every atom v in V.
   - Appends explicit register parity features:
       p_reg(v) = [d_{C1}(v)/20, d_{C1}(v) mod 2, sin(pi*d_{C1}(v)), cos(pi*d_{C1}(v)),
                   sin(pi*d_{C1}(v)/2), cos(pi*d_{C1}(v)/2), is_c1(v), RWSE_{1..16}(v), LapPE_{1..4}(v)]
     breaking the local subgraph symmetry between inline PKS KR/AT groups (odd/even C1 register)
     and post-PKS tailoring modifications.

2. SynCat Molecule-Level Cross-Attention (`SynCat/src/attention.py` & `SynCat/src/model.py`):
   - Learns a bank of SynKit ITS Rule Prototype embeddings Q in R^{R x D} initialized with
     each rule's SynKit ITS reaction-center descriptors (RC node count, RC edge count, phase,
     bridge-forming flag, oxygen-capping flag).
   - Uses `SynCatSingleHeadAttention` (LayerNorm + scaled dot-product cross-attention) so each
     candidate reverse enzymatic rule r^{-1} attends directly to exposed non-core reaction-center
     atoms K = h_v * sigma(exposed_logit(v)) * (1 - p_core(v)).

3. PATRO Pathway-History LSTM + Highway Gate (`PATRO/ASKCOSv2/retro/template_relevance/models.py`):
   - Encodes the pathway history / phase context via an LSTM (`PATROPathwayLSTM`) and fuses it
     with the SynCat cross-attended rule representation through a `PATROHighway` gating layer
     (`transform * gate + x * (1 - gate)`).

4. Three Cooperative Output Heads:
   - Head 1: Core vs. Tailoring Node Shield `p_core(v)`, 10-way Register Role, & Bridge-Bond Segmentation
   - Head 2: Causal Exposed-Leaf Reaction Center `pi_exposed(v)` & Next Reverse Rule Policy `pi_peel(r^{-1} | G, H)`
   - Head 3: Distance-to-PKS Value Network `V(G) = exp(-d/2.5)` & Nascent PKS Core Classifier
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem, rdmolops

RDLogger.DisableLog("rdApp.*")


# =============================================================================
# Constants & Vocabularies
# =============================================================================

NUM_ENZYMATIC_RULES = 33

REGISTER_ROLES = [
    "C1",
    "alpha",
    "beta",
    "starter_backbone",
    "alpha_branch",
    "beta_oxygen",
    "c1_hetero",
    "core_other",
    "tailoring_decoration",
    "corrupted_core",
]
REGISTER_ROLE_MAP = {r: idx for idx, r in enumerate(REGISTER_ROLES)}
NUM_REGISTER_ROLES = len(REGISTER_ROLES)

ATOM_TYPES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
CHIRAL_TAGS = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    rdchem.ChiralType.CHI_OTHER,
]
BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]
BOND_STEREOS = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
]

RWSE_STEPS = 16
LAP_PE_DIM = 4
C1_REG_DIM = 7  # [d_c1/20, d_c1%2, sin(pi*d), cos(pi*d), sin(pi*d/2), cos(pi*d/2), is_c1]


def _build_mapping(values: Iterable) -> Dict:
    return {value: idx for idx, value in enumerate(values)}


ATOM_MAP = _build_mapping(ATOM_TYPES)
DEGREE_MAP = _build_mapping(DEGREES)
CHARGE_MAP = _build_mapping(FORMAL_CHARGES)
NUM_H_MAP = _build_mapping(NUM_HS)
HYB_MAP = {hyb: idx for idx, hyb in enumerate(HYBRIDIZATIONS)}
CHIRAL_MAP = {ct: idx for idx, ct in enumerate(CHIRAL_TAGS)}
BOND_MAP = {bond: idx for idx, bond in enumerate(BOND_TYPES)}
BOND_STEREO_MAP = {bs: idx for idx, bs in enumerate(BOND_STEREOS)}

_C1_CANDIDATE_SMARTS = [
    Chem.MolFromSmarts("[CX3](=O)[S;D1]"),
    Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),
    Chem.MolFromSmarts("[CX3;R](=O)[OX2H0;R]"),
    Chem.MolFromSmarts("[CX3;R](=O)[NX3;R]"),
    Chem.MolFromSmarts("[CX3](=O)[NH1][CX4][CX3](=O)[OX2H1]"),
    Chem.MolFromSmarts("[CX3](=O)[O,N]"),
]


def _one_hot(value, mapping: Dict) -> np.ndarray:
    size = len(mapping) + 1
    vec = np.zeros(size, dtype=np.float32)
    vec[mapping.get(value, len(mapping))] = 1.0
    return vec


# =============================================================================
# C1-Register Parity Encoding + RWSE (16-step) + Laplacian PE
# =============================================================================

def find_c1_anchor_idx(mol: Chem.Mol) -> Optional[int]:
    """Identify the primary C1 acyl carbonyl carbon index in `mol`."""
    for pat in _C1_CANDIDATE_SMARTS:
        if pat is None:
            continue
        matches = mol.GetSubstructMatches(pat)
        if matches:
            return int(matches[0][0])
    return None


def compute_c1_register_encoding(mol: Chem.Mol) -> np.ndarray:
    """
    Compute the C1-Register Positional Encoding specified in `build_guidelines.txt` Section 5.1:
        p_reg(v) = [d_{C1}(v)/20, d_{C1}(v) mod 2, sin(pi*d), cos(pi*d), sin(pi*d/2), cos(pi*d/2), is_c1]
    """
    n = mol.GetNumAtoms()
    if n == 0:
        return np.zeros((0, C1_REG_DIM), dtype=np.float32)

    c1_idx = find_c1_anchor_idx(mol)
    dist_mat = rdmolops.GetDistanceMatrix(mol)

    reg_pe = np.zeros((n, C1_REG_DIM), dtype=np.float32)
    if c1_idx is not None and 0 <= c1_idx < n:
        for v in range(n):
            d = float(dist_mat[c1_idx, v])
            if d > 100:  # disconnected component fallback
                d = 20.0
            reg_pe[v, 0] = min(d, 25.0) / 20.0
            reg_pe[v, 1] = float(int(round(d)) % 2)
            reg_pe[v, 2] = float(np.sin(np.pi * d))
            reg_pe[v, 3] = float(np.cos(np.pi * d))
            reg_pe[v, 4] = float(np.sin(0.5 * np.pi * d))
            reg_pe[v, 5] = float(np.cos(0.5 * np.pi * d))
            reg_pe[v, 6] = 1.0 if v == c1_idx else 0.0
    return reg_pe


def compute_rwse_and_lappe(adj: np.ndarray, k_rw: int = RWSE_STEPS, k_lap: int = LAP_PE_DIM) -> np.ndarray:
    """
    Compute 16-step Random Walk Structural Encoding (landing probabilities P^1..P^16)
    and top-4 non-trivial Laplacian Eigenvector magnitudes per node.
    """
    n = adj.shape[0]
    if n == 0:
        return np.zeros((0, k_rw + k_lap), dtype=np.float32)

    deg = adj.sum(axis=1, keepdims=True)
    trans = adj / np.maximum(deg, 1.0)

    rw_diag = np.zeros((n, k_rw), dtype=np.float32)
    p_pow = np.eye(n, dtype=np.float32)
    for step in range(k_rw):
        p_pow = p_pow @ trans
        rw_diag[:, step] = np.diag(p_pow)

    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg.squeeze(-1), 1.0))
    norm_adj = (adj * deg_inv_sqrt[:, None]) * deg_inv_sqrt[None, :]
    lap = np.eye(n, dtype=np.float32) - norm_adj

    lap_pe = np.zeros((n, k_lap), dtype=np.float32)
    if n > 1:
        try:
            _, eigvecs = np.linalg.eigh(lap)
            non_trivial = np.abs(eigvecs[:, 1 : 1 + k_lap])
            lap_pe[:, : non_trivial.shape[1]] = non_trivial
        except Exception:
            pass

    return np.concatenate([rw_diag, lap_pe], axis=1)


def atom_to_feature(atom: rdchem.Atom, ring_info) -> np.ndarray:
    """Convert an RDKit atom into a stereochemistry-, tautomer-, and ring-aware feature vector."""
    idx = atom.GetIdx()
    cip_code = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
    cip_vec = np.array(
        [1.0 if cip_code == "R" else 0.0, 1.0 if cip_code == "S" else 0.0],
        dtype=np.float32,
    )

    in_r3 = float(ring_info.IsAtomInRingOfSize(idx, 3))
    in_r4 = float(ring_info.IsAtomInRingOfSize(idx, 4))
    in_r5 = float(ring_info.IsAtomInRingOfSize(idx, 5))
    in_r6 = float(ring_info.IsAtomInRingOfSize(idx, 6))
    in_r7 = float(ring_info.IsAtomInRingOfSize(idx, 7))
    in_macro = float(any(len(r) >= 8 and idx in r for r in ring_info.AtomRings()))

    is_carbonyl_c = float(
        atom.GetAtomicNum() == 6
        and any(
            b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(atom).GetAtomicNum() == 8
            for b in atom.GetBonds()
        )
    )
    is_hydroxyl_o = float(atom.GetAtomicNum() == 8 and atom.GetTotalNumHs() >= 1)
    is_carrier_s = float(atom.GetAtomicNum() == 16 and atom.GetDegree() == 1)
    prochiral_h = float(
        atom.GetAtomicNum() == 6
        and atom.GetHybridization() == rdchem.HybridizationType.SP3
        and atom.GetTotalNumHs() in (1, 2)
    )

    feats = [
        _one_hot(atom.GetAtomicNum(), ATOM_MAP),                         # 13
        _one_hot(atom.GetTotalDegree(), DEGREE_MAP),                     # 7
        _one_hot(atom.GetFormalCharge(), CHARGE_MAP),                    # 6
        _one_hot(atom.GetTotalNumHs(includeNeighbors=True), NUM_H_MAP),  # 6
        _one_hot(atom.GetHybridization(), HYB_MAP),                      # 6
        _one_hot(atom.GetChiralTag(), CHIRAL_MAP),                       # 5
        cip_vec,                                                         # 2
        np.array(
            [
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
                in_r3,
                in_r4,
                in_r5,
                in_r6,
                in_r7,
                in_macro,
                is_carbonyl_c,
                is_hydroxyl_o,
                is_carrier_s,
                prochiral_h,
            ],
            dtype=np.float32,
        ),  # 12
    ]
    return np.concatenate(feats, axis=0)


def bond_to_feature(bond: Optional[rdchem.Bond], ring_info=None) -> np.ndarray:
    """Convert an RDKit bond into a stereo-, conjugation-, and macrocycle-aware feature vector."""
    if bond is None:
        bt = np.zeros(len(BOND_TYPES) + 1, dtype=np.float32)
        bt[-1] = 1.0
        bs = np.zeros(len(BOND_STEREOS) + 1, dtype=np.float32)
        bs[0] = 1.0
        extra = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return np.concatenate([bt, bs, extra], axis=0)

    bt = _one_hot(bond.GetBondType(), BOND_MAP)
    bs = _one_hot(bond.GetStereo(), BOND_STEREO_MAP)
    in_macro = 0.0
    if ring_info is not None and bond.IsInRing():
        b_idx = bond.GetIdx()
        in_macro = float(any(len(r) >= 8 and b_idx in r for r in ring_info.BondRings()))
    extra = np.array(
        [
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            in_macro,
            0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([bt, bs, extra], axis=0)


def smiles_to_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert SMILES into stereo-, C1-register-, and RWSE/LapPE-enriched graph tensors:
      - node_feat: [N, NODE_FEAT_DIM] (atom features + C1-Register PE + 16-step RWSE + LapPE)
      - edge_index: [2, E]
      - edge_attr: [E, EDGE_FEAT_DIM]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    n = mol.GetNumAtoms()
    if n == 0:
        raise ValueError(f"SMILES with no atoms: {smiles}")

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    ring_info = mol.GetRingInfo()

    atom_feats = np.vstack([atom_to_feature(atom, ring_info) for atom in mol.GetAtoms()]).astype(np.float32)
    c1_reg_pe = compute_c1_register_encoding(mol)
    adj = Chem.GetAdjacencyMatrix(mol).astype(np.float32)
    rw_lap_pe = compute_rwse_and_lappe(adj)
    node_feat = np.concatenate([atom_feats, c1_reg_pe, rw_lap_pe], axis=1).astype(np.float32)

    edges: List[Tuple[int, int]] = []
    edge_feat: List[np.ndarray] = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_to_feature(bond, ring_info)
        edges.append((u, v))
        edges.append((v, u))
        edge_feat.append(feat)
        edge_feat.append(feat)

    loop = bond_to_feature(None)
    for i in range(n):
        edges.append((i, i))
        edge_feat.append(loop)

    edge_index = np.array(edges, dtype=np.int64).T
    edge_attr = np.vstack(edge_feat).astype(np.float32)
    return node_feat, edge_index, edge_attr


_test_nf, _, _test_ea = smiles_to_graph("CC(=O)O")
NODE_FEAT_DIM = int(_test_nf.shape[1])
EDGE_FEAT_DIM = int(_test_ea.shape[1])


# =============================================================================
# Dataset & Batch Collation for 3-Head Supervision + PATRO Pathway Context
# =============================================================================

@dataclass
class PeelingGraphSample:
    node_feat: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    is_core_atom: np.ndarray
    register_role_idx: np.ndarray
    is_exposed_atom: np.ndarray
    is_bridge_edge: np.ndarray
    exposed_rule_mask: np.ndarray
    primary_rule_id: int
    pathway_context_tokens: np.ndarray  # [4] int64 sequence of phase/step context for PATRO LSTM
    value_target: float
    label: int


class PeelingTrajectoryDataset(Dataset):
    """Loads multi-step trajectory states from parquet for 3-head GNN training with in-memory graph caching."""

    def __init__(self, parquet_path: str, max_samples: Optional[int] = None):
        df = pd.read_parquet(parquet_path)
        if max_samples is not None:
            df = df.head(max_samples)
        self.df = df.reset_index(drop=True)
        self.node_feat_dim = NODE_FEAT_DIM
        self.edge_feat_dim = EDGE_FEAT_DIM
        self._cache: Dict[int, PeelingGraphSample] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> PeelingGraphSample:
        if idx in self._cache:
            return self._cache[idx]
        row = self.df.iloc[idx]
        smi = str(row["smiles"])
        nf, ei, ea = smiles_to_graph(smi)
        n = nf.shape[0]
        num_edges = ei.shape[1]

        if "is_pks_core_atom_json" in row and pd.notna(row["is_pks_core_atom_json"]):
            raw_core = json.loads(row["is_pks_core_atom_json"])
            is_core = np.array(raw_core[:n] + [False] * max(0, n - len(raw_core)), dtype=np.float32)
        else:
            is_core = np.full((n,), float(row.get("label", 0)), dtype=np.float32)

        if "atom_register_role_json" in row and pd.notna(row["atom_register_role_json"]):
            raw_roles = json.loads(row["atom_register_role_json"])
            role_ids = [REGISTER_ROLE_MAP.get(r, REGISTER_ROLE_MAP["core_other"]) for r in raw_roles[:n]]
            if len(role_ids) < n:
                role_ids.extend([REGISTER_ROLE_MAP["tailoring_decoration"]] * (n - len(role_ids)))
            register_role_idx = np.array(role_ids, dtype=np.int64)
        else:
            default_role = REGISTER_ROLE_MAP["core_other"] if row.get("label", 0) == 1 else REGISTER_ROLE_MAP["tailoring_decoration"]
            register_role_idx = np.full((n,), default_role, dtype=np.int64)

        if "is_exposed_peel_atom_json" in row and pd.notna(row["is_exposed_peel_atom_json"]):
            raw_exp = json.loads(row["is_exposed_peel_atom_json"])
            is_exposed = np.array(raw_exp[:n] + [False] * max(0, n - len(raw_exp)), dtype=np.float32)
        else:
            is_exposed = np.zeros((n,), dtype=np.float32)

        is_bridge_edge = np.zeros((num_edges,), dtype=np.float32)
        if "cleavable_bridge_bonds_json" in row and pd.notna(row["cleavable_bridge_bonds_json"]):
            bridge_pairs = {
                (int(u), int(v)) for u, v in json.loads(row["cleavable_bridge_bonds_json"])
            }
            bridge_pairs |= {(v, u) for (u, v) in list(bridge_pairs)}
            for e_i in range(num_edges):
                u, v = int(ei[0, e_i]), int(ei[1, e_i])
                if (u, v) in bridge_pairs:
                    is_bridge_edge[e_i] = 1.0

        exposed_rule_mask = np.zeros((NUM_ENZYMATIC_RULES,), dtype=np.float32)
        if "exposed_rule_ids_json" in row and pd.notna(row["exposed_rule_ids_json"]):
            for r_id in json.loads(row["exposed_rule_ids_json"]):
                if 0 <= int(r_id) < NUM_ENZYMATIC_RULES:
                    exposed_rule_mask[int(r_id)] = 1.0

        primary_rule_id = int(row.get("primary_reverse_rule_id", -1))

        # PATRO pathway history token sequence (following `PATRO/ASKCOSv2/retro/template_relevance/`
        # and `mcts_controller.py` / `retro_star_controller.py`):
        # Token 0 = "na" (empty / pad reaction slot at Step 0 when no reactions have been peeled yet).
        # Tokens 1..NUM_ENZYMATIC_RULES = rule_id + 1 of prior reverse reactions peeled along the path,
        # padded on the left with 0 ("na") so the MOST RECENT peeled reaction is LAST (index -1).
        rxn_ctx_rules: List[int] = []
        if "rxn_context_rule_ids_json" in row and pd.notna(row["rxn_context_rule_ids_json"]):
            raw_ctx = json.loads(row["rxn_context_rule_ids_json"])
            rxn_ctx_rules = [int(r) + 1 for r in raw_ctx if 0 <= int(r) < NUM_ENZYMATIC_RULES]
        rxn_ctx_rules = rxn_ctx_rules[-4:]
        padded_ctx = [0] * (4 - len(rxn_ctx_rules)) + rxn_ctx_rules
        pathway_tokens = np.array(padded_ctx, dtype=np.int64)

        value_target = float(row.get("value_target", float(row.get("label", 0))))
        label = int(row.get("label", 0))

        sample = PeelingGraphSample(
            node_feat=nf,
            edge_index=ei,
            edge_attr=ea,
            is_core_atom=is_core,
            register_role_idx=register_role_idx,
            is_exposed_atom=is_exposed,
            is_bridge_edge=is_bridge_edge,
            exposed_rule_mask=exposed_rule_mask,
            primary_rule_id=primary_rule_id,
            pathway_context_tokens=pathway_tokens,
            value_target=value_target,
            label=label,
        )
        self._cache[idx] = sample
        return sample


PAD_NODES_MULTIPLE = 256
PAD_EDGES_MULTIPLE = 1024


def collate_peeling_graphs(batch: List[PeelingGraphSample]) -> Dict[str, torch.Tensor]:
    node_feats, edge_indices, edge_attrs, batch_index = [], [], [], []
    is_core_list, role_list, is_exp_list, is_bridge_list = [], [], [], []
    rule_masks, primary_rules, path_tokens, val_targets, labels = [], [], [], [], []
    offset = 0

    for g_idx, sample in enumerate(batch):
        x = torch.from_numpy(sample.node_feat)
        ei = torch.from_numpy(sample.edge_index) + offset
        ea = torch.from_numpy(sample.edge_attr)
        n = x.size(0)

        node_feats.append(x)
        edge_indices.append(ei)
        edge_attrs.append(ea)
        batch_index.append(torch.full((n,), g_idx, dtype=torch.long))

        is_core_list.append(torch.from_numpy(sample.is_core_atom))
        role_list.append(torch.from_numpy(sample.register_role_idx))
        is_exp_list.append(torch.from_numpy(sample.is_exposed_atom))
        is_bridge_list.append(torch.from_numpy(sample.is_bridge_edge))

        rule_masks.append(torch.from_numpy(sample.exposed_rule_mask))
        primary_rules.append(sample.primary_rule_id)
        path_tokens.append(torch.from_numpy(sample.pathway_context_tokens))
        val_targets.append(sample.value_target)
        labels.append(sample.label)
        offset += n

    cat_nf = torch.cat(node_feats, dim=0)
    cat_ei = torch.cat(edge_indices, dim=1)
    cat_ea = torch.cat(edge_attrs, dim=0)
    cat_batch = torch.cat(batch_index, dim=0)
    cat_core = torch.cat(is_core_list, dim=0)
    cat_role = torch.cat(role_list, dim=0)
    cat_exp = torch.cat(is_exp_list, dim=0)
    cat_bridge = torch.cat(is_bridge_list, dim=0)

    # Pad total nodes and edges to fixed bucket multiples so Apple Metal (MPS)
    # reuses pre-compiled shader graphs instead of recompiling for every batch shape!
    num_real_graphs = len(batch)
    dummy_graph_idx = num_real_graphs  # Extra bucket graph index ignored by loss/outputs
    real_n = cat_nf.size(0)
    pad_n = ((real_n + PAD_NODES_MULTIPLE - 1) // PAD_NODES_MULTIPLE) * PAD_NODES_MULTIPLE - real_n
    if pad_n == 0:
        pad_n = 1  # Always have at least 1 dummy node for dummy edges

    cat_nf = F.pad(cat_nf, (0, 0, 0, pad_n))
    cat_batch = torch.cat([cat_batch, torch.full((pad_n,), dummy_graph_idx, dtype=torch.long)], dim=0)

    real_e = cat_ei.size(1)
    pad_e = ((real_e + PAD_EDGES_MULTIPLE - 1) // PAD_EDGES_MULTIPLE) * PAD_EDGES_MULTIPLE - real_e
    if pad_e > 0:
        dummy_ei = torch.full((2, pad_e), real_n, dtype=torch.long)
        cat_ei = torch.cat([cat_ei, dummy_ei], dim=1)
        cat_ea = F.pad(cat_ea, (0, 0, 0, pad_e))

    return {
        "node_feat": cat_nf,
        "edge_index": cat_ei,
        "edge_attr": cat_ea,
        "batch": cat_batch,
        "real_num_nodes": real_n,
        "real_num_edges": real_e,
        "real_num_graphs": num_real_graphs,
        "is_core_atom": cat_core,
        "register_role_idx": cat_role,
        "is_exposed_atom": cat_exp,
        "is_bridge_edge": cat_bridge,
        "exposed_rule_mask": torch.stack(rule_masks, dim=0),
        "primary_rule_id": torch.tensor(primary_rules, dtype=torch.long),
        "pathway_context_tokens": torch.stack(path_tokens, dim=0),
        "value_target": torch.tensor(val_targets, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# =============================================================================
# SynCat Cross-Attention & PATRO Pathway LSTM + Highway Modules
# =============================================================================

class SynCatSingleHeadAttention(nn.Module):
    """
    Adapted directly from `SynCat/src/attention.py` (`SingleHeadAttention`):
    Computes LayerNorm-stabilized scaled dot-product cross-attention between
    SynKit ITS Rule Queries Q [num_rules, D] and Exposed Molecular Node Keys K [num_nodes, D].
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5
        self.attention_norm = nn.LayerNorm(emb_dim)
        self.linear_q = nn.Linear(emb_dim, emb_dim)
        self.linear_k = nn.Linear(emb_dim, emb_dim)
        self.linear_v = nn.Linear(emb_dim, emb_dim)

    def forward(
        self,
        q_rules: torch.Tensor,
        k_nodes: torch.Tensor,
        node_batch: torch.Tensor,
        num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q_rules: [R, D] rule prototype queries
            k_nodes: [N, D] exposed-weighted node embeddings
            node_batch: [N] graph index per node
            num_graphs: B

        Returns:
            rule_ctx_per_graph: [B, R, D] rule-specific reaction-center embeddings
            last_att_weights: [R, N] cross-attention map over atoms
        """
        q = self.linear_q(self.attention_norm(q_rules)) * self.scale  # [R, D]
        k = self.linear_k(self.attention_norm(k_nodes))               # [N, D]
        v = self.linear_v(k_nodes)                                    # [N, D]

        raw_scores = torch.matmul(q, k.transpose(0, 1))               # [R, N]
        num_rules = q.shape[0]
        dim = v.shape[1]

        # Fully vectorized group-wise softmax across all graphs b in 0..B-1 using scatter_add_ only (no amax synchronization)
        s_clamped = (raw_scores - raw_scores.detach().mean(dim=-1, keepdim=True)).clamp(min=-15.0, max=15.0)
        exp_scores = torch.exp(s_clamped)                             # [R, N]
        idx_2d = node_batch.unsqueeze(0).expand(num_rules, -1)        # [R, N]

        denom_per_g = torch.zeros((num_rules, num_graphs), device=q.device, dtype=q.dtype)
        denom_per_g.scatter_add_(1, idx_2d, exp_scores)
        att_full = exp_scores / (denom_per_g.gather(1, idx_2d) + 1e-12)  # [R, N]

        # Weighted node aggregation per graph and rule: [B, R, D]
        weighted_v = att_full.unsqueeze(-1) * v.unsqueeze(0)          # [R, N, D]
        rule_ctx = torch.zeros((num_rules, num_graphs, dim), device=q.device, dtype=q.dtype)
        idx_3d = node_batch.view(1, -1, 1).expand(num_rules, -1, dim)
        rule_ctx.scatter_add_(1, idx_3d, weighted_v)

        return rule_ctx.transpose(0, 1), att_full


class PATROHighway(nn.Module):
    """
    Adapted directly from `PATRO/ASKCOSv2/retro/template_relevance/models.py` (`Highway`):
    Gates transformed pathway-aware features against carry-through molecular features.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.dense = nn.Linear(in_features, in_features, bias=True)
        self.gate = nn.Linear(in_features, in_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transform = F.elu(self.dense(x))
        transform_gate = torch.sigmoid(self.gate(x))
        return transform * transform_gate + x * (1.0 - transform_gate)


class PATROPathwayEncoder(nn.Module):
    """
    Faithful adaptation of `PATRO/ASKCOSv2/retro/template_relevance/models.py` (`LSTM_Model` + `RxnContext` + `Highway`):
      - Token 0 represents `"na"` (no prior reaction history, mapped to a zero vector via `padding_idx=0`
        exactly like PATRO's `np.zeros((1, fp_size))` in `_gen_reaction_context`).
      - Tokens `1..NUM_ENZYMATIC_RULES` represent `rule_id + 1` of prior reverse reactions peeled along the route,
        ordered oldest-to-most-recent (`rxn_lst[-4:]`).
      - At Step 1 of search (`pathway_tokens = [0, 0, 0, 0]`), the LSTM receives all zeros (no history bias),
        so 100% of the step-1 decision comes from the GAT + `SynCat` Cross-Attention over the molecular graph.
      - At Step t > 1, the LSTM encodes the sequence of previously peeled reactions `ht[-1]` -> `LayerNorm` -> `ReLU`,
        concatenates `torch.cat([prod, rxn_ctx], dim=-1)` as in `RxnContext.forward`, and gates via `PATROHighway`.
    """

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 192) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.fuse_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.highway = PATROHighway(hidden_dim)

    def forward(self, pathway_tokens: torch.Tensor, graph_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pathway_tokens: [B, L] int64 sequence of prior peeled rule tokens (0 = 'na', 1..R = rule_id + 1)
            graph_embed: [B, D] pooled molecular graph embedding (`prod` in PATRO `RxnContext`)

        Returns:
            gated_context: [B, D] pathway-conditioned context vector
        """
        emb = self.token_emb(pathway_tokens)                      # [B, L, D] (0 vector when token == 0)
        _, (ht, _) = self.lstm(emb)                               # ht: [1, B, D]
        rxn_ctx = F.relu(self.layer_norm(self.linear(ht[-1])))    # [B, D] (PATRO LSTM_Model.forward)
        has_history = (pathway_tokens > 0).any(dim=-1, keepdim=True).to(graph_embed.dtype)
        rxn_ctx = rxn_ctx * has_history                           # Strictly 0 at Step 1 (when all tokens are 'na'=0)
        concat_feat = F.elu(self.fuse_proj(torch.cat([graph_embed, rxn_ctx], dim=-1)))
        return self.highway(concat_feat)


# =============================================================================
# 3-Head Reverse-Peeling GNN Architecture
# =============================================================================

def edge_softmax(dst: torch.Tensor, scores: torch.Tensor, num_nodes: int) -> torch.Tensor:
    # Vectorized across all attention heads simultaneously using scatter_add_ (no scatter_reduce_ amax stall on MPS)
    heads = scores.size(1)
    s_clamped = (scores - scores.detach().mean(dim=0, keepdim=True)).clamp(min=-15.0, max=15.0)
    exp_s = torch.exp(s_clamped)                                      # [E, H]
    dst_2d = dst.unsqueeze(1).expand(-1, heads)                       # [E, H]
    denom = torch.zeros((num_nodes, heads), device=scores.device, dtype=scores.dtype)
    denom.scatter_add_(0, dst_2d, exp_s)
    return exp_s / (denom.gather(0, dst_2d) + 1e-12)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim

        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(heads, out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(heads, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.edge_proj = nn.Linear(edge_dim, heads, bias=False)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.edge_proj.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        num_nodes = h.size(0)
        h = h.view(num_nodes, self.heads, self.out_dim)

        att_src = (h * self.att_src).sum(dim=-1)
        att_dst = (h * self.att_dst).sum(dim=-1)

        src, dst = edge_index
        edge_logits = att_src[src] + att_dst[dst] + self.edge_proj(edge_attr)
        edge_logits = F.leaky_relu(edge_logits, 0.2)

        edge_alpha = edge_softmax(dst, edge_logits, num_nodes)
        edge_alpha = F.dropout(edge_alpha, p=self.dropout, training=self.training)

        out = h[src] * edge_alpha.unsqueeze(-1)
        agg = torch.zeros(num_nodes, self.heads, self.out_dim, device=x.device)
        agg.index_add_(0, dst, out)

        return agg.mean(dim=1) + self.bias


class ReversePeelingGNN(nn.Module):
    """
    3-Head Cooperative GNN integrating:
      - C1-Register Parity + 16-step RWSE + LapPE input representation
      - Head 1: Core vs. Decoration Shield (`core_node_head`), Register Role, & Bridge-Bond Segmentation
      - Head 2: `SynCatSingleHeadAttention` Rule-to-Atom Cross-Attention + `PATROPathwayEncoder` Highway Policy
      - Head 3: Distance-to-PKS Value Network `V(G)` & Core Discriminator
    """

    def __init__(
        self,
        node_dim: int = NODE_FEAT_DIM,
        edge_dim: int = EDGE_FEAT_DIM,
        hidden_dim: int = 192,
        heads: int = 4,
        num_layers: int = 3,
        num_rules: int = NUM_ENZYMATIC_RULES,
        num_register_roles: int = NUM_REGISTER_ROLES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.num_layers = num_layers
        self.num_rules = num_rules
        self.num_register_roles = num_register_roles
        self.dropout = dropout

        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(inplace=True),
        )

        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GraphAttentionLayer(hidden_dim, hidden_dim, heads, edge_dim, dropout)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # ── HEAD 1: Core vs. Decoration Shield, Register Parity & Bridge-Bond Segmentation ──
        self.core_node_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.register_node_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim // 2, num_register_roles),
        )
        self.bridge_edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 2, hidden_dim // 2),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ── HEAD 2: Causal Leaf-Peeling Policy with SynCat Cross-Attention & PATRO Highway ──
        self.exposed_node_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + num_register_roles, hidden_dim // 2),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Learnable SynKit ITS Rule Prototype Queries [num_rules, hidden_dim]
        self.rule_prototypes = nn.Parameter(torch.randn(num_rules, hidden_dim) * 0.05)
        self.syncat_cross_attn = SynCatSingleHeadAttention(hidden_dim)
        self.patro_pathway_encoder = PATROPathwayEncoder(vocab_size=64, hidden_dim=hidden_dim)
        self.rule_policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_rules),
        )
        # Per-rule bilinear scorer from SynCat rule-to-node cross-attention
        self.syncat_rule_scorer = nn.Linear(hidden_dim, 1)

        # ── HEAD 3: Distance-to-PKS Value Network V(G) & Nascent Core Discriminator ──
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward_all_heads(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
        bgc_rule_mask: Optional[torch.Tensor] = None,
        pathway_context_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.input_proj(node_feat)

        for gat, norm in zip(self.gat_layers, self.layer_norms):
            residual = x
            x = gat(x, edge_index, edge_attr)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        graph_embed = torch.zeros(num_graphs, self.hidden_dim, device=x.device)
        counts = torch.zeros(num_graphs, device=x.device)
        graph_embed.index_add_(0, batch, x)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
        graph_embed = graph_embed / counts.clamp_min(1.0).unsqueeze(-1)

        g_per_node = graph_embed[batch]
        node_ctx = torch.cat([x, g_per_node], dim=-1)

        # Head 1: Core Shield, Register Role, Bridge Bond
        core_node_logit = self.core_node_head(node_ctx).squeeze(-1)
        p_core = torch.sigmoid(core_node_logit).unsqueeze(-1)
        register_logits = self.register_node_head(node_ctx)
        register_probs = F.softmax(register_logits, dim=-1)

        src, dst = edge_index
        edge_ctx = torch.cat([x[src], x[dst], edge_attr, p_core[src], p_core[dst]], dim=-1)
        bridge_edge_logit = self.bridge_edge_head(edge_ctx).squeeze(-1)

        # Head 2: Exposed-Leaf Node Scoring + SynCat Rule-to-Node Cross-Attention + PATRO Pathway Highway
        peel_node_in = torch.cat([node_ctx, 1.0 - p_core, register_probs], dim=-1)
        exposed_node_logit = self.exposed_node_head(peel_node_in).squeeze(-1)
        exposed_weights = torch.sigmoid(exposed_node_logit).unsqueeze(-1)

        # Exposed non-core node features for SynCat cross-attention
        k_exposed_nodes = x * exposed_weights * (1.0 - p_core * 0.5)
        rule_ctx_per_graph, syncat_att_map = self.syncat_cross_attn(
            self.rule_prototypes, k_exposed_nodes, batch, num_graphs
        )  # [B, R, D], [R, N]
        syncat_rule_delta = self.syncat_rule_scorer(
            rule_ctx_per_graph - self.rule_prototypes.unsqueeze(0)
        ).squeeze(-1)  # [B, R]

        exposed_pool = torch.zeros(num_graphs, self.hidden_dim, device=x.device)
        weight_sum = torch.zeros(num_graphs, 1, device=x.device)
        exposed_pool.index_add_(0, batch, x * exposed_weights)
        weight_sum.index_add_(0, batch, exposed_weights)
        exposed_pool = exposed_pool / weight_sum.clamp_min(1e-6)

        if pathway_context_tokens is None:
            pathway_context_tokens = torch.zeros((num_graphs, 4), dtype=torch.long, device=x.device)
        elif pathway_context_tokens.size(0) < num_graphs:
            pad_g = num_graphs - pathway_context_tokens.size(0)
            pathway_context_tokens = F.pad(pathway_context_tokens, (0, 0, 0, pad_g))
        patro_ctx = self.patro_pathway_encoder(pathway_context_tokens, graph_embed)  # [B, D]

        policy_ctx = torch.cat([graph_embed, exposed_pool, patro_ctx], dim=-1)      # [B, 3*D]
        rule_logits = self.rule_policy_head(policy_ctx) + 0.5 * syncat_rule_delta   # [B, R]
        if bgc_rule_mask is not None:
            if bgc_rule_mask.size(0) < num_graphs:
                bgc_rule_mask = F.pad(bgc_rule_mask, (0, 0, 0, num_graphs - bgc_rule_mask.size(0)), value=1.0)
            rule_logits = rule_logits.masked_fill(bgc_rule_mask <= 0, -1e9)

        # Head 3: Distance-to-PKS Value V(G) & Core Classifier
        val_ctx = torch.cat([graph_embed, exposed_pool], dim=-1)
        value_pred = self.value_head(val_ctx).squeeze(-1)
        cls_logit = self.cls_head(graph_embed)

        return {
            "cls_logit": cls_logit,
            "graph_embed": graph_embed,
            "node_embed": x,
            "core_node_logit": core_node_logit,
            "register_logits": register_logits,
            "bridge_edge_logit": bridge_edge_logit,
            "exposed_node_logit": exposed_node_logit,
            "rule_logits": rule_logits,
            "syncat_att_map": syncat_att_map,
            "value_pred": value_pred,
        }

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward_all_heads(node_feat, edge_index, batch, edge_attr)
        return out["cls_logit"], out["graph_embed"]


SupervisedGNNClassifier = ReversePeelingGNN


# =============================================================================
# Multi-Task Loss & Evaluation
# =============================================================================

def compute_multitask_peeling_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    pos_weight: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = outputs["cls_logit"].device
    rn = int(batch.get("real_num_nodes", outputs["core_node_logit"].size(0)))
    re = int(batch.get("real_num_edges", outputs["bridge_edge_logit"].size(0)))
    rg = int(batch.get("real_num_graphs", outputs["cls_logit"].size(0)))

    labels = batch["labels"].float().unsqueeze(1).to(device)
    loss_cls = F.binary_cross_entropy_with_logits(outputs["cls_logit"][:rg], labels, pos_weight=pos_weight)

    is_core_target = batch["is_core_atom"].to(device)
    loss_core_node = F.binary_cross_entropy_with_logits(outputs["core_node_logit"][:rn], is_core_target)

    reg_target = batch["register_role_idx"].to(device)
    loss_register = F.cross_entropy(outputs["register_logits"][:rn], reg_target)

    bridge_target = batch["is_bridge_edge"].to(device)
    bridge_pos_weight = torch.tensor([8.0], device=device)
    loss_bridge = F.binary_cross_entropy_with_logits(
        outputs["bridge_edge_logit"][:re], bridge_target, pos_weight=bridge_pos_weight
    )

    exposed_target = batch["is_exposed_atom"].to(device)
    exp_pos_weight = torch.tensor([4.0], device=device)
    loss_exposed_node = F.binary_cross_entropy_with_logits(
        outputs["exposed_node_logit"][:rn], exposed_target, pos_weight=exp_pos_weight
    )

    rule_mask_target = batch["exposed_rule_mask"].to(device)
    rule_logits_real = outputs["rule_logits"][:rg]
    loss_rule_multilabel = F.binary_cross_entropy_with_logits(rule_logits_real, rule_mask_target)

    primary_rule = batch["primary_rule_id"].to(device)
    valid_rule_mask = primary_rule >= 0
    if valid_rule_mask.any():
        loss_primary_rule = F.cross_entropy(
            rule_logits_real[valid_rule_mask], primary_rule[valid_rule_mask]
        )
    else:
        loss_primary_rule = torch.tensor(0.0, device=device)

    # Direct Reaction-Center Supervision on `SynCatSingleHeadAttention` (`syncat_att_map`):
    # Train `syncat_att_map[r*, v]` for each graph b's primary rule r* = primary_rule[b]
    # to concentrate its probability mass onto the exposed reaction-center atoms (`is_exposed_atom == 1`)!
    loss_syncat_attn = torch.tensor(0.0, device=device)
    if "syncat_att_map" in outputs and valid_rule_mask.any():
        att_real = outputs["syncat_att_map"][:, :rn]                 # [R, rn]
        batch_real = batch["batch"][:rn].to(device)                  # [rn]
        r_for_node = primary_rule[batch_real]                        # [rn]
        valid_node = r_for_node >= 0
        if valid_node.any():
            r_clamped = r_for_node.clamp_min(0)
            node_att = att_real.gather(0, r_clamped.unsqueeze(0)).squeeze(0)  # [rn]
            exp_att = node_att * exposed_target * valid_node.to(node_att.dtype)
            mass_per_g = torch.zeros(rg, device=device, dtype=node_att.dtype)
            mass_per_g.scatter_add_(0, batch_real, exp_att)
            exp_cnt_per_g = torch.zeros(rg, device=device, dtype=node_att.dtype)
            exp_cnt_per_g.scatter_add_(0, batch_real, exposed_target)
            valid_g = valid_rule_mask & (exp_cnt_per_g > 0)
            if valid_g.any():
                loss_syncat_attn = -torch.log(mass_per_g[valid_g].clamp(min=1e-6, max=1.0)).mean()

    val_target = batch["value_target"].to(device)
    loss_value = F.smooth_l1_loss(outputs["value_pred"][:rg], val_target)

    total_loss = (
        loss_cls
        + 1.0 * (loss_core_node + 0.5 * loss_register + 0.5 * loss_bridge)
        + 1.2 * (loss_exposed_node + loss_rule_multilabel + 0.5 * loss_primary_rule + 0.4 * loss_syncat_attn)
        + 1.0 * loss_value
    )

    metrics = {
        "loss_total": float(total_loss.item()),
        "loss_head1_core": float(loss_core_node.item()),
        "loss_head1_reg": float(loss_register.item()),
        "loss_head2_policy": float((loss_exposed_node + loss_rule_multilabel).item()),
        "loss_syncat_attn": float(loss_syncat_attn.item()),
        "loss_head3_value": float(loss_value.item()),
    }
    return total_loss, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3-Head Reverse-Peeling GNN (C1-Register PE + SynCat + PATRO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default="data/", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="models/supervised_gnn/", help="Checkpoint output directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--pos_weight", type=float, default=4.0, help="Positive class weight for nascent PKS cores")
    parser.add_argument("--hidden_dim", type=int, default=192, help="GAT hidden dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GAT layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def train_epoch(
    model: ReversePeelingGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    all_probs, all_labels = [], []
    core_node_preds, core_node_targets = [], []
    rule_top1_hits, rule_top1_total = 0, 0
    val_abs_err = []

    raw_model = model.module if hasattr(model, "module") else model

    for batch in loader:
        node_feat = batch["node_feat"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        batch_idx = batch["batch"].to(device)
        path_tokens = batch["pathway_context_tokens"].to(device)

        optimizer.zero_grad()
        outputs = raw_model.forward_all_heads(
            node_feat, edge_index, batch_idx, edge_attr, pathway_context_tokens=path_tokens
        )
        loss, _ = compute_multitask_peeling_loss(outputs, batch, pos_weight)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        rn = int(batch.get("real_num_nodes", outputs["core_node_logit"].size(0)))
        rg = int(batch.get("real_num_graphs", outputs["cls_logit"].size(0)))

        all_probs.append(torch.sigmoid(outputs["cls_logit"][:rg]).detach().cpu().numpy())
        all_labels.append(batch["labels"].numpy())

        core_node_preds.append((torch.sigmoid(outputs["core_node_logit"][:rn]).detach().cpu().numpy() >= 0.5))
        core_node_targets.append(batch["is_core_atom"].numpy() >= 0.5)

        prim = batch["primary_rule_id"].to(device)
        valid = prim >= 0
        if valid.any():
            pred_rule = outputs["rule_logits"][:rg][valid].argmax(dim=-1)
            rule_top1_hits += int((pred_rule == prim[valid]).sum().item())
            rule_top1_total += int(valid.sum().item())

        val_abs_err.append(
            torch.abs(outputs["value_pred"][:rg].detach().cpu() - batch["value_target"]).numpy()
        )

    probs = np.concatenate(all_probs).ravel() if all_probs else np.zeros(1)
    labels_np = np.concatenate(all_labels).ravel() if all_labels else np.zeros(1)
    auprc = float(average_precision_score(labels_np, probs)) if len(np.unique(labels_np)) > 1 else 0.0
    core_acc = float(np.mean(np.concatenate(core_node_preds) == np.concatenate(core_node_targets))) if core_node_preds else 0.0
    rule_acc = rule_top1_hits / max(1, rule_top1_total)
    val_mae = float(np.mean(np.concatenate(val_abs_err))) if val_abs_err else 0.0

    return {
        "loss": total_loss / max(1, num_batches),
        "auprc": auprc,
        "core_node_acc": core_acc,
        "rule_top1_acc": rule_acc,
        "value_mae": val_mae,
    }


@torch.no_grad()
def eval_epoch(
    model: ReversePeelingGNN,
    loader: DataLoader,
    pos_weight: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_probs, all_labels = [], []
    core_node_preds, core_node_targets = [], []
    rule_top1_hits, rule_top1_total = 0, 0
    rule_top3_hits = 0
    syncat_rc_recall_hits, syncat_rc_recall_total = 0, 0
    val_abs_err = []

    raw_model = model.module if hasattr(model, "module") else model

    for batch in loader:
        node_feat = batch["node_feat"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        batch_idx = batch["batch"].to(device)
        path_tokens = batch["pathway_context_tokens"].to(device)

        outputs = raw_model.forward_all_heads(
            node_feat, edge_index, batch_idx, edge_attr, pathway_context_tokens=path_tokens
        )
        loss, _ = compute_multitask_peeling_loss(outputs, batch, pos_weight)

        if torch.isfinite(loss):
            total_loss += loss.item()
            num_batches += 1
        rn = int(batch.get("real_num_nodes", outputs["core_node_logit"].size(0)))
        rg = int(batch.get("real_num_graphs", outputs["cls_logit"].size(0)))

        all_probs.append(torch.sigmoid(outputs["cls_logit"][:rg]).cpu().numpy())
        all_labels.append(batch["labels"].numpy())

        core_node_preds.append((torch.sigmoid(outputs["core_node_logit"][:rn]).cpu().numpy() >= 0.5))
        core_node_targets.append(batch["is_core_atom"].numpy() >= 0.5)

        prim = batch["primary_rule_id"].to(device)
        valid = prim >= 0
        if valid.any():
            logits_v = outputs["rule_logits"][:rg][valid]
            pred_top1 = logits_v.argmax(dim=-1)
            rule_top1_hits += int((pred_top1 == prim[valid]).sum().item())
            top3 = torch.topk(logits_v, k=min(3, logits_v.size(-1)), dim=-1).indices
            rule_top3_hits += int((top3 == prim[valid].unsqueeze(-1)).any(dim=-1).sum().item())
            rule_top1_total += int(valid.sum().item())

            # Measure SynCat Reaction-Center Attention Recall@Top-5 atoms per valid graph
            if "syncat_att_map" in outputs:
                att_np = outputs["syncat_att_map"][:, :rn].cpu().numpy()
                b_np = batch["batch"][:rn].numpy()
                exp_np = batch["is_exposed_atom"].numpy()
                prim_np = batch["primary_rule_id"].numpy()
                for g_i in range(rg):
                    r_g = int(prim_np[g_i])
                    if r_g < 0:
                        continue
                    g_mask = np.where(b_np == g_i)[0]
                    exp_g = exp_np[g_mask]
                    if exp_g.sum() == 0:
                        continue
                    att_g = att_np[r_g, g_mask]
                    top5_idx = np.argsort(-att_g)[:5]
                    syncat_rc_recall_total += 1
                    if exp_g[top5_idx].sum() > 0:
                        syncat_rc_recall_hits += 1

        val_abs_err.append(torch.abs(outputs["value_pred"][:rg].cpu() - batch["value_target"]).numpy())

    probs = np.concatenate(all_probs).ravel() if all_probs else np.zeros(1)
    labels_np = np.concatenate(all_labels).ravel() if all_labels else np.zeros(1)
    auprc = float(average_precision_score(labels_np, probs)) if len(np.unique(labels_np)) > 1 else 0.0
    core_acc = float(np.mean(np.concatenate(core_node_preds) == np.concatenate(core_node_targets))) if core_node_preds else 0.0
    rule_acc = rule_top1_hits / max(1, rule_top1_total)
    rule_top3_acc = rule_top3_hits / max(1, rule_top1_total)
    syncat_rc_rec5 = syncat_rc_recall_hits / max(1, syncat_rc_recall_total)
    val_mae = float(np.mean(np.concatenate(val_abs_err))) if val_abs_err else 0.0

    return {
        "loss": total_loss / max(1, num_batches),
        "auprc": auprc,
        "core_node_acc": core_acc,
        "rule_top1_acc": rule_acc,
        "rule_top3_acc": rule_top3_acc,
        "syncat_rc_rec5": syncat_rc_rec5,
        "value_mae": val_mae,
    }


def get_compute_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = get_compute_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train" / "supcon_train.parquet"
    val_path = data_dir / "val" / "supcon_val.parquet"

    print("=" * 88)
    print("TRAINING 3-HEAD REVERSE-PEELING GNN (C1-Register PE + SynCat Cross-Attn + PATRO Highway)")
    print("=" * 88)
    print(f"Device: {device} | Node Feat Dim: {NODE_FEAT_DIM} (C1-Reg={C1_REG_DIM}, RWSE={RWSE_STEPS}, LapPE={LAP_PE_DIM}) | Edge Feat Dim: {EDGE_FEAT_DIM}")

    train_ds = PeelingTrajectoryDataset(str(train_path), max_samples=args.max_train_samples)
    val_ds = PeelingTrajectoryDataset(str(val_path))
    print(f"Loaded {len(train_ds):,} train states and {len(val_ds):,} val states")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_peeling_graphs,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_peeling_graphs,
        num_workers=args.num_workers,
    )

    model = ReversePeelingGNN(
        node_dim=NODE_FEAT_DIM,
        edge_dim=EDGE_FEAT_DIM,
        hidden_dim=args.hidden_dim,
        heads=args.num_heads,
        num_layers=args.num_layers,
        num_rules=NUM_ENZYMATIC_RULES,
        num_register_roles=NUM_REGISTER_ROLES,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    pos_weight = torch.tensor([args.pos_weight], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(
        f"{'Epoch':>5} | {'TrLoss':>7} | {'VaLoss':>7} | {'Head1 CoreAcc':>13} | "
        f"{'Head2 Top1/Top3':>15} | {'SynCat RC@5':>11} | {'Head3 ValMAE':>12} | {'Core AUPRC':>10} | {'Best':>4}"
    )
    print("-" * 106)

    best_val_loss = float("inf")
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, pos_weight, device)
        va = eval_epoch(model, val_loader, pos_weight, device)
        scheduler.step()

        is_best = va["loss"] < best_val_loss
        if is_best:
            best_val_loss = va["loss"]

        print(
            f"{epoch:>5d} | {tr['loss']:>7.4f} | {va['loss']:>7.4f} | "
            f"{100*va['core_node_acc']:>12.2f}% | "
            f"{100*va['rule_top1_acc']:>6.1f}%/{100*va['rule_top3_acc']:>5.1f}% | "
            f"{100*va['syncat_rc_rec5']:>10.1f}% | "
            f"{va['value_mae']:>12.4f} | {va['auprc']:>10.4f} | {'*' if is_best else '':>4}"
        )

        history_rows.append({"epoch": epoch, "train": tr, "val": va})

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "train_auprc": tr["auprc"],
            "val_auprc": va["auprc"],
            "val_core_node_acc": va["core_node_acc"],
            "val_rule_top1_acc": va["rule_top1_acc"],
            "val_rule_top3_acc": va["rule_top3_acc"],
            "val_value_mae": va["value_mae"],
            "best_val_loss": best_val_loss,
            "node_dim": NODE_FEAT_DIM,
            "edge_dim": EDGE_FEAT_DIM,
            "num_rules": NUM_ENZYMATIC_RULES,
            "num_register_roles": NUM_REGISTER_ROLES,
            "args": vars(args),
        }

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(ckpt, output_dir / f"checkpoint_epoch_{epoch:03d}.pt")
        if is_best:
            torch.save(ckpt, output_dir / "best_model.pt")

    with open(output_dir / "peeling_gnn_training_metrics.json", "w") as f:
        json.dump(history_rows, f, indent=2)

    print("\nTraining complete! Best checkpoint saved to:", output_dir / "best_model.pt")


if __name__ == "__main__":
    main()
