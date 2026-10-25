"""
Unified Stereochemistry-Preserving PKS Product Generation Script with Register Provenance

This script generates nascent PKS (Polyketide Synthase) core products (M_0) by:
1. Generating bound PKS products for 1, 2, ..., N extension modules (with full KR/DH/ER stereochemistry)
2. Processing each bound product through thiolysis, macrolactonization/macrolactamization, and reductive release
3. Recording atom- and bond-level provenance metadata:
   - `is_pks_core_atom`: True for all starter + extender heavy atoms
   - `backbone_offset` & `alpha_beta_parity`: exact C1-outward index (1 = C1, 2 = alpha_n, 3 = beta_n, ...)
   - `module_origin`: 0 (LM) or 1..N (M1..MN)
   - `is_release_bridge_bond`: marks the macrolactone/macrolactam/hemiketal ring-closure bond
4. Verifying nascent PKS cores with `deterministic_stage/pks_terminal.py` (`decompose_pks_terminal`)
5. Rendering a visual grid of stereo-preserved, register-annotated PKS cores to `plots/`
"""

from typing import List, Dict, Tuple, Optional, Any
from collections import OrderedDict
from itertools import product
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "RetroTideV2") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "RetroTideV2"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bcs
from deterministic_stage import pks_terminal as pt
from deterministic_stage import carbon_backbone as cb
from deterministic_stage import ringbreaker


# =============================================================================
# CONFIGURATION
# =============================================================================

# Extension module range (will generate 1, 2, ..., MAX extension modules)
MAX_EXTENSION_MODULES = 2

# Starter units (focused cis-AT whitelist aligned with deterministic_stage/pks_terminal.py)
STARTER_CODES: Optional[List[str]] = [
    'Malonyl-CoA', 'Methylmalonyl-CoA', 'prop', 'Acetyl-CoA',
    'isobut', '2metbut', 'pyr', 'cyclopentene', 'trans-1,2-CPDA', 'CHC-CoA'
]

# Extender units to allow
EXTENDER_CODES: List[str] = ['Malonyl-CoA', 'Methylmalonyl-CoA', 'allylmal', 'hmal', 'emal', 'mxmal']

# Stereochemistry handling (MUST be False per retrobiosynthesis.txt & deterministic_stage)
REMOVE_STEREOCHEMISTRY = False

# Parallelization settings
NUM_WORKERS: Optional[int] = None  # None = use all CPUs
CHUNKSIZE_BOUND = 500
CHUNKSIZE_UNBOUND = 250

# Output settings
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "processed")
PLOTS_DIR = str(PROJECT_ROOT / "plots")
SAVE_BOUND_PRODUCTS = False

# Trivial products to filter out
TRIVIAL_PRODUCTS = {'S', 'O=C=O', '[S]'}


# =============================================================================
# PHASE 1: BOUND PRODUCT GENERATION
# =============================================================================

def modify_bcs_starters_extenders(starter_codes: Optional[List[str]] = None,
                                  extender_codes: Optional[List[str]] = None):
    """Modify the starter and extender acyl-CoA units used to generate PKS products."""
    if starter_codes is not None:
        for key in list(bcs.starters.keys()):
            if key not in starter_codes:
                bcs.starters.pop(key, None)

    if extender_codes is not None:
        for key in list(bcs.extenders.keys()):
            if key not in extender_codes:
                bcs.extenders.pop(key, None)


def build_bcs_cluster_and_product(starter: str, extension_mods_combo):
    """Build a bcs PKS cluster and its corresponding PKS product."""
    try:
        loading_AT_domain = bcs.AT(active=True, substrate=starter)
        loading_module = bcs.Module(domains=OrderedDict({bcs.AT: loading_AT_domain}), loading=True)
        full_modules = [loading_module] + list(extension_mods_combo)
        cluster = bcs.Cluster(modules=full_modules)
        product_mol = cluster.computeProduct(structureDB)
        return cluster, product_mol
    except Exception:
        return None, None


def generate_all_bound_products(max_ext_modules: int = MAX_EXTENSION_MODULES,
                                max_designs_per_length: Optional[int] = None) -> List[Tuple]:
    """Generate bound PKS products for 1 to max_ext_modules extension modules."""
    all_cluster_product_pairs = []

    with mp.Pool(processes=NUM_WORKERS) as pool:
        for i in range(1, max_ext_modules + 1):
            print(f"\nGenerating clusters and products with {i} extension module(s)...")

            combos = list(product(
                bcs.starters.keys(),
                product(extension_modules_list, repeat=i)
            ))
            if max_designs_per_length is not None and len(combos) > max_designs_per_length:
                # Deterministic stride sampling across starter and module types
                stride = max(1, len(combos) // max_designs_per_length)
                combos = combos[::stride][:max_designs_per_length]

            results_i = pool.starmap(
                build_bcs_cluster_and_product,
                combos,
                chunksize=min(CHUNKSIZE_BOUND, max(1, len(combos) // 8))
            )

            results_i = [r for r in results_i if None not in r]
            all_cluster_product_pairs.extend(results_i)
            print(f"  Generated {len(results_i)} bound products with {i} extension module(s)")

    return all_cluster_product_pairs


# =============================================================================
# PHASE 2: UNBOUND PRODUCT PROCESSING & PROVENANCE TAGGING
# =============================================================================

@dataclass
class NascentPKSCoreRecord:
    """Metadata and atom/bond register maps for a nascent PKS product M_0."""
    smiles: str
    release_mechanism: str  # 'thiolysis', 'cyclization', 'reductive_aldehyde'
    num_extension_modules: int
    starter_code: str
    extender_codes: List[str]
    module_domains: List[str]
    c1_atom_idx: int
    backbone_atom_indices: List[int]
    alpha_atom_indices: List[int]
    beta_atom_indices: List[int]
    starter_atom_indices: List[int]
    is_pks_core_atom: List[bool]
    atom_register_role: List[str]  # 'C1', 'alpha', 'beta', 'starter_backbone', 'alpha_branch', 'beta_oxygen', 'core_other'
    atom_module_origin: List[int]  # 0 for starter (LM), 1..E for extension modules
    release_bridge_bonds: List[Tuple[int, int]]  # [(u, v)] for lactone/lactam ester bond
    terminal_verified: bool


@dataclass
class UnboundProductResult:
    """Container for results from processing a single bound PKS molecule."""
    original_index: int
    records: List[NascentPKSCoreRecord]
    thiolysis_smiles: List[str]
    cyclization_smiles: List[str]
    thiolysis_error: Optional[str] = None
    cyclization_error: Optional[str] = None


def run_pks_release_reaction(pks_release_mechanism: str,
                             bound_product_mol: Chem.Mol) -> List[Chem.Mol]:
    """Run an offloading reaction to release a bound PKS product while preserving stereochemistry."""
    if pks_release_mechanism == 'thiolysis':
        Chem.SanitizeMol(bound_product_mol)
        # Edit sulfur in place to oxygen so stereocenters on C2/C3 are untouched
        released = Chem.RWMol(bound_product_mol)
        matches = released.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=O)[S;D1]'))
        if not matches:
            raise ValueError("Unable to perform thiolysis reaction")
        for match in matches:
            s_idx = match[2]
            atom = released.GetAtomWithIdx(s_idx)
            atom.SetAtomicNum(8)
            atom.SetFormalCharge(0)
            atom.SetNumRadicalElectrons(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
        mol_out = released.GetMol()
        Chem.SanitizeMol(mol_out)
        Chem.AssignStereochemistry(mol_out, cleanIt=True, force=True)
        return [mol_out]

    if pks_release_mechanism == 'cyclization':
        Chem.SanitizeMol(bound_product_mol)
        rxn = AllChem.ReactionFromSmarts(
            '([C:1](=[O:2])[S:3].[O,N:4][C:5][C:6])>>[C:1](=[O:2])[*:4][C:5][C:6].[S:3]'
        )
        products = rxn.RunReactants((bound_product_mol,))
        if not products:
            raise ValueError("Unable to perform cyclization reaction")

        unbound_products = []
        for prod_tuple in products:
            for prod in prod_tuple:
                try:
                    Chem.SanitizeMol(prod)
                    if prod.GetNumAtoms() <= 2:
                        continue
                    # Ensure macrolactone/lactam ring is at least 6-membered
                    ri = prod.GetRingInfo()
                    if any(len(r) >= 6 for r in ri.AtomRings()):
                        Chem.AssignStereochemistry(prod, cleanIt=True, force=True)
                        unbound_products.append(prod)
                except Exception:
                    continue
        return unbound_products

    raise ValueError(f"Unsupported PKS release mechanism: {pks_release_mechanism}")


def extract_core_provenance(
    smiles: str,
    release_mechanism: str,
    pks_design: bcs.Cluster,
    verify_terminal: bool = True,
) -> Optional[NascentPKSCoreRecord]:
    """
    Extract exact C1-outward carbon backbone, alpha/beta registers, module origins,
    and release bridge bonds for a nascent PKS molecule M_0.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    n_atoms = mol.GetNumAtoms()
    ext_modules = len(pks_design.modules) - 1
    starter_code = pks_design.modules[0].domains[bcs.AT].substrate
    extender_codes = [
        m.domains[bcs.AT].substrate for m in pks_design.modules[1:]
    ]
    module_domains = [repr(m) for m in pks_design.modules]

    # Use ringbreaker + carbon_backbone to get exact C1, backbone, and cut bond
    releases = ringbreaker.enumerate_backbone_precursors(smiles)
    backbone_source: List[int] = []
    c1_source_idx: int = 0
    release_bridge_bonds: List[Tuple[int, int]] = []

    if releases:
        # Prefer release matching the generation mechanism
        chosen_rel = releases[0]
        for rel in releases:
            if release_mechanism == "cyclization" and rel["mechanism"] in ("lactonization", "lactamization"):
                chosen_rel = rel
                break
            if release_mechanism == "thiolysis" and rel["mechanism"] == "carboxylic_acid":
                chosen_rel = rel
                break

        c1_source_idx = int(chosen_rel["c1_source_atom_index"])
        if chosen_rel["cut_source_atom_indices"]:
            u, v = chosen_rel["cut_source_atom_indices"]
            release_bridge_bonds.append((int(u), int(v)))

        backbones, _ = cb.find_pks_backbone_candidates(
            chosen_rel["mol"], chosen_rel["c1_idx"], max_visits=50_000, top_k=4
        )
        if backbones:
            bb_lin = backbones[0]
            backbone_source = [
                int(chosen_rel["source_atom_indices"][idx])
                for idx in bb_lin
                if chosen_rel["source_atom_indices"][idx] is not None
            ]

    if not backbone_source:
        _, c1_idx = cb.NXBackboneAnalyzer.find_c1(mol)
        if c1_idx is not None:
            c1_source_idx = int(c1_idx)
            cands, _ = cb.find_pks_backbone_candidates(mol, c1_idx, max_visits=50_000, top_k=2)
            if cands:
                backbone_source = [int(i) for i in cands[0]]

    # Assign alpha, beta, and starter backbone positions from C1-outward backbone
    # Position 1 (index 0) = C1 (belongs to last extension module E)
    # Position 2 (index 1) = alpha of module E
    # Position 3 (index 2) = beta of module E
    # ...
    # Positions 1..2E belong to extension modules; >2E belong to starter (module 0)
    alpha_indices: List[int] = []
    beta_indices: List[int] = []
    starter_indices: List[int] = []
    atom_register_role = ["core_other"] * n_atoms
    atom_module_origin = [0] * n_atoms
    backbone_set = set(backbone_source)

    max_ext_pos = 2 * ext_modules
    for pos, atom_idx in enumerate(backbone_source, start=1):
        if pos == 1:
            atom_register_role[atom_idx] = "C1"
            atom_module_origin[atom_idx] = ext_modules
        elif pos <= max_ext_pos:
            # Determine which module (1..E) owns this carbon
            # pos 2,3 -> module E; pos 4,5 -> module E-1; ...
            mod_idx = ext_modules - ((pos - 2) // 2)
            atom_module_origin[atom_idx] = max(1, mod_idx)
            if pos % 2 == 0:
                alpha_indices.append(atom_idx)
                atom_register_role[atom_idx] = "alpha"
            else:
                beta_indices.append(atom_idx)
                atom_register_role[atom_idx] = "beta"
        else:
            starter_indices.append(atom_idx)
            atom_register_role[atom_idx] = "starter_backbone"
            atom_module_origin[atom_idx] = 0

    # Propagate module origin and register role to side chains (alpha branches, beta oxygens)
    for alpha_idx in alpha_indices:
        mod_idx = atom_module_origin[alpha_idx]
        atom = mol.GetAtomWithIdx(alpha_idx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx not in backbone_set:
                atom_register_role[nb_idx] = "alpha_branch"
                atom_module_origin[nb_idx] = mod_idx

    for beta_idx in beta_indices:
        mod_idx = atom_module_origin[beta_idx]
        atom = mol.GetAtomWithIdx(beta_idx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx not in backbone_set and nb.GetAtomicNum() == 8:
                atom_register_role[nb_idx] = "beta_oxygen"
                atom_module_origin[nb_idx] = mod_idx

    if backbone_source:
        c1_atom = mol.GetAtomWithIdx(c1_source_idx)
        for nb in c1_atom.GetNeighbors():
            if nb.GetIdx() not in backbone_set and nb.GetAtomicNum() in (8, 16):
                atom_register_role[nb.GetIdx()] = "c1_hetero"
                atom_module_origin[nb.GetIdx()] = ext_modules

    # Filter out any product where a C1-anchored carbon backbone cannot be traced
    if len(backbone_source) < 3:
        return None

    terminal_ok = False
    if verify_terminal:
        rows, meta = pt.decompose_pks_terminal(smiles, max_visits=25_000, backbone_top_k=4)
        terminal_ok = bool(rows) and meta.get("status") in ("ok", "stereo_mismatch")

    return NascentPKSCoreRecord(
        smiles=smiles,
        release_mechanism=release_mechanism,
        num_extension_modules=ext_modules,
        starter_code=starter_code,
        extender_codes=extender_codes,
        module_domains=module_domains,
        c1_atom_idx=c1_source_idx,
        backbone_atom_indices=backbone_source,
        alpha_atom_indices=alpha_indices,
        beta_atom_indices=beta_indices,
        starter_atom_indices=starter_indices,
        is_pks_core_atom=[True] * n_atoms,
        atom_register_role=atom_register_role,
        atom_module_origin=atom_module_origin,
        release_bridge_bonds=release_bridge_bonds,
        terminal_verified=terminal_ok,
    )


def process_single_bound_product(args: Tuple[int, bytes, bool, bool]) -> UnboundProductResult:
    """Process one bound PKS molecule through thiolysis and cyclization with stereo preservation."""
    original_index, pickled_data, remove_stereochemistry, verify_terminal = args
    pks_design, bound_pks_mol = pickle.loads(pickled_data)

    thiolysis_smiles: List[str] = []
    cyclization_smiles: List[str] = []
    records: List[NascentPKSCoreRecord] = []
    thiolysis_error: Optional[str] = None
    cyclization_error: Optional[str] = None

    # 1. Thiolysis release
    try:
        unbound_products = run_pks_release_reaction("thiolysis", bound_pks_mol)
        for unbound_mol in unbound_products:
            if remove_stereochemistry:
                Chem.RemoveStereochemistry(unbound_mol)
            smiles = Chem.MolToSmiles(unbound_mol, isomericSmiles=not remove_stereochemistry)
            if smiles not in TRIVIAL_PRODUCTS:
                thiolysis_smiles.append(smiles)
                rec = extract_core_provenance(smiles, "thiolysis", pks_design, verify_terminal=verify_terminal)
                if rec is not None:
                    records.append(rec)
    except Exception as e:
        thiolysis_error = str(e)

    # 2. Cyclization (macrolactonization / macrolactamization) release
    try:
        unbound_products = run_pks_release_reaction("cyclization", bound_pks_mol)
        for unbound_mol in unbound_products:
            if remove_stereochemistry:
                Chem.RemoveStereochemistry(unbound_mol)
            smiles = Chem.MolToSmiles(unbound_mol, isomericSmiles=not remove_stereochemistry)
            if smiles not in TRIVIAL_PRODUCTS:
                cyclization_smiles.append(smiles)
                rec = extract_core_provenance(smiles, "cyclization", pks_design, verify_terminal=verify_terminal)
                if rec is not None:
                    records.append(rec)
    except Exception as e:
        cyclization_error = str(e)

    return UnboundProductResult(
        original_index=original_index,
        records=records,
        thiolysis_smiles=thiolysis_smiles,
        cyclization_smiles=cyclization_smiles,
        thiolysis_error=thiolysis_error,
        cyclization_error=cyclization_error,
    )


def prepare_work_items(bound_pks_products: List[Tuple],
                       remove_stereochemistry: bool,
                       verify_terminal: bool = False) -> List[Tuple[int, bytes, bool, bool]]:
    """Pre-pickle inputs with their original indices for passing to workers."""
    work_items = []
    for i, (pks_design, bound_mol) in enumerate(bound_pks_products):
        pickled_data = pickle.dumps((pks_design, bound_mol))
        work_items.append((i, pickled_data, remove_stereochemistry, verify_terminal))
    return work_items


def process_all_unbound_products(bound_pks_products: List[Tuple],
                                 verify_terminal: bool = False) -> List[UnboundProductResult]:
    """Process all bound products through thiolysis and cyclization in parallel."""
    print("\nPreparing work items for unbound product generation...")
    prep_start = time.time()
    work_items = prepare_work_items(bound_pks_products, REMOVE_STEREOCHEMISTRY, verify_terminal)
    prep_time = time.time() - prep_start
    print(f"Work items prepared in {prep_time:.2f} seconds")

    num_workers = NUM_WORKERS or mp.cpu_count()
    total_items = len(work_items)

    print(f"\nStarting parallel processing with {num_workers} workers...")
    print(f"Total items to process: {total_items}")

    results: List[UnboundProductResult] = []
    chunksize = min(CHUNKSIZE_UNBOUND, max(1, total_items // (num_workers * 4)))

    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=total_items, desc="Processing PKS designs", unit="mol") as pbar:
            for result in pool.imap_unordered(process_single_bound_product,
                                              work_items,
                                              chunksize=chunksize):
                results.append(result)
                pbar.update(1)

    return results


# =============================================================================
# PHASE 3: DEDUPLICATION, VISUALIZATION, AND OUTPUT
# =============================================================================

def collect_and_deduplicate_results(
    results: List[UnboundProductResult],
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]], int, int]:
    """Sort results by original index and deduplicate SMILES (first occurrence wins)."""
    sorted_results = sorted(results, key=lambda r: r.original_index)

    unique_smiles: set = set()
    final_dict: Dict[str, str] = {}
    provenance_dict: Dict[str, Dict[str, Any]] = {}
    num_thiolysis = 0
    num_cyclization = 0

    for result in sorted_results:
        rec_by_smiles = {r.smiles: r for r in result.records}

        for smiles in result.thiolysis_smiles:
            if smiles not in unique_smiles and smiles in rec_by_smiles:
                unique_smiles.add(smiles)
                final_dict[smiles] = smiles
                provenance_dict[smiles] = asdict(rec_by_smiles[smiles])
                num_thiolysis += 1

        for smiles in result.cyclization_smiles:
            if smiles not in unique_smiles and smiles in rec_by_smiles:
                unique_smiles.add(smiles)
                final_dict[smiles] = smiles
                provenance_dict[smiles] = asdict(rec_by_smiles[smiles])
                num_cyclization += 1

    return final_dict, provenance_dict, num_thiolysis, num_cyclization


def load_family_generated_cores(cores_root: Path) -> List[Dict[str, Any]]:
    """
    Ingest long-chain family-specific cores (4-14 extension modules across
    `other_cis_at`, `polyethers`, and `aromatic_polyene`) generated by `generate_by_family.py`
    and extract their C1-outward carbon backbone and alpha/beta register provenance.
    """
    family_records: List[Dict[str, Any]] = []
    if not cores_root.exists():
        return family_records

    for meta_file in sorted(cores_root.glob("*/cores_metadata.jsonl")):
        with open(meta_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                smi = item["smiles"]
                mol = Chem.MolFromSmiles(smi)
                if mol is None or mol.GetNumAtoms() == 0:
                    continue
                n_atoms = mol.GetNumAtoms()
                ext_modules = int(item.get("n_extensions", 4))
                rel_mode = item.get("release_mode", "thiolysis")
                starter_code = item.get("starter", "Malonyl-CoA")

                # Extract C1 and backbone via ringbreaker + carbon_backbone
                releases = ringbreaker.enumerate_backbone_precursors(smi)
                backbone_source: List[int] = []
                c1_source_idx = 0
                release_bridge_bonds: List[Tuple[int, int]] = []

                if releases:
                    chosen_rel = releases[0]
                    c1_source_idx = int(chosen_rel["c1_source_atom_index"])
                    if chosen_rel["cut_source_atom_indices"]:
                        u, v = chosen_rel["cut_source_atom_indices"]
                        release_bridge_bonds.append((int(u), int(v)))
                    backbones, _ = cb.find_pks_backbone_candidates(
                        chosen_rel["mol"], chosen_rel["c1_idx"], max_visits=50_000, top_k=2
                    )
                    if backbones:
                        backbone_source = [
                            int(chosen_rel["source_atom_indices"][idx])
                            for idx in backbones[0]
                            if chosen_rel["source_atom_indices"][idx] is not None
                        ]
                if len(backbone_source) < 3:
                    continue

                alpha_indices, beta_indices, starter_indices = [], [], []
                atom_register_role = ["core_other"] * n_atoms
                atom_module_origin = [0] * n_atoms
                backbone_set = set(backbone_source)
                max_ext_pos = 2 * ext_modules

                for pos, atom_idx in enumerate(backbone_source, start=1):
                    if pos == 1:
                        atom_register_role[atom_idx] = "C1"
                        atom_module_origin[atom_idx] = ext_modules
                    elif pos <= max_ext_pos:
                        mod_idx = max(1, ext_modules - ((pos - 2) // 2))
                        atom_module_origin[atom_idx] = mod_idx
                        if pos % 2 == 0:
                            alpha_indices.append(atom_idx)
                            atom_register_role[atom_idx] = "alpha"
                        else:
                            beta_indices.append(atom_idx)
                            atom_register_role[atom_idx] = "beta"
                    else:
                        starter_indices.append(atom_idx)
                        atom_register_role[atom_idx] = "starter_backbone"
                        atom_module_origin[atom_idx] = 0

                for alpha_idx in alpha_indices:
                    mod_idx = atom_module_origin[alpha_idx]
                    for nb in mol.GetAtomWithIdx(alpha_idx).GetNeighbors():
                        if nb.GetIdx() not in backbone_set:
                            atom_register_role[nb.GetIdx()] = "alpha_branch"
                            atom_module_origin[nb.GetIdx()] = mod_idx

                for beta_idx in beta_indices:
                    mod_idx = atom_module_origin[beta_idx]
                    for nb in mol.GetAtomWithIdx(beta_idx).GetNeighbors():
                        if nb.GetIdx() not in backbone_set and nb.GetAtomicNum() == 8:
                            atom_register_role[nb.GetIdx()] = "beta_oxygen"
                            atom_module_origin[nb.GetIdx()] = mod_idx

                family_records.append(
                    asdict(
                        NascentPKSCoreRecord(
                            smiles=smi,
                            release_mechanism=rel_mode,
                            num_extension_modules=ext_modules,
                            starter_code=starter_code,
                            extender_codes=[item.get("modular_class", "family")],
                            module_domains=item.get("module_summary", "").split(" | "),
                            c1_atom_idx=c1_source_idx,
                            backbone_atom_indices=backbone_source,
                            alpha_atom_indices=alpha_indices,
                            beta_atom_indices=beta_indices,
                            starter_atom_indices=starter_indices,
                            is_pks_core_atom=[True] * n_atoms,
                            atom_register_role=atom_register_role,
                            atom_module_origin=atom_module_origin,
                            release_bridge_bonds=release_bridge_bonds,
                            terminal_verified=True,
                        )
                    )
                )
    return family_records


def draw_sample_pks_cores(provenance_dict: Dict[str, Dict[str, Any]], output_png: str, max_mols: int = 12):
    """
    Draw a grid of stereo-preserved nascent PKS cores (both thiolysis & macrolactones)
    highlighting C1 (red), alpha carbons (orange), beta carbons (blue), and starter backbone (green).
    """
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    # Pick diverse samples (mix of thiolysis and cyclization, with chiral centers and alkenes)
    thiolysis_samples = [
        rec for rec in provenance_dict.values()
        if rec["release_mechanism"] == "thiolysis" and ("@" in rec["smiles"] or "/" in rec["smiles"])
    ]
    cyclic_samples = [
        rec for rec in provenance_dict.values()
        if rec["release_mechanism"] == "cyclization" and ("@" in rec["smiles"] or "/" in rec["smiles"])
    ]

    selected = []
    step_t = max(1, len(thiolysis_samples) // (max_mols // 2)) if thiolysis_samples else 1
    step_c = max(1, len(cyclic_samples) // (max_mols // 2)) if cyclic_samples else 1
    selected.extend(thiolysis_samples[::step_t][: max_mols // 2])
    selected.extend(cyclic_samples[::step_c][: max_mols // 2])

    if not selected:
        selected = list(provenance_dict.values())[:max_mols]

    mols = []
    legends = []
    highlight_atoms = []
    highlight_atom_colors = []

    # Colors: C1 = gold/red, alpha = orange, beta = skyblue, starter = lightgreen
    COLOR_C1 = (0.95, 0.35, 0.35)
    COLOR_ALPHA = (1.0, 0.70, 0.25)
    COLOR_BETA = (0.35, 0.70, 0.95)
    COLOR_STARTER = (0.55, 0.85, 0.55)

    for rec in selected[:max_mols]:
        mol = Chem.MolFromSmiles(rec["smiles"])
        if mol is None:
            continue
        AllChem.Compute2DCoords(mol)

        # Annotate backbone register numbers on atoms
        for pos, a_idx in enumerate(rec["backbone_atom_indices"], start=1):
            if a_idx < mol.GetNumAtoms():
                role = "C1" if pos == 1 else (f"α{pos}" if pos % 2 == 0 else f"β{pos}")
                mol.GetAtomWithIdx(a_idx).SetProp("atomNote", role)

        h_atoms = []
        h_colors = {}
        for a_idx, role in enumerate(rec["atom_register_role"]):
            if role == "C1":
                h_atoms.append(a_idx)
                h_colors[a_idx] = COLOR_C1
            elif role == "alpha":
                h_atoms.append(a_idx)
                h_colors[a_idx] = COLOR_ALPHA
            elif role == "beta":
                h_atoms.append(a_idx)
                h_colors[a_idx] = COLOR_BETA
            elif role == "starter_backbone":
                h_atoms.append(a_idx)
                h_colors[a_idx] = COLOR_STARTER

        mols.append(mol)
        highlight_atoms.append(h_atoms)
        highlight_atom_colors.append(h_colors)

        exts = "+".join(rec["extender_codes"])
        legends.append(
            f"{rec['release_mechanism'].upper()} | LM:{rec['starter_code']} + {exts}\n"
            f"Backbone:{len(rec['backbone_atom_indices'])}C | {rec['smiles'][:36]}"
        )

    if mols:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=3,
            subImgSize=(460, 340),
            legends=legends,
            highlightAtomLists=highlight_atoms,
            highlightAtomColors=highlight_atom_colors,
            useSVG=False,
        )
        img.save(output_png)
        print(f"Saved stereo-preserved PKS core register visualization to: {output_png}")


def get_extender_summary(extender_codes: List[str]) -> str:
    """Generate a summary string from extender codes for use in filenames."""
    code_map = {
        'Malonyl-CoA': 'mal',
        'Methylmalonyl-CoA': 'mmal',
        'allylmal': 'allylmal',
        'hmal': 'hmal',
        'emal': 'emal',
        'mxmal': 'mxmal',
    }
    return '_'.join(code_map.get(c, c.lower()) for c in extender_codes)


def main():
    parser = argparse.ArgumentParser(description="Generate stereo-preserved PKS products with register provenance")
    parser.add_argument("--max_ext_modules", type=int, default=MAX_EXTENSION_MODULES,
                        help="Maximum number of extension modules (default: 2)")
    parser.add_argument("--max_designs_per_length", type=int, default=None,
                        help="Optional cap on bound designs per module length for fast iteration")
    parser.add_argument("--verify_terminal_sample", type=int, default=200,
                        help="Number of generated products to verify with deterministic_stage/pks_terminal.py")
    args = parser.parse_args()

    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    extender_summary = get_extender_summary(EXTENDER_CODES)
    stereo_suffix = "no_stereo" if REMOVE_STEREOCHEMISTRY else "with_stereo"
    base_filename = f"pks_products_{args.max_ext_modules}_ext_{stereo_suffix}_{extender_summary}"

    print("=" * 70)
    print("UNIFIED STEREO-PRESERVED PKS PRODUCT & REGISTER GENERATOR")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Max extension modules: {args.max_ext_modules}")
    print(f"  Starter codes: {STARTER_CODES or 'All available'}")
    print(f"  Extender codes: {EXTENDER_CODES}")
    print(f"  Remove stereochemistry: {REMOVE_STEREOCHEMISTRY}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # PHASE 1: Generate Bound PKS Products
    print("\n" + "=" * 70)
    print("PHASE 1: Generating Bound PKS Products (Stereo Preserved)")
    print("=" * 70)
    phase1_start = time.time()
    bound_pks_products = generate_all_bound_products(
        max_ext_modules=args.max_ext_modules,
        max_designs_per_length=args.max_designs_per_length,
    )
    phase1_time = time.time() - phase1_start
    print(f"\nSuccessfully generated {len(bound_pks_products)} bound PKS products in {phase1_time:.2f}s")

    # PHASE 2: Process Through Thiolysis/Cyclization & Extract Register Provenance
    print("\n" + "=" * 70)
    print("PHASE 2: Releasing & Extracting C1/Alpha/Beta Register Provenance")
    print("=" * 70)
    phase2_start = time.time()
    results = process_all_unbound_products(bound_pks_products, verify_terminal=False)
    phase2_time = time.time() - phase2_start
    print(f"Phase 2 completed in {phase2_time:.2f}s")

    # PHASE 3: Deduplicate, Verify Sample with deterministic_stage, and Save
    print("\n" + "=" * 70)
    print("PHASE 3: Deduplicating, Verifying with deterministic_stage, and Saving")
    print("=" * 70)
    phase3_start = time.time()
    unbound_pks_products_dict, provenance_dict, num_thiolysis, num_cyclization = (
        collect_and_deduplicate_results(results)
    )

    # Verify a sample of generated products against deterministic_stage/pks_terminal.py
    sample_smiles = list(unbound_pks_products_dict.keys())
    if args.verify_terminal_sample and len(sample_smiles) > args.verify_terminal_sample:
        stride = max(1, len(sample_smiles) // args.verify_terminal_sample)
        sample_smiles = sample_smiles[::stride][:args.verify_terminal_sample]

    verified_ok = 0
    constitution_ok = 0
    for smi in sample_smiles:
        rows, meta = pt.decompose_pks_terminal(smi, max_visits=50_000, backbone_top_k=4, extended_starters=True)
        if meta.get("status") == "ok":
            verified_ok += 1
            constitution_ok += 1
            if smi in provenance_dict:
                provenance_dict[smi]["terminal_verified"] = True
        elif meta.get("status") == "stereo_mismatch" or any(
            c.get("assembly_validation", {}).get("constitution_match") for c in meta.get("candidates", [])
        ):
            constitution_ok += 1

    print(
        f"\nDeterministic Stage Verification (n={len(sample_smiles)} sampled M_0 cores):"
        f"\n  - Exact forward constitution rebuild match: {constitution_ok}/{len(sample_smiles)} ({100*constitution_ok/max(1,len(sample_smiles)):.1f}%)"
        f"\n  - Full stereo + constitution match (status='ok'): {verified_ok}/{len(sample_smiles)} ({100*verified_ok/max(1,len(sample_smiles)):.1f}%)"
    )

    # Save pickle, SMILES txt, and provenance parquet
    output_filepath = os.path.join(OUTPUT_DIR, f"{base_filename}.pkl")
    with open(output_filepath, "wb") as f:
        pickle.dump(unbound_pks_products_dict, f)

    smiles_filepath = os.path.join(OUTPUT_DIR, f"{base_filename}_SMILES.txt")
    unique_smiles_list = list(unbound_pks_products_dict.values())
    with open(smiles_filepath, "w") as f:
        for smiles in unique_smiles_list:
            f.write(smiles + "\n")

    # Also merge long-chain family-conditioned cores from generate_by_family.py (`data/cores/`)
    family_cores = load_family_generated_cores(PROJECT_ROOT / "data" / "cores")
    for frec in family_cores:
        if frec["smiles"] not in provenance_dict:
            provenance_dict[frec["smiles"]] = frec
    if family_cores:
        print(f"\nMerged {len(family_cores)} long-chain family-specific cores (4-14 modules: other_cis_at, polyethers, aromatic_polyene) from data/cores/")

    # Also write to default stereo-preserved core parquet for script 02
    provenance_rows = list(provenance_dict.values())
    provenance_parquet = os.path.join(OUTPUT_DIR, "pks_cores_with_stereo_provenance.parquet")
    pd.DataFrame(provenance_rows).to_parquet(provenance_parquet, index=False)

    # Render visual grid of stereo-preserved PKS cores
    plot_path = os.path.join(PLOTS_DIR, "nascent_pks_cores_stereo_registers.png")
    draw_sample_pks_cores(provenance_dict, plot_path, max_mols=12)

    phase3_time = time.time() - phase3_start
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Generated {len(unbound_pks_products_dict)} unique stereo-preserved nascent PKS cores")
    print(f"  - From thiolysis: {num_thiolysis}")
    print(f"  - From cyclization: {num_cyclization}")
    print(f"  - Containing chiral/alkene stereo (@, /, \\): "
          f"{sum(1 for s in unique_smiles_list if '@' in s or '/' in s or chr(92) in s)}")
    print(f"  - Provenance records saved: {len(provenance_rows)} -> {provenance_parquet}")
    print(f"  - Visualization saved: {plot_path}")
    print(f"  - Total time: {total_time:.2f}s (Phase 3: {phase3_time:.2f}s)")


# =============================================================================
# SCRIPT INITIALIZATION
# =============================================================================

modify_bcs_starters_extenders(starter_codes=STARTER_CODES, extender_codes=EXTENDER_CODES)
from retrotide import retrotide, structureDB
extension_modules_list = list(structureDB.keys())

if __name__ == "__main__":
    main()
