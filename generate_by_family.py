#!/usr/bin/env python3
"""
Generate modular Type I PKS core scaffolds for diverse scaffold-generation profiles.

This generator creates polyketide core structures for several major biosynthetic families.
Each *profile* represents a distinct PKS logic with configurable parameters and validation.

- The ``other_cis_at`` profile covers canonical cis-AT modular polyketides
  not belonging to the narrower polyether-ionophore or aromatic-polyene
  categories. This includes linear polyols, macrolactones/lactams, polyene
  macrolides, and most reduced cis-AT products. It does **not** represent a
  chemically uniform family; instead it aggregates non-specialized modular
  PKS products.

- Programmed iterative/stuttering PKSs (e.g., Borrelidin, reused NorA modules
  in spectinabilin/neoaureothin) are excluded from these profiles: every module
  is sampled once; module reuse/iteration is not modeled.

- The ``aromatic_polyene`` profile samples aromatic, conjugated-polyene
  precursor scaffolds. Products must contain ≥1 aromatic atom and a conjugated
  run of ≥3 aliphatic C=C bonds. This is for scaffold diversity, not actual BGC
  reconstruction.

- The ``polyethers`` profile uses a polyene/polyol precursor core, followed by a
  filter designed to select scaffolds compatible with (virtual) sequential 5/6-membered
  ring-forming epoxide-opening cascades. The filter checks for chain length,
  non-aromatic character, and a sufficient pattern of epoxidizable alkenes.
  Only the scaffold is produced; stereochemistry is retained. No epoxidation or
  ether formation is performed here; see ``augment.py`` for downstream modification.

Generation follows the original RetroTide / bcs workflow:

    bcs Cluster → ACP-bound product → release product

Outputs are written to ``data/cores/<profile>/``:
    - cores_SMILES.txt           (core SMILES, one per line)
    - cores_metadata.jsonl       (core metadata, JSON-per-line)
    - generation_stats.json      (summary statistics)

Optional: For polyether filtering, a rejection log can be written (``--rejection-log-dir``):
    - cores_gold_SMILES.txt          (passed gold-standard subset)
    - cores_gold_rejection.jsonl     (scaffolds or reasons for rejection)

Usage examples:
    python scripts/generate.py --class polyethers --target 5000
    python scripts/generate.py --all --target 250000
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
if str(ROOT / "RetroTideV2") not in sys.path:
    sys.path.insert(0, str(ROOT / "RetroTideV2"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bcs  # noqa: E402
DEFAULT_OUT = ROOT / "data" / "cores"
TRIVIAL_SMILES = {"", "S", "O=C=O"}

# ---------------------------------------------------------------------------
# Class profiles
# ---------------------------------------------------------------------------
CLASS_PROFILES: dict[str, dict[str, Any]] = {
    "other_cis_at": {
        "description": (
            "Other canonical cis-AT modular polyketides generated from mixed "
            "reduction-state modules, including linear polyols, "
            "macrolactones, macrolactams, and related reduced products."
        ),
        "release_modes": ("thiolysis", "cyclization"),
        "min_cyclization_ring_size": 12,
        "min_ext": 4,
        "max_ext": 10,
        "starter_codes": None,
        "extender_codes": [
            "Malonyl-CoA",
            "Methylmalonyl-CoA",
            "emal",
            "mxmal",
            "hmal",
            "allylmal",
        ],
        "prefer_non_aromatic": True,
    },
    "polyethers": {
        "description": (
            "Ionophore-like linear polyene/polyol precursors enriched for "
            "topologically feasible sequential 5/6-membered epoxide-opening "
            "cascades."
        ),
        "release_modes": ("thiolysis",),
        "min_ext": 11,
        "max_ext": 14,
        "starter_codes": [
            "Malonyl-CoA",
            "Methylmalonyl-CoA",
        ],
        "extender_codes": [
            "Malonyl-CoA",
            "Methylmalonyl-CoA",
            "emal",
        ],
        "prefer_non_aromatic": True,
        "min_alkenes": 3,
        "min_alcohols": 1,
        "min_cascade_steps": 2,
        "allowed_ring_sizes": (5, 6),
        "max_virtual_cascade_steps": 5,
    },
    "aromatic_polyene": {
        "description": (
            "Aromatic cis-AT PKS precursors with unsaturation-biased "
            "extension modules and a required conjugated polyene segment."
        ),
        "release_modes": ("thiolysis", "cyclization"),
        "min_ext": 3,
        "max_ext": 12,
        "starter_codes": None,
        "extender_codes": [
            "Malonyl-CoA",
            "Methylmalonyl-CoA",
            "emal",
            "allylmal",
        ],
        "prefer_non_aromatic": False,
        "require_aromatic": True,
        "min_aromatic_atoms": 1,
        "min_conjugated_alkene_run": 3,
        "min_cyclization_ring_size": 12,
    },
}

ALL_CLASSES = tuple(CLASS_PROFILES)
# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClosureAction:
    """One prospective intramolecular epoxide-opening action."""

    nucleophile_kind: str
    nucleophile_atom: Optional[int]
    nucleophile_anchor_carbon: Optional[int]
    alkene_bond: int
    attacked_carbon: int
    epoxide_other_carbon: int
    prospective_ring_size: int
    cascade_step: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolyetherTopologyResult:
    """Topology-level precursor assessment."""

    n_alcohols: int
    n_alkenes: int
    n_initial_closure_actions: int
    max_cascade_steps: int
    passed: bool
    best_trajectory: list[dict[str, Any]] = field(default_factory=list)
    alcohol_atoms: list[int] = field(default_factory=list)
    alkene_bonds: list[int] = field(default_factory=list)
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoreRecord:
    smiles: str
    modular_class: str
    release_mode: str
    n_extensions: int
    starter: str
    module_summary: str
    topology_summary: str = ""
    stereo_retained: bool = False
    aromatic_atom_count: int = 0
    max_conjugated_alkene_run: int = 0
    # Multi-class schema (Type I defaults). modular_class remains the family alias.
    biosynthetic_class: str = "type_i_pks"
    family: str = ""
    phase: str = "assembly"
    is_core_policy: str = "type_i_te_release"
    core_source: str = "generate"


# ---------------------------------------------------------------------------
# bcs / RetroTide wiring
# ---------------------------------------------------------------------------

def _validate_codes(
    registry: dict,
    codes: Optional[Sequence[str]],
    label: str,
) -> list[str]:
    """Resolve configured names against a bcs registry without fallback."""
    available = list(registry.keys())
    if codes is None:
        return available

    missing = [code for code in codes if code not in registry]
    if missing:
        raise RuntimeError(
            f"Unknown {label} code(s): {', '.join(missing)}. "
            f"Available codes: {', '.join(available)}"
        )
    return list(codes)


def modify_bcs(
    starter_codes: Optional[Sequence[str]],
    extender_codes: Optional[Sequence[str]],
) -> tuple[list[str], list[str]]:
    """Restrict global bcs registries before importing RetroTide structureDB."""
    selected_starters = _validate_codes(
        bcs.starters, starter_codes, "starter"
    )
    selected_extenders = _validate_codes(
        bcs.extenders, extender_codes, "extender"
    )

    if starter_codes is not None:
        keep = set(selected_starters)
        for key in list(bcs.starters):
            if key not in keep:
                bcs.starters.pop(key, None)

    if extender_codes is not None:
        keep = set(selected_extenders)
        for key in list(bcs.extenders):
            if key not in keep:
                bcs.extenders.pop(key, None)

    return selected_starters, selected_extenders


def build_bound(
    starter: str,
    extension_mods: Sequence,
    structure_db,
) -> Optional[Chem.Mol]:
    """Construct an ACP-bound modular PKS product."""
    try:
        loading_at = bcs.AT(active=True, substrate=starter)
        loading_module = bcs.Module(
            domains=OrderedDict({bcs.AT: loading_at}),
            loading=True,
        )
        cluster = bcs.Cluster(
            modules=[loading_module] + list(extension_mods)
        )
        bound = cluster.computeProduct(structure_db)
        Chem.SanitizeMol(bound)
        return bound
    except Exception:
        return None


def release_product(bound: Chem.Mol, mechanism: str) -> list[Chem.Mol]:
    """Release an ACP-bound product by hydrolysis or macrolactonization."""
    Chem.SanitizeMol(bound)

    if mechanism == "thiolysis":
        rxn = AllChem.ReactionFromSmarts(
            "[C:1](=[O:2])[S:3]>>[C:1](=[O:2])[O].[S:3]"
        )
    elif mechanism == "cyclization":
        rxn = AllChem.ReactionFromSmarts(
            "([C:1](=[O:2])[S:3].[O,N:4][C:5][C:6])"
            ">>[C:1](=[O:2])[*:4][C:5][C:6].[S:3]"   
        )
    else:
        raise ValueError(f"Unsupported release mechanism: {mechanism}")

    released: list[Chem.Mol] = []
    for product_tuple in rxn.RunReactants((bound,)) or []:
        valid_products: list[Chem.Mol] = []
        for product in product_tuple:
            try:
                Chem.SanitizeMol(product)
            except Exception:
                continue
            valid_products.append(product)
        if valid_products:
            released.append(
                max(
                    valid_products,
                    key=lambda product: product.GetNumHeavyAtoms(),
                )
            )
    return released


# ---------------------------------------------------------------------------
# Molecule utilities
# ---------------------------------------------------------------------------
def canonicalize(
    mol: Chem.Mol,
    *,
    strip_stereo: bool,
) -> Optional[str]:
    """Return a sanitized canonical SMILES or None."""
    candidate = Chem.Mol(mol)

    if strip_stereo:
        Chem.RemoveStereochemistry(candidate)
    else:
        Chem.AssignStereochemistry(
            candidate,
            cleanIt=True,
            force=True,
        )

    try:
        Chem.SanitizeMol(candidate)
        smiles = Chem.MolToSmiles(
            candidate,
            canonical=True,
            isomericSmiles=not strip_stereo,
        )
    except Exception:
        return None

    if smiles in TRIVIAL_SMILES:
        return None
    return smiles


def count_aromatic_atoms(mol: Chem.Mol) -> int:
    """Return the number of aromatic atoms in a molecule."""
    return sum(atom.GetIsAromatic() for atom in mol.GetAtoms())


def max_conjugated_alkene_run(mol: Chem.Mol) -> int:
    """Return the largest connected run of conjugated aliphatic C=C bonds."""
    double_bonds: list[tuple[int, set[int]]] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if bond.GetIsAromatic():
            continue
        if (
            bond.GetBeginAtom().GetAtomicNum() != 6
            or bond.GetEndAtom().GetAtomicNum() != 6
        ):
            continue
        double_bonds.append(
            (
                bond.GetIdx(),
                {
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                },
            )
        )

    adjacency = {
        bond_idx: set()
        for bond_idx, _ in double_bonds
    }
    for position, (bond_a, atoms_a) in enumerate(double_bonds):
        for bond_b, atoms_b in double_bonds[position + 1 :]:
            conjugated = any(
                (
                    connecting := mol.GetBondBetweenAtoms(atom_a, atom_b)
                ) is not None
                and connecting.GetBondType() == Chem.BondType.SINGLE
                and not connecting.GetIsAromatic()
                for atom_a in atoms_a
                for atom_b in atoms_b
            )
            if conjugated:
                adjacency[bond_a].add(bond_b)
                adjacency[bond_b].add(bond_a)

    longest = 0
    visited: set[int] = set()
    for bond_idx in adjacency:
        if bond_idx in visited:
            continue
        component = [bond_idx]
        visited.add(bond_idx)
        size = 0
        while component:
            current = component.pop()
            size += 1
            for neighbor in adjacency[current] - visited:
                visited.add(neighbor)
                component.append(neighbor)
        longest = max(longest, size)
    return longest


def module_summary(modules: Sequence) -> str:
    values: list[str] = []
    for module in modules:
        try:
            values.append(str(module))
        except Exception:
            values.append("?")
    return " | ".join(values)[:1000]


def _module_tags(module) -> set[str]:
    """Extract reduction-domain tags from a RetroTide module string."""
    text = str(module)
    tags: set[str] = set()
    for domain in ("KR", "DH", "ER"):
        if domain in text:
            tags.add(domain)
    return tags


def split_module_pools(extension_modules: Sequence) -> dict[str, list]:
    """Partition extension modules by reduction-state fingerprint."""
    pools: dict[str, list] = {
        "olefinic": [],  # KR + DH, no ER
        "hydroxy": [],   # KR, no DH
        "saturated": [], # KR + DH + ER
        "keto_or_other": [],
    }

    for module in extension_modules:
        tags = _module_tags(module)
        if {"KR", "DH"}.issubset(tags) and "ER" not in tags:
            pools["olefinic"].append(module)
        elif "KR" in tags and "DH" not in tags:
            pools["hydroxy"].append(module)
        elif {"KR", "DH", "ER"}.issubset(tags):
            pools["saturated"].append(module)
        else:
            pools["keto_or_other"].append(module)

    return pools


# ---------------------------------------------------------------------------
# Polyether precursor topology
# ---------------------------------------------------------------------------
ALIPHATIC_ALCOHOL_QUERY = Chem.MolFromSmarts(
    "[O;H1;+0;!$([O]-[C,S,P]=O)]-[C;!a;!$(C=O);!$(C#N)]"
)

CARBOXYLIC_ACID_QUERY = Chem.MolFromSmarts(
    "[O;H1;+0]-[C](=O)"
)


def find_aliphatic_alcohol_oxygens(mol: Chem.Mol) -> list[int]:
    """Find neutral aliphatic alcohol oxygens, excluding carboxylic acids."""
    if ALIPHATIC_ALCOHOL_QUERY is None:
        raise RuntimeError("Failed to compile alcohol SMARTS")

    acid_oxygens: set[int] = set()
    if CARBOXYLIC_ACID_QUERY is not None:
        acid_oxygens = {
            match[0]
            for match in mol.GetSubstructMatches(CARBOXYLIC_ACID_QUERY)
        }

    oxygens = {
        match[0]
        for match in mol.GetSubstructMatches(ALIPHATIC_ALCOHOL_QUERY)
        if match[0] not in acid_oxygens
    }
    return sorted(oxygens)


def find_epoxidizable_alkene_bonds(mol: Chem.Mol) -> list[int]:
    """
    Find non-aromatic, non-ring C=C bonds suitable for topology screening.

    This intentionally remains broader than an enzyme-specific MonCI/Lsd18
    substrate model. Enzyme-family filters belong in the later augmentation
    stage.
    """
    output: list[int] = []

    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if bond.GetIsAromatic() or bond.IsInRing():
            continue

        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetAtomicNum() != 6 or end.GetAtomicNum() != 6:
            continue

        # Exclude cumulenes/allenes and directly carbonyl-conjugated centers
        # from the broad precursor pool.
        if _has_other_multiple_bond(begin, bond.GetIdx()):
            continue
        if _has_other_multiple_bond(end, bond.GetIdx()):
            continue

        output.append(bond.GetIdx())

    return output


def _has_other_multiple_bond(atom: Chem.Atom, excluded_bond_idx: int) -> bool:
    for bond in atom.GetBonds():
        if bond.GetIdx() == excluded_bond_idx:
            continue
        if bond.GetBondType() in (
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
        ):
            return True
    return False


def _path_is_acyclic(
    mol: Chem.Mol,
    path: Sequence[int],
) -> bool:
    """Reject closures whose precursor path already traverses a ring."""
    return not any(
        mol.GetAtomWithIdx(atom_idx).IsInRing()
        for atom_idx in path
    )


def enumerate_initial_closure_actions(
    mol: Chem.Mol,
    alcohol_atoms: Sequence[int],
    alkene_bonds: Sequence[int],
    allowed_ring_sizes: set[int],
) -> list[ClosureAction]:
    """Enumerate OH -> virtual epoxide attack actions."""
    actions: list[ClosureAction] = []

    for oxygen_idx in alcohol_atoms:
        for bond_idx in alkene_bonds:
            bond = mol.GetBondWithIdx(bond_idx)
            carbon_a = bond.GetBeginAtomIdx()
            carbon_b = bond.GetEndAtomIdx()

            for attacked, other in (
                (carbon_a, carbon_b),
                (carbon_b, carbon_a),
            ):
                path = Chem.rdmolops.GetShortestPath(
                    mol,
                    oxygen_idx,
                    attacked,
                )
                ring_size = len(path)

                if ring_size not in allowed_ring_sizes:
                    continue
                if not _path_is_acyclic(mol, path):
                    continue

                actions.append(
                    ClosureAction(
                        nucleophile_kind="preexisting_alcohol",
                        nucleophile_atom=oxygen_idx,
                        nucleophile_anchor_carbon=None,
                        alkene_bond=bond_idx,
                        attacked_carbon=attacked,
                        epoxide_other_carbon=other,
                        prospective_ring_size=ring_size,
                        cascade_step=1,
                    )
                )

    return actions


def enumerate_propagating_actions(
    mol: Chem.Mol,
    nucleophile_anchor_carbon: int,
    alkene_bonds: Sequence[int],
    used_alkene_bonds: set[int],
    allowed_ring_sizes: set[int],
    cascade_step: int,
) -> list[ClosureAction]:
    """
    Enumerate subsequent virtual cascade actions.

    After epoxide opening, the epoxide oxygen remains attached to the
    non-attacked epoxide carbon and becomes the next alcohol. The virtual
    oxygen is not present in the precursor graph, so:

        prospective ring size =
            shortest-path atom count(anchor carbon -> attacked carbon) + 1

    The +1 accounts for the new nucleophilic oxygen.
    """
    actions: list[ClosureAction] = []

    for bond_idx in alkene_bonds:
        if bond_idx in used_alkene_bonds:
            continue

        bond = mol.GetBondWithIdx(bond_idx)
        carbon_a = bond.GetBeginAtomIdx()
        carbon_b = bond.GetEndAtomIdx()

        for attacked, other in (
            (carbon_a, carbon_b),
            (carbon_b, carbon_a),
        ):
            path = Chem.rdmolops.GetShortestPath(
                mol,
                nucleophile_anchor_carbon,
                attacked,
            )
            ring_size = len(path) + 1

            if ring_size not in allowed_ring_sizes:
                continue
            if not _path_is_acyclic(mol, path):
                continue

            actions.append(
                ClosureAction(
                    nucleophile_kind="epoxide_derived_alcohol",
                    nucleophile_atom=None,
                    nucleophile_anchor_carbon=nucleophile_anchor_carbon,
                    alkene_bond=bond_idx,
                    attacked_carbon=attacked,
                    epoxide_other_carbon=other,
                    prospective_ring_size=ring_size,
                    cascade_step=cascade_step,
                )
            )

    return actions


def enumerate_virtual_cascades(
    mol: Chem.Mol,
    initial_actions: Sequence[ClosureAction],
    alkene_bonds: Sequence[int],
    allowed_ring_sizes: set[int],
    max_steps: int,
) -> list[list[ClosureAction]]:
    """Depth-first enumeration of topology-level epoxide-opening cascades."""
    trajectories: list[list[ClosureAction]] = []

    def dfs(
        trajectory: list[ClosureAction],
        used_bonds: set[int],
        next_anchor: int,
    ) -> None:
        trajectories.append(list(trajectory))

        if len(trajectory) >= max_steps:
            return

        next_actions = enumerate_propagating_actions(
            mol=mol,
            nucleophile_anchor_carbon=next_anchor,
            alkene_bonds=alkene_bonds,
            used_alkene_bonds=used_bonds,
            allowed_ring_sizes=allowed_ring_sizes,
            cascade_step=len(trajectory) + 1,
        )

        for action in next_actions:
            trajectory.append(action)
            used_bonds.add(action.alkene_bond)

            dfs(
                trajectory=trajectory,
                used_bonds=used_bonds,
                next_anchor=action.epoxide_other_carbon,
            )

            used_bonds.remove(action.alkene_bond)
            trajectory.pop()

    for first in initial_actions:
        dfs(
            trajectory=[first],
            used_bonds={first.alkene_bond},
            next_anchor=first.epoxide_other_carbon,
        )

    trajectories.sort(
        key=lambda trajectory: (
            len(trajectory),
            tuple(
                action.prospective_ring_size
                for action in trajectory
            ),
        ),
        reverse=True,
    )
    return trajectories


def polyether_precursor_topology(
    mol: Chem.Mol,
    *,
    min_alkenes: int,
    min_alcohols: int,
    min_cascade_steps: int,
    allowed_ring_sizes: Sequence[int],
    max_virtual_cascade_steps: int,
) -> PolyetherTopologyResult:
    """Assess whether a PKS precursor supports a virtual ether cascade."""
    alcohol_atoms = find_aliphatic_alcohol_oxygens(mol)
    alkene_bonds = find_epoxidizable_alkene_bonds(mol)

    if len(alcohol_atoms) < min_alcohols:
        return PolyetherTopologyResult(
            n_alcohols=len(alcohol_atoms),
            n_alkenes=len(alkene_bonds),
            n_initial_closure_actions=0,
            max_cascade_steps=0,
            passed=False,
            alcohol_atoms=alcohol_atoms,
            alkene_bonds=alkene_bonds,
            rejection_reason="insufficient_aliphatic_alcohols",
        )

    if len(alkene_bonds) < min_alkenes:
        return PolyetherTopologyResult(
            n_alcohols=len(alcohol_atoms),
            n_alkenes=len(alkene_bonds),
            n_initial_closure_actions=0,
            max_cascade_steps=0,
            passed=False,
            alcohol_atoms=alcohol_atoms,
            alkene_bonds=alkene_bonds,
            rejection_reason="insufficient_epoxidizable_alkenes",
        )

    ring_sizes = set(int(value) for value in allowed_ring_sizes)
    initial_actions = enumerate_initial_closure_actions(
        mol=mol,
        alcohol_atoms=alcohol_atoms,
        alkene_bonds=alkene_bonds,
        allowed_ring_sizes=ring_sizes,
    )

    if not initial_actions:
        return PolyetherTopologyResult(
            n_alcohols=len(alcohol_atoms),
            n_alkenes=len(alkene_bonds),
            n_initial_closure_actions=0,
            max_cascade_steps=0,
            passed=False,
            alcohol_atoms=alcohol_atoms,
            alkene_bonds=alkene_bonds,
            rejection_reason="no_5_or_6_membered_initial_closure",
        )

    trajectories = enumerate_virtual_cascades(
        mol=mol,
        initial_actions=initial_actions,
        alkene_bonds=alkene_bonds,
        allowed_ring_sizes=ring_sizes,
        max_steps=max_virtual_cascade_steps,
    )

    best = trajectories[0] if trajectories else []
    max_steps = len(best)
    passed = max_steps >= min_cascade_steps

    return PolyetherTopologyResult(
        n_alcohols=len(alcohol_atoms),
        n_alkenes=len(alkene_bonds),
        n_initial_closure_actions=len(initial_actions),
        max_cascade_steps=max_steps,
        passed=passed,
        best_trajectory=[
            action.to_dict()
            for action in best
        ],
        alcohol_atoms=alcohol_atoms,
        alkene_bonds=alkene_bonds,
        rejection_reason=(
            ""
            if passed
            else "virtual_cascade_shorter_than_required"
        ),
    )

# ---------------------------------------------------------------------------
# Class-aware module sampling
# ---------------------------------------------------------------------------
def _random_positions(
    rng: random.Random,
    n_ext: int,
    count: int,
    *,
    forbidden: Optional[set[int]] = None,
) -> set[int]:
    forbidden = forbidden or set()
    candidates = [
        idx
        for idx in range(n_ext)
        if idx not in forbidden
    ]
    count = min(count, len(candidates))
    return set(rng.sample(candidates, count))


def sample_polyether_modules(
    n_ext: int,
    extension_modules: Sequence,
    pools: dict[str, list],
    rng: random.Random,
) -> list:
    """
    Generate a topology-biased but non-periodic reduction-state sequence.

    The previous implementation forced strict olefinic/hydroxy alternation.
    This rewrite instead:
      - selects 3-6 olefinic positions;
      - selects 1-4 hydroxy positions;
      - fills remaining positions from saturated/keto/other modules;
      - allows neighboring olefins and non-repeating local patterns;
      - relies on the molecular virtual-cascade filter for final acceptance.
    """
    olefinic_pool = pools["olefinic"] or list(extension_modules)
    hydroxy_pool = pools["hydroxy"] or list(extension_modules)

    background_pool = (
        pools["saturated"]
        + pools["keto_or_other"]
        + pools["hydroxy"]
        + pools["olefinic"]
    ) or list(extension_modules)

    max_olefinic = min(6, max(3, n_ext // 2))
    n_olefinic = rng.randint(3, max_olefinic)

    max_hydroxy = min(4, max(1, n_ext // 3))
    n_hydroxy = rng.randint(1, max_hydroxy)

    olefinic_positions = _random_positions(
        rng,
        n_ext,
        n_olefinic,
    )
    hydroxy_positions = _random_positions(
        rng,
        n_ext,
        n_hydroxy,
        forbidden=olefinic_positions,
    )

    modules: list = []
    for position in range(n_ext):
        if position in olefinic_positions:
            modules.append(rng.choice(olefinic_pool))
        elif position in hydroxy_positions:
            modules.append(rng.choice(hydroxy_pool))
        else:
            modules.append(rng.choice(background_pool))

    return modules


def sample_extension_modules(
    modular_class: str,
    n_ext: int,
    extension_modules: Sequence,
    rng: random.Random,
) -> list:
    """Class-aware module sampling."""
    pools = split_module_pools(extension_modules)

    if modular_class == "polyethers":
        return sample_polyether_modules(
            n_ext=n_ext,
            extension_modules=extension_modules,
            pools=pools,
            rng=rng,
        )

    if modular_class == "aromatic_polyene":
        # Unsaturation bias is secondary. Product-level aromaticity is enforced
        # later, so non-aromatic polyenes cannot enter this corpus.
        olefinic_pool = pools["olefinic"] or list(extension_modules)
        rest_pool = (
            pools["hydroxy"]
            + pools["saturated"]
            + pools["keto_or_other"]
        ) or list(extension_modules)

        n_olefinic = max(2, n_ext // 2)
        modules = [
            rng.choice(olefinic_pool)
            for _ in range(n_olefinic)
        ]
        modules.extend(
            rng.choice(rest_pool)
            for _ in range(n_ext - n_olefinic)
        )
        rng.shuffle(modules)
        return modules

    return [
        rng.choice(extension_modules)
        for _ in range(n_ext)
    ]


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def _empty_stats(
    profile: dict[str, Any],
    pools: dict[str, list],
) -> dict[str, Any]:
    return {
        "attempts": 0,
        "bound_fail": 0,
        "release_fail": 0,
        "canonicalization_fail": 0,
        "filter_aromatic": 0,
        "filter_missing_required_aromaticity": 0,
        "filter_missing_required_polyene": 0,
        "filter_cyclization_ring_size": 0,
        "accepted_aromatic_atom_counts": {},
        "accepted_conjugated_alkene_runs": {},
        "filter_polyether_topology": 0,
        "duplicate": 0,
        "accepted": 0,
        "by_release": {
            mode: 0
            for mode in profile["release_modes"]
        },
        "by_n_ext": {},
        "polyether_rejection_reasons": {},
        "polyether_max_cascade_steps": {},
        "module_pools": {
            name: len(values)
            for name, values in pools.items()
        },
    }


def generate_class_cores(
    modular_class: str,
    target: int,
    structure_db,
    *,
    seed: int,
    strip_stereo: bool,
    max_attempts_factor: int,
    rejection_log_limit: int = 0,
) -> tuple[list[CoreRecord], dict[str, Any], list[dict[str, Any]]]:
    """Generate one modular-class corpus."""
    profile = CLASS_PROFILES[modular_class]
    available_starters = list(bcs.starters)
    extension_modules = list(structure_db.keys())

    if not available_starters or not extension_modules:
        raise RuntimeError(
            f"No starters/modules for class={modular_class}: "
            f"starters={len(available_starters)} "
            f"modules={len(extension_modules)}"
        )

    pools = split_module_pools(extension_modules)
    print(
        "  module pools: "
        + " ".join(
            f"{name}={len(values)}"
            for name, values in pools.items()
        )
    )

    rng = random.Random(seed)
    seen: set[str] = set()
    records: list[CoreRecord] = []
    rejected_log: list[dict[str, Any]] = []
    stats = _empty_stats(profile, pools)

    if modular_class == "polyethers":
        max_attempts_factor = max(max_attempts_factor, 100)
    elif modular_class == "aromatic_polyene":
        # Aromatic products are a minority when all RetroTide starters are
        # available, so allow a larger search budget without relaxing the
        # aromatic filters.
        max_attempts_factor = max(max_attempts_factor, 120)

    max_attempts = max(
        target * max_attempts_factor,
        target + 1000,
    )

    min_ext = int(profile["min_ext"])
    max_ext = int(profile["max_ext"])

    while (
        len(records) < target
        and stats["attempts"] < max_attempts
    ):
        stats["attempts"] += 1

        n_ext = rng.randint(min_ext, max_ext)
        starter = rng.choice(available_starters)
        modules = sample_extension_modules(
            modular_class=modular_class,
            n_ext=n_ext,
            extension_modules=extension_modules,
            rng=rng,
        )

        bound = build_bound(
            starter=starter,
            extension_mods=modules,
            structure_db=structure_db,
        )
        if bound is None:
            stats["bound_fail"] += 1
            continue

        release_mode = rng.choice(
            list(profile["release_modes"])
        )
        try:
            released_products = release_product(
                bound,
                release_mode,
            )
        except Exception:
            stats["release_fail"] += 1
            continue

        if not released_products:
            stats["release_fail"] += 1
            continue

        if release_mode == "cyclization":
            min_ring_size = int(
                profile.get("min_cyclization_ring_size", 0)
            )
            if min_ring_size:
                released_products = [
                    product
                    for product in released_products
                    if any(
                        len(ring) >= min_ring_size
                        for ring in product.GetRingInfo().AtomRings()
                    )
                ]
                if not released_products:
                    stats["filter_cyclization_ring_size"] += 1
                    continue

        # A bound chain can contain several O/N nucleophiles. Treat these as
        # alternative release outcomes rather than accepting every closure
        # from one sampled assembly line and biasing the corpus toward cycles.
        released_products = [rng.choice(released_products)]

        for released in released_products:
            smiles = canonicalize(
                released,
                strip_stereo=strip_stereo,
            )
            if smiles is None:
                stats["canonicalization_fail"] += 1
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                stats["canonicalization_fail"] += 1
                continue

            aromatic_atom_count = count_aromatic_atoms(mol)
            conjugated_alkene_run = max_conjugated_alkene_run(mol)

            if (
                profile.get("prefer_non_aromatic", False)
                and aromatic_atom_count > 0
            ):
                stats["filter_aromatic"] += 1
                continue

            if profile.get("require_aromatic", False):
                min_aromatic_atoms = int(
                    profile.get("min_aromatic_atoms", 1)
                )
                if aromatic_atom_count < min_aromatic_atoms:
                    stats["filter_missing_required_aromaticity"] += 1
                    if len(rejected_log) < rejection_log_limit:
                        rejected_log.append(
                            {
                                "outcome": "rejected",
                                "reason": "missing_required_aromaticity",
                                "smiles": smiles,
                                "aromatic_atom_count": aromatic_atom_count,
                                "required_aromatic_atoms": min_aromatic_atoms,
                                "n_extensions": n_ext,
                                "starter": starter,
                                "module_summary": module_summary(modules),
                                "stereo_retained": not strip_stereo,
                            }
                        )
                    continue

            min_polyene_run = int(
                profile.get("min_conjugated_alkene_run", 0)
            )
            if conjugated_alkene_run < min_polyene_run:
                stats["filter_missing_required_polyene"] += 1
                if len(rejected_log) < rejection_log_limit:
                    rejected_log.append(
                        {
                            "outcome": "rejected",
                            "reason": "missing_required_polyene",
                            "smiles": smiles,
                            "max_conjugated_alkene_run": (
                                conjugated_alkene_run
                            ),
                            "required_conjugated_alkene_run": (
                                min_polyene_run
                            ),
                            "n_extensions": n_ext,
                            "starter": starter,
                            "module_summary": module_summary(modules),
                            "stereo_retained": not strip_stereo,
                        }
                    )
                continue

            topology: Optional[PolyetherTopologyResult] = None
            if modular_class == "polyethers":
                topology = polyether_precursor_topology(
                    mol,
                    min_alkenes=int(profile["min_alkenes"]),
                    min_alcohols=int(profile["min_alcohols"]),
                    min_cascade_steps=int(
                        profile["min_cascade_steps"]
                    ),
                    allowed_ring_sizes=profile[
                        "allowed_ring_sizes"
                    ],
                    max_virtual_cascade_steps=int(
                        profile["max_virtual_cascade_steps"]
                    ),
                )

                cascade_key = str(topology.max_cascade_steps)
                stats["polyether_max_cascade_steps"][
                    cascade_key
                ] = (
                    stats["polyether_max_cascade_steps"].get(
                        cascade_key, 0
                    )
                    + 1
                )

                if not topology.passed:
                    stats["filter_polyether_topology"] += 1
                    reason = (
                        topology.rejection_reason
                        or "unknown_polyether_topology_failure"
                    )
                    stats["polyether_rejection_reasons"][
                        reason
                    ] = (
                        stats["polyether_rejection_reasons"].get(
                            reason, 0
                        )
                        + 1
                    )

                    if (
                        len(rejected_log)
                        < rejection_log_limit
                    ):
                        rejected_log.append(
                            {
                                "outcome": "rejected",
                                "reason": reason,
                                "smiles": smiles,
                                "topology_summary": (
                                    topology.to_dict()
                                ),
                                "n_extensions": n_ext,
                                "starter": starter,
                                "module_summary": (
                                    module_summary(modules)
                                ),
                                "stereo_retained": (
                                    not strip_stereo
                                ),
                            }
                        )
                    continue

            if smiles in seen:
                stats["duplicate"] += 1
                continue

            seen.add(smiles)
            topology_summary = (
                json.dumps(
                    topology.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if topology is not None
                else ""
            )

            record = CoreRecord(
                smiles=smiles,
                modular_class=modular_class,
                release_mode=release_mode,
                n_extensions=n_ext,
                starter=starter,
                module_summary=module_summary(modules),
                topology_summary=topology_summary,
                stereo_retained=not strip_stereo,
                aromatic_atom_count=aromatic_atom_count,
                max_conjugated_alkene_run=conjugated_alkene_run,
                biosynthetic_class="type_i_pks",
                family=modular_class,
                phase="assembly",
                is_core_policy="type_i_te_release",
                core_source="generate",
            )
            records.append(record)

            stats["accepted"] += 1
            aromatic_key = str(aromatic_atom_count)
            stats["accepted_aromatic_atom_counts"][aromatic_key] = (
                stats["accepted_aromatic_atom_counts"].get(
                    aromatic_key, 0
                )
                + 1
            )
            conjugation_key = str(conjugated_alkene_run)
            stats["accepted_conjugated_alkene_runs"][
                conjugation_key
            ] = (
                stats["accepted_conjugated_alkene_runs"].get(
                    conjugation_key, 0
                )
                + 1
            )
            stats["by_release"][release_mode] = (
                stats["by_release"].get(
                    release_mode, 0
                )
                + 1
            )
            extension_key = str(n_ext)
            stats["by_n_ext"][extension_key] = (
                stats["by_n_ext"].get(
                    extension_key, 0
                )
                + 1
            )

            if len(records) % 1000 == 0 or len(records) >= target:
                print(
                    f"  progress {modular_class}: "
                    f"{len(records)}/{target} cores "
                    f"(attempts={stats['attempts']} "
                    f"bound_fail={stats['bound_fail']} "
                    f"release_fail={stats['release_fail']} "
                    f"filter_aromatic={stats['filter_aromatic']} "
                    f"filter_polyether={stats['filter_polyether_topology']} "
                    f"filter_polyene={stats['filter_missing_required_polyene']} "
                    f"filter_missing_arom={stats['filter_missing_required_aromaticity']} "
                    f"filter_ring={stats['filter_cyclization_ring_size']} "
                    f"dup={stats['duplicate']})",
                    flush=True,
                )

            if len(records) >= target:
                break

    if modular_class == "aromatic_polyene":
        non_aromatic_records = [
            record.smiles
            for record in records
            if record.aromatic_atom_count < int(
                profile.get("min_aromatic_atoms", 1)
            )
        ]
        if non_aromatic_records:
            raise RuntimeError(
                "Internal aromatic_polyene admission failure: "
                f"{len(non_aromatic_records)} accepted records lack required "
                "aromatic atoms."
            )

    stats["n_unique"] = len(records)
    stats["target"] = target
    stats["hit_target"] = len(records) >= target
    stats["acceptance_rate"] = (
        len(records) / stats["attempts"]
        if stats["attempts"]
        else 0.0
    )

    return records, stats, rejected_log


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_class_outputs(
    modular_class: str,
    records: Sequence[CoreRecord],
    stats: dict[str, Any],
    out_root: Path,
    profile: dict[str, Any],
) -> Path:
    out_dir = out_root / modular_class
    out_dir.mkdir(parents=True, exist_ok=True)

    smiles_path = out_dir / "cores_SMILES.txt"
    metadata_path = out_dir / "cores_metadata.jsonl"
    stats_path = out_dir / "generation_stats.json"

    with smiles_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.smiles + "\n")

    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    asdict(record),
                    ensure_ascii=False,
                )
                + "\n"
            )

    payload = {
        "modular_class": modular_class,
        "family": modular_class,
        "biosynthetic_class": "type_i_pks",
        "is_core_policy": "type_i_te_release",
        "core_source": "generate",
        "phase": "assembly",
        "description": profile.get("description"),
        "profile": {
            key: value
            for key, value in profile.items()
            if key != "description"
        },
        "n_cores": len(records),
        "stats": stats,
        "outputs": {
            "smiles": str(smiles_path),
            "metadata": str(metadata_path),
        },
    }
    stats_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return out_dir


def write_polyether_rejection_log(
    out_dir: Path,
    records: Sequence[CoreRecord],
    rejected: Sequence[dict[str, Any]],
) -> None:
    """Write accepted/rejected polyether topology filter records."""
    out_dir.mkdir(parents=True, exist_ok=True)

    smiles_path = out_dir / "cores_gold_SMILES.txt"
    rejection_path = out_dir / "cores_gold_rejection.jsonl"

    with smiles_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.smiles + "\n")

    with rejection_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = asdict(record)
            payload["outcome"] = "accepted"
            payload["reason"] = (
                "virtual_epoxide_opening_cascade_passed"
            )
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n"
            )

        for payload in rejected:
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--class",
        dest="modular_class",
        choices=ALL_CLASSES,
        help="Generate one modular class.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Generate all classes in isolated subprocesses so global "
            "bcs registries cannot leak across classes."
        ),
    )

    parser.add_argument(
        "--target",
        type=int,
        default=5000,
        help="Number of unique cores per class.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Use target=200 when --target remains at its default."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT,
    )
    parser.add_argument(
        "--keep-stereo",
        action="store_true",
        help=(
            "Retain stereochemistry for non-polyether classes. "
            "Polyether generation always retains stereochemistry."
        ),
    )
    parser.add_argument(
        "--max-attempts-factor",
        type=int,
        default=40,
        help=(
            "Maximum attempts = target x factor. Polyethers use at least 100."
        ),
    )
    parser.add_argument(
        "--rejection-log-dir",
        type=Path,
        default=None,
        help=(
            "Write accepted/rejected polyether topology filter records. "
            "Valid only with --class polyethers."
        ),
    )

    return parser.parse_args()


def run_all_classes(
    args: argparse.Namespace,
    target: int,
) -> None:
    """Generate every class in a fresh Python process."""
    start = time.time()

    for index, modular_class in enumerate(ALL_CLASSES):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--class",
            modular_class,
            "--target",
            str(target),
            "--seed",
            str(args.seed + 1009 * index),
            "--out-root",
            str(args.out_root),
            "--max-attempts-factor",
            str(args.max_attempts_factor),
        ]

        if args.keep_stereo:
            command.append("--keep-stereo")

        print("=" * 72, flush=True)
        print(
            f"Launching isolated class process: "
            f"{modular_class} target={target}",
            flush=True,
        )
        subprocess.run(command, check=True)

    print(
        f"All isolated class runs completed in "
        f"{time.time() - start:.1f}s"
    )


def main() -> None:
    args = parse_args()

    target = (
        200
        if args.pilot and args.target == 5000
        else args.target
    )
    if target <= 0:
        raise SystemExit("--target must be a positive integer")
    if args.max_attempts_factor <= 0:
        raise SystemExit(
            "--max-attempts-factor must be positive"
        )

    if args.all:
        if args.rejection_log_dir is not None:
            raise SystemExit(
                "--rejection-log-dir requires --class polyethers"
            )
        run_all_classes(args, target)
        return

    modular_class = args.modular_class
    if (
        args.rejection_log_dir is not None
        and modular_class != "polyethers"
    ):
        raise SystemExit(
            "--rejection-log-dir requires --class polyethers"
        )

    profile = CLASS_PROFILES[modular_class]

    selected_starters, selected_extenders = modify_bcs(
        starter_codes=profile["starter_codes"],
        extender_codes=profile["extender_codes"],
    )

    print(
        f"selected starters={len(selected_starters)}: "
        + ", ".join(selected_starters)
    )
    print(
        f"selected extenders={len(selected_extenders)}: "
        + ", ".join(selected_extenders)
    )

    # structureDB must be imported after trimming bcs extenders.
    from retrotide import structureDB  # type: ignore

    print(f"structureDB modules={len(structureDB)}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    # Polyether epoxidation/cyclization requires stereochemical retention.
    strip_stereo = (
        False
        if modular_class == "polyethers"
        else not args.keep_stereo
    )

    start = time.time()
    print("=" * 72)
    print(f"CLASS {modular_class} target={target}")
    print(profile["description"])
    print(f"strip_stereo={strip_stereo}")

    records, stats, rejected = generate_class_cores(
        modular_class=modular_class,
        target=target,
        structure_db=structureDB,
        seed=args.seed,
        strip_stereo=strip_stereo,
        max_attempts_factor=args.max_attempts_factor,
        rejection_log_limit=(
            target
            if args.rejection_log_dir is not None
            else 0
        ),
    )

    stats["selected_starters"] = selected_starters
    stats["selected_extenders"] = selected_extenders

    out_dir = write_class_outputs(
        modular_class=modular_class,
        records=records,
        stats=stats,
        out_root=args.out_root,
        profile=profile,
    )

    if args.rejection_log_dir is not None:
        write_polyether_rejection_log(
            out_dir=args.rejection_log_dir,
            records=records,
            rejected=rejected,
        )
        print(
            f"  wrote topology rejection log -> "
            f"{args.rejection_log_dir} "
            f"(accepted={len(records)} "
            f"rejected={len(rejected)})"
        )

    print(
        f"  wrote {len(records)} cores -> {out_dir} "
        f"(attempts={stats['attempts']} "
        f"bound_fail={stats['bound_fail']} "
        f"release_fail={stats['release_fail']} "
        f"missing_required_aromaticity="
        f"{stats['filter_missing_required_aromaticity']} "
        f"acceptance_rate={stats['acceptance_rate']:.6f})"
    )
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()