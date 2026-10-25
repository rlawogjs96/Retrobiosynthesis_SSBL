#!/usr/bin/env python3
"""Assign Keiser et al. 2025 KR subtypes from the observed β-OH / β-keto.

This is the KR-subtype predictor.

Keiser et al., J. Am. Chem. Soc. 2025, 147, 42237, Figure 1 and SI Table S1
define A1/A2/B1/B2 on a ketide that still has the β-OH (and, for
methylmalonyl, the α-methyl). The four authentic Lip1-TE acids and the
typed native-substrate SMILES in Table S1 are:

    A1 = 2R, 3S     A2 = 2S, 3S     B1 = 2R, 3R     B2 = 2S, 3R
    C-type          no NADPH reduction (ketone remains)

Those papers do not assign a KR subtype after DH or ER has removed the
β-OH. This script therefore writes U on enoyl and saturated modules.
DH geometry and ER α-configuration are still recorded separately; they
are not KR 1/2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

STAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE_DIR.parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

import predict_extenders as pe  # noqa: E402

RETROTIDE_BCS_DATA = (
    STAGE_DIR / "data"
    if (STAGE_DIR / "data" / "starters.smi").exists()
    else (
        PROJECT_ROOT / "data"
        if (PROJECT_ROOT / "data" / "starters.smi").exists()
        else (
            PROJECT_ROOT / "RetroTideV2" / "bcs" / "data"
            if (PROJECT_ROOT / "RetroTideV2" / "bcs" / "data").exists()
            else PROJECT_ROOT / "RetroTide" / "bcs" / "data"
        )
    )
)


def _alpha_substituents(mol, alpha_idx, backbone_set):
    """Return non-backbone heavy atom neighbor indices attached to `alpha_idx`."""
    atom = mol.GetAtomWithIdx(int(alpha_idx))
    return [
        nb.GetIdx()
        for nb in atom.GetNeighbors()
        if nb.GetAtomicNum() > 1 and nb.GetIdx() not in backbone_set
    ]


def _build_domain_string(
    at_annotation,
    beta_state,
    kr_subtype="",
    *,
    is_starter=False,
    is_terminal=False,
):
    """Format canonical PKS module domain string."""
    at_norm = _normalize_label(at_annotation)
    if is_starter:
        return f"AT({at_norm}) - ACP"
    state = str(beta_state).strip()
    parts = ["KS", f"AT({at_norm})"]
    if state in {"β-hydroxy", "β hydroxyl"}:
        kr_str = f"KR({kr_subtype})" if kr_subtype and kr_subtype != "U" else "KR"
        parts.append(kr_str)
    elif state == "enoyl":
        parts.extend(["KR", "DH"])
    elif state == "saturated":
        parts.extend(["KR", "DH", "ER"])
    elif state in {"β-keto", "β keto"} and kr_subtype in {"C1", "C2"}:
        parts.append(f"KR({kr_subtype})")
    parts.append("ACP")
    if is_terminal:
        parts.append("TE")
    return " - ".join(parts)


def _kr_subtype(
    mol,
    alpha_idx,
    beta_idx,
    beta_state,
    backbone=None,
    alpha_pos=None,
    beta_pos=None,
):
    """Read KR subtype (A1, A2, B1, B2, A, B, C1, C2, U) in the Lip1-TE backbone frame."""
    state = str(beta_state).strip()
    if state in {"β-hydroxy", "β hydroxyl"}:
        subtype, _, _ = _infer_beta_hydroxy_kr_type(
            mol,
            alpha_idx,
            beta_idx,
            backbone=backbone,
            alpha_pos=alpha_pos,
            beta_pos=beta_pos,
        )
        return _pad_malonyl_digit(subtype)
    if state in {"β-keto", "β keto"}:
        backbone_set = set(backbone) if backbone is not None else set()
        subs = _alpha_substituents(mol, alpha_idx, backbone_set)
        at_guess = "mmal" if subs else "mal"
        kr_type, _, _, _ = _infer_beta_keto_c_type(mol, alpha_idx, at_guess)
        alpha_cfg, _, _ = _infer_alpha_config(mol, alpha_idx, backbone, alpha_pos)
        if kr_type in {"C1", "C2"} and alpha_cfg in {"1", "2"}:
            kr_type = f"C{alpha_cfg}"
        elif alpha_cfg in {"1", "2"}:
            kr_type = f"C{alpha_cfg}"
        return _pad_malonyl_digit(kr_type)
    return "U"


PR = sys.modules[__name__]

# --- chain simulation, DH/ER geometry, TE release (formerly chemistry-first) ---

def _normalize_label(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def _parse_kr_type(kr_annotation):
    s = _normalize_label(kr_annotation)
    if not s.startswith("type "):
        return ""
    return s.split(",")[0].replace("type ", "").strip()


def _atom_cip_code(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    return atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""


def _permutation_parity(perm):
    """Return 0 for even permutation, 1 for odd."""
    visited = [False] * len(perm)
    parity = 0
    for i in range(len(perm)):
        if visited[i]:
            continue
        j = i
        cycle_len = 0
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            cycle_len += 1
        parity += cycle_len - 1
    return parity % 2


def _effective_chirality(mol, center_idx, canonical_neighbors):
    """Interpret an RDKit chiral tag in a backbone-defined neighbor order."""
    atom = mol.GetAtomWithIdx(center_idx)
    tag = atom.GetChiralTag()
    if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        base = 1
    elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        base = -1
    else:
        return 0

    stored = [nb.GetIdx() for nb in atom.GetNeighbors()]
    if len(stored) != 3 or len(canonical_neighbors) != 3:
        return 0

    idx_map = {v: i for i, v in enumerate(stored)}
    if not all(c in idx_map for c in canonical_neighbors):
        return 0
    perm = [idx_map[c] for c in canonical_neighbors]
    if _permutation_parity(perm) == 1:
        base = -base
    return base


def _find_oh_oxygen(mol, atom_idx):
    for nb in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        if nb.GetAtomicNum() != 8:
            continue
        bond = mol.GetBondBetweenAtoms(atom_idx, nb.GetIdx())
        if bond is not None and bond.GetBondTypeAsDouble() == 1.0:
            return nb.GetIdx()
    return None


# The alpha substituent (a carbon branch, or oxygen for methoxy- and
# hydroxymalonyl) is defined once, in `predict_reduction._alpha_substituents`;
# the stereo frame (C1 side, starter side, substituent) does not depend on its element.


# --------------------------------------------------------------------------
# Backbone-frame stereochemistry.
#
# Every stereocentre and olefin below is described relative to the polyketide
# backbone rather than by CIP letters, because a CIP label for one and the same
# spatial arrangement flips as the surrounding chain grows.  The retro pass
# reads a configuration in this frame and the forward simulation writes it back
# in the same frame, so the two are inverses of each other by construction.
#
# alpha convention (identical to the KR subtype convention of
# Keiser et al., J. Am. Chem. Soc. 2025, 147, 42237, Figure 1):
#     +1  ->  subtype 1, unepimerized alpha-substituent
#     -1  ->  subtype 2, epimerized alpha-substituent
# anchored against that paper's four authentic Lip1-TE 3-hydroxyacid standards
# (A1 = 2R,3S; A2 = 2S,3S; B1 = 2R,3R; B2 = 2S,3R).
#
# beta convention:
#     +1  ->  B family (3R in the Lip1-TE frame)
#     -1  ->  A family (3S in the Lip1-TE frame)
# --------------------------------------------------------------------------

KR_FAMILY_BETA = {"a": -1, "b": 1}
KR_SUBTYPE_ALPHA = {"1": 1, "2": -1}
ALPHA_LABEL_BY_SIGN = {1: "1", -1: "2"}


def _set_backbone_chirality(mol, center_idx, canonical_neighbors, want):
    """Force `_effective_chirality(mol, center_idx, canonical_neighbors)` to equal `want`.

    Returns True if the requested configuration could be installed.
    """
    atom = mol.GetAtomWithIdx(center_idx)
    if want == 0:
        # The evidence does not fix this centre, so do not assert one.
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        return True
    if want not in (1, -1):
        return False
    if atom.GetDegree() != 3:
        return False
    for tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
        atom.SetChiralTag(tag)
        if _effective_chirality(mol, center_idx, canonical_neighbors) == want:
            return True
    atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return False


def _set_alkene_relation(mol, idx_a, idx_b, ref_a, ref_b, relation):
    """Place `ref_a` and `ref_b` cis or trans across the idx_a=idx_b double bond."""
    bond = mol.GetBondBetweenAtoms(idx_a, idx_b)
    if bond is None or bond.GetBondTypeAsDouble() != 2.0:
        return False
    if relation not in ("trans", "cis"):
        return False
    if mol.GetBondBetweenAtoms(idx_a, ref_a) is None:
        return False
    if mol.GetBondBetweenAtoms(idx_b, ref_b) is None:
        return False
    # RDKit expects the reference atoms in (begin-side, end-side) order.
    if bond.GetBeginAtomIdx() == idx_a:
        bond.SetStereoAtoms(ref_a, ref_b)
    elif bond.GetBeginAtomIdx() == idx_b:
        bond.SetStereoAtoms(ref_b, ref_a)
    else:
        return False
    bond.SetStereo(
        Chem.BondStereo.STEREOTRANS if relation == "trans" else Chem.BondStereo.STEREOCIS
    )
    Chem.SetDoubleBondNeighborDirections(mol)
    return True


def _read_alkene_relation(mol, idx_a, idx_b, ref_a, ref_b):
    """Report whether `ref_a` and `ref_b` sit cis or trans across idx_a=idx_b.

    Decided by re-writing the bond both ways and keeping the one that reproduces
    the molecule's own canonical SMILES, which sidesteps the begin/end atom
    bookkeeping of RDKit's stereo-atom pairs.
    """
    bond = mol.GetBondBetweenAtoms(idx_a, idx_b)
    if bond is None or bond.GetBondTypeAsDouble() != 2.0:
        return ""
    if bond.GetStereo() in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY):
        return ""
    try:
        target = Chem.MolToSmiles(mol)
    except Exception:
        return ""
    for relation in ("trans", "cis"):
        probe = Chem.Mol(mol)
        if not _set_alkene_relation(probe, idx_a, idx_b, ref_a, ref_b, relation):
            return ""
        try:
            if Chem.MolToSmiles(probe) == target:
                return relation
        except Exception:
            return ""
    return ""


def _main_chain_from_c1(mol, c1_idx):
    """Longest carbon path starting at C1; the acyl backbone of an acyclic chain."""
    best = []

    def walk(idx, previous, path):
        nonlocal best
        path = path + [idx]
        if len(path) > len(best):
            best = path
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetAtomicNum() == 6 and nb.GetIdx() not in path:
                walk(nb.GetIdx(), idx, path)

    walk(c1_idx, -1, [])
    return best


def _acyl_c1_index(mol):
    """Index of the thioester (or released carboxyl) carbon."""
    for patt in ("[CX3](=O)[SX2,SX1]", "[CX3](=O)[OX2H1]", "[CX3](=O)[OX1-]"):
        match = mol.GetSubstructMatch(Chem.MolFromSmarts(patt))
        if match:
            return match[0]
    return None


def _load_observed_smiles_map(path, smiles_col="target_smiles"):
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if not {"bgc_id", smiles_col}.issubset(df.columns):
        return {}
    out = {}
    for _, r in df.iterrows():
        bgc = str(r.get("bgc_id", "")).strip()
        smi = str(r.get(smiles_col, "")).strip()
        if bgc and smi and smi != "nan":
            out[bgc] = smi
    return out


def _largest_ring_size(mol):
    rings = mol.GetRingInfo().AtomRings()
    return max((len(r) for r in rings), default=0)


def _infer_te_release(observed_smiles):
    if not observed_smiles or str(observed_smiles).strip() in {"", "nan"}:
        return "", "low", "no observed product SMILES"

    mol = Chem.MolFromSmiles(str(observed_smiles).strip())
    if mol is None:
        return "", "low", "observed product SMILES invalid"

    patt_thioester = Chem.MolFromSmarts("[CX3](=O)[S]")
    patt_carboxylic_acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    patt_ring_ester = Chem.MolFromSmarts("[CX3;R](=O)[O;R]")
    patt_ring_lactam = Chem.MolFromSmarts("[CX3;R](=O)[N;R]")

    has_thioester = bool(mol.HasSubstructMatch(patt_thioester))
    has_carboxylic_acid = bool(mol.HasSubstructMatch(patt_carboxylic_acid))
    has_ring_ester = bool(mol.HasSubstructMatch(patt_ring_ester))
    has_ring_lactam = bool(mol.HasSubstructMatch(patt_ring_lactam))
    has_ring_release = has_ring_ester or has_ring_lactam
    has_macrocycle = _largest_ring_size(mol) >= 8

    if has_thioester and not has_carboxylic_acid and not has_ring_release:
        return "", "high", "terminal thioester signature dominates release evidence"
    if has_ring_release:
        conf = "high" if has_macrocycle else "medium"
        return "cyclic", conf, "ring-embedded ester/lactam supports cyclization"
    if has_carboxylic_acid:
        return "non-cyclic", "high", "carboxylic-acid release signature supports hydrolysis"
    return "", "low", "release mechanism unresolved from observed product"


def _load_module_at_map(path):
    p = Path(path)
    if not p.exists():
        return {}

    df = pd.read_csv(p)
    if not {"bgc_id", "module_num", "at_annotation"}.issubset(df.columns):
        return {}

    out = {}
    for r in df.itertuples(index=False):
        bgc_id = getattr(r, "bgc_id", None)
        module_num = int(getattr(r, "module_num", -1))
        at_ann = _normalize_label(getattr(r, "at_annotation", ""))
        if not bgc_id or module_num < 0:
            continue
        out.setdefault(str(bgc_id), {})
        out[str(bgc_id)][module_num] = at_ann
    return out


def _infer_module_at_fallback(mol, backbone, starter_carbons, starter_preference):
    hypotheses = pe._enumerate_starter_hypotheses(len(backbone), starter_carbons)
    if not hypotheses:
        return {}

    evaluated = pe._evaluate_starter_hypotheses(
        mol, backbone, hypotheses, starter_preference=starter_preference
    )
    best = pe._apply_c3_branch_signature_override(evaluated)
    best = pe._apply_mal_vs_2metbut_override(best, evaluated)
    best = pe._apply_mmal_vs_mal_offbyone_override(
        best, mol, backbone, starter_carbons
    )
    out = {0: _normalize_label(best.get("starter_unit", ""))}
    extenders = best.get("extenders", [])
    out.update({i: _normalize_label(x) for i, x in enumerate(extenders, start=1)})
    return out


def _load_starter_smiles(path):
    df = pd.read_csv(path, sep="\t")
    if not {"shortName", "smiles"}.issubset(df.columns):
        return {}
    out = {}
    for _, r in df.iterrows():
        name = _normalize_label(r.get("shortName"))
        smiles = str(r.get("smiles", "")).strip()
        if name and smiles:
            out[name] = smiles
    for alias, key in {"acetyl": "acetyl-coa", "isobut": "isobutmal",
                       "2metbut": "2metbutmal", "chc": "chc-coa",
                       "trans": "trans-1,2-cpda"}.items():
        if alias not in out and key in out:
            out[alias] = out[key]
    return out


# Fixed executor substrates missing from the original generator dictionary.
# These are a finite whitelist, never fragments extracted from the query.
# Keep the historical ClusterCAD code for the branched seven-carbon side chain;
# it is constitutionally distinct from hexmal and must not be aliased to it.
ADDITIONAL_EXTENDER_SMILES = {
    "propmal": "CCCC(C(=O)O)C(=O)[S]",
    "isopropmal": "CC(C)C(C(=O)O)C(=O)[S]",
    "2metbutmal": "CCC(C)C(C(=O)O)C(=O)[S]",
    "3-me-hexmal": "CCC(C)CCCC(C(=O)O)C(=O)[S]",
}


def _load_extender_fragments(path):
    p = Path(path)
    if not p.exists():
        return {}

    df = pd.read_csv(p, sep="\t")
    if not {"shortName", "smiles"}.issubset(df.columns):
        return {}

    rm_coa = AllChem.ReactionFromSmarts(
        "[O:1]=[C:2]([O:3])[C:4][C:5](=[O:6])[S:7]>>"
        "[*:10]-[C:2](=[O:1])[C:4][C:5](=[O:6])[S:7].[O:3]"
    )

    out = {}
    records = df.to_dict("records")
    existing = {_normalize_label(r.get("shortName")) for r in records}
    records.extend({"shortName": name, "smiles": smiles}
                   for name, smiles in ADDITIONAL_EXTENDER_SMILES.items() if name not in existing)
    for r in records:
        name = _normalize_label(r.get("shortName"))
        mol = Chem.MolFromSmiles(str(r.get("smiles", "")).strip())
        if not name or mol is None:
            continue
        try:
            products = rm_coa.RunReactants((mol,))
        except Exception:
            continue
        if not products:
            continue
        frag = products[0][0]
        try:
            Chem.SanitizeMol(frag)
        except Exception:
            continue
        out[name] = frag
    return out


def _reaction_templates():
    """Constitution-only templates.

    None of these carry stereochemistry.  KR, DH and ER outcomes are stereo-
    specific, but the specificity is a property of the individual domain, not of
    the reaction, so it cannot live in a shared SMARTS.  Each template builds the
    right connectivity and the caller then stamps the configuration recorded for
    that module (see `_apply_module_stereochemistry`).
    """
    condense = AllChem.ReactionFromSmarts(
        "[#6:10][C:1](=[O:2])[S:3].[*:4][C:5]~[C:6]>>"
        "[#6:10][C:5]~[C:6].[*:4].[C:1](=[O:2])[S:3]"
    )
    kr_generic = AllChem.ReactionFromSmarts(
        "[*:1][C:2](=[O:3])[C:4][C:5](=[O:6])[S:7]>>"
        "[*:1][C:2]([O:3])[C:4][C:5](=[O:6])[S:7]"
    )
    dh = AllChem.ReactionFromSmarts(
        "[*:1][C:2]([O:3])[C:4][C:6](=[O:7])[S:8]>>"
        "[*:1][CH1:2]=[CH1:4][C:6](=[O:7])[S:8].[O:3]"
    )
    # alpha-substituted modules: the alpha carbon keeps its branch, so no H there
    dh_alt = AllChem.ReactionFromSmarts(
        "[*:1][C:2]([O:3])[C:4][C:6](=[O:7])[S:8]>>"
        "[*:1][CH1:2]=[CH0:4][C:6](=[O:7])[S:8].[O:3]"
    )
    er = AllChem.ReactionFromSmarts(
        "[*:1][C:2]=[C:3][C:4](=[O:5])[S:6]>>"
        "[*:1][CH2:2][CH2:3][C:4](=[O:5])[S:6]"
    )
    # alpha-substituted modules: one H on the alpha carbon, configuration set later
    er_alt = AllChem.ReactionFromSmarts(
        "[*:1][C:2]=[C:3][C:4](=[O:5])[S:6]>>"
        "[*:1][CH2:2][CH1:3][C:4](=[O:5])[S:6]"
    )
    return {
        "condense": condense,
        "kr_generic": kr_generic,
        "dh": dh,
        "dh_alt": dh_alt,
        "er": er,
        "er_alt": er_alt,
    }


def _run_reaction_first(rxn, reactants):
    try:
        products = rxn.RunReactants(tuple(reactants))
    except Exception:
        return None
    if not products:
        return None
    prod = products[0][0]
    try:
        Chem.SanitizeMol(prod)
    except Exception:
        return None
    return prod


def _apply_non_cyclic_te_release(chain, c1_idx):
    """Hydrolytic release: the carrier sulfur on C1 becomes the acid oxygen.

    The sulfur atom is edited in place, so every atom keeps its index and its
    backbone label.
    """
    released = Chem.RWMol(chain)
    sulfur = [nb.GetIdx() for nb in released.GetAtomWithIdx(c1_idx).GetNeighbors()
              if nb.GetAtomicNum() == 16]
    if len(sulfur) != 1:
        return None
    atom = released.GetAtomWithIdx(sulfur[0])
    atom.SetAtomicNum(8)
    atom.SetNumRadicalElectrons(0)
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(False)
    try:
        Chem.SanitizeMol(released)
    except Exception:
        return None
    return released.GetMol()


# --------------------------------------------------------------------------
# Identity of the assembly chain.
#
# The starter and every extender are known, so the polyketide backbone is never
# searched for: it is carried through each condensation.  Backbone atoms are
# marked with isotope numbers (C1 first) and the marks are removed before any
# SMILES is produced.  Isotopes are used because RDKit copies a reactant atom's
# isotope onto its product atom even where the template maps that atom, whereas
# custom atom properties are dropped there.  A step whose product does not still
# contain the whole bonded, labelled backbone is rejected instead of guessed.
# --------------------------------------------------------------------------

BACKBONE_LABEL_BASE = 1000
_FRAGMENT_BACKBONE = Chem.MolFromSmarts("[#0]-[CX3](=O)-[CX4]-[CX3](=O)-[SX2,SX1]")


def _starter_backbone(chain):
    """[C1, alpha]: the only starter atoms that later steps address."""
    c1 = _acyl_c1_index(chain)
    if c1 is None:
        return None
    carbons = [nb.GetIdx() for nb in chain.GetAtomWithIdx(c1).GetNeighbors()
               if nb.GetAtomicNum() == 6]
    return [c1, carbons[0]] if len(carbons) == 1 else None


def _label_extender(fragment, first_label):
    """Copy of an extender fragment with its [thioester C, alpha, carbonyl C] labelled.

    The three labels are returned in that order, which is their order from C1 once
    the fragment has condensed onto the chain.
    """
    matches = fragment.GetSubstructMatches(_FRAGMENT_BACKBONE)
    if len(matches) != 1:
        return None, None
    _, carbonyl, _, alpha, thio_carbon = matches[0][:5]
    labelled = Chem.Mol(fragment)
    labels = [first_label, first_label + 1, first_label + 2]
    for label, idx in zip(labels, (thio_carbon, alpha, carbonyl)):
        labelled.GetAtomWithIdx(idx).SetIsotope(label)
    return labelled, labels


def _backbone_indices(mol, labels):
    """Current atom indices of the labelled backbone in C1-first order, or None.

    None unless every label occurs exactly once and consecutive backbone atoms are
    bonded.
    """
    marked = {atom.GetIsotope(): atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsotope()}
    if sorted(marked) != sorted(labels) or sum(a.GetIsotope() > 0 for a in mol.GetAtoms()) != len(labels):
        return None
    indices = [marked[label] for label in labels]
    if any(mol.GetBondBetweenAtoms(a, b) is None for a, b in zip(indices, indices[1:])):
        return None
    return indices


def _without_backbone_labels(mol):
    plain = Chem.Mol(mol)
    for atom in plain.GetAtoms():
        atom.SetIsotope(0)
    return plain


def _run_on_backbone(rxn, chain, labels):
    """Apply a one-reactant template; (product, backbone indices) or (None, None)."""
    product = _run_reaction_first(rxn, [chain])
    backbone = _backbone_indices(product, labels) if product is not None else None
    return (product, backbone) if backbone else (None, None)


def _normalize_extender_key(ext_ann):
    aliases = {
        "mxmal_acp": "mxmal",
        "dcpat": "dcp",
        "3-me-hxmal": "3-me-hexmal",
    }
    return aliases.get(_normalize_label(ext_ann), _normalize_label(ext_ann))


def _apply_module_stereochemistry(chain, records, chain_atoms):
    """Re-stamp every configuration recorded so far onto the current chain.

    `chain_atoms` is the assembly backbone (atom indices, C1 first) carried through
    the condensations, never a path searched for in the finished molecule.
    `records` holds entries keyed by their offset along that backbone measured
    from C1, so a record survives later extensions simply by having its offset
    shifted.  Re-applying all of them after every step means nothing is lost if a
    reaction template drops a tag or a bond direction along the way.
    """
    backbone_set = set(chain_atoms)
    n = len(chain_atoms)

    for rec in records:
        offset = rec["offset"]

        if rec["kind"] == "alkene":
            if offset + 2 >= n or offset < 1:
                continue
            idx_a, idx_b = chain_atoms[offset], chain_atoms[offset + 1]
            _set_alkene_relation(
                chain, idx_a, idx_b,
                chain_atoms[offset - 1], chain_atoms[offset + 2],
                rec["relation"],
            )
            continue

        # tetrahedral centre carrying a branch (alpha) or a hydroxyl (beta)
        if offset < 1 or offset + 1 >= n:
            continue
        center = chain_atoms[offset]
        neighbours = [chain_atoms[offset - 1], chain_atoms[offset + 1]]
        if rec["kind"] == "alpha":
            substituents = PR._alpha_substituents(chain, center, backbone_set)
            third = substituents[0] if len(substituents) == 1 else None
        else:
            third = _find_oh_oxygen(chain, center)
        if third is None:
            continue
        _set_backbone_chirality(chain, center, neighbours + [third], rec["sign"])

    Chem.SetDoubleBondNeighborDirections(chain)
    return chain


def _simulate_bgc_intermediates(starter_ann, module_rows, starter_smiles_map, extender_fragments, rxn_tpl):
    starter_key = _normalize_label(starter_ann)
    starter_smiles = starter_smiles_map.get(starter_key, "")
    if not starter_smiles:
        return "", {}

    chain = Chem.MolFromSmiles(starter_smiles)
    if chain is None:
        return "", {}
    starter_backbone = _starter_backbone(chain)
    if starter_backbone is None:
        return "", {}
    # Backbone atoms (C1 first) are labelled once and followed through every step.
    labels = list(range(BACKBONE_LABEL_BASE, BACKBONE_LABEL_BASE + len(starter_backbone)))
    for label, idx in zip(labels, starter_backbone):
        chain.GetAtomWithIdx(idx).SetIsotope(label)
    backbone = starter_backbone
    next_label = labels[-1] + 1

    # Stereochemistry installed by earlier modules, carried forward by offset.
    records = []
    by_module_smiles = {}

    for r in sorted(module_rows, key=lambda x: int(x["module_num"])):
        module_num = int(r["module_num"])
        ext_key = _normalize_extender_key(r.get("at_annotation", ""))
        frag = extender_fragments.get(ext_key)
        if frag is None:
            by_module_smiles[module_num] = ""
            continue

        frag, frag_labels = _label_extender(frag, next_label)
        if frag is None:
            by_module_smiles[module_num] = ""
            continue
        next_label += len(frag_labels)
        extended = _run_reaction_first(rxn_tpl["condense"], [chain, frag])
        # The new C1 is the extender's thioester carbon; the old C1 is released.
        extended_labels = frag_labels + labels[1:]
        extended_backbone = _backbone_indices(extended, extended_labels) if extended is not None else None
        if extended_backbone is None:
            by_module_smiles[module_num] = ""
            continue
        chain, labels, backbone = extended, extended_labels, extended_backbone
        # every earlier centre just moved two carbons further from C1
        for rec in records:
            rec["offset"] += 2

        kr_active = _normalize_label(r.get("kr_activity_pred")) == "active"
        dh_active = _normalize_label(r.get("dh_activity_pred")) == "active"
        er_active = _normalize_label(r.get("er_activity_pred")) == "active"
        kr_type = _parse_kr_type(r.get("kr_annotation_pred", ""))

        if not (kr_active and kr_type not in {"", "c1", "c2"}) and not dh_active:
            # No reducing KR ran, so nothing re-sets the alpha centre except a
            # C-type epimerase. Reproduce what the observed product shows there,
            # and clear the extender's own tag when it shows nothing.
            alpha_cfg = _normalize_label(r.get("kr_alpha_pred"))
            if alpha_cfg in KR_SUBTYPE_ALPHA:
                records.append(dict(kind="alpha", offset=1, sign=KR_SUBTYPE_ALPHA[alpha_cfg]))
            elif alpha_cfg == "unresolved":
                records.append(dict(kind="alpha", offset=1, sign=0))

        if kr_active and kr_type not in {"", "c1", "c2"}:
            reduced, reduced_backbone = _run_on_backbone(rxn_tpl["kr_generic"], chain, labels)
            if reduced is not None:
                chain, backbone = reduced, reduced_backbone
                if not dh_active:
                    # The KR sets both centres and, with no DH downstream, both
                    # survive into the product.
                    family = KR_FAMILY_BETA.get(kr_type[:1])
                    if family is not None:
                        records.append(dict(kind="beta", offset=2, sign=family))
                    subtype = KR_SUBTYPE_ALPHA.get(kr_type[1:2])
                    if subtype is not None:
                        records.append(dict(kind="alpha", offset=1, sign=subtype))

        if dh_active:
            dehydrated, dehydrated_backbone = _run_on_backbone(rxn_tpl["dh"], chain, labels)
            if dehydrated is None:
                dehydrated, dehydrated_backbone = _run_on_backbone(rxn_tpl["dh_alt"], chain, labels)
            if dehydrated is not None:
                chain, backbone = dehydrated, dehydrated_backbone
                # A DH erases whatever the KR set at C2 and C3 and replaces it
                # with an olefin, so drop those records before adding the new one.
                records = [rec for rec in records if rec["offset"] > 2]
                relation = "cis" if _normalize_label(r.get("dh_geometry_pred")) == "z" else "trans"
                records.append(dict(kind="alkene", offset=1, relation=relation))

        if er_active:
            reduced, reduced_backbone = _run_on_backbone(rxn_tpl["er"], chain, labels)
            if reduced is None:
                reduced, reduced_backbone = _run_on_backbone(rxn_tpl["er_alt"], chain, labels)
            if reduced is not None:
                chain, backbone = reduced, reduced_backbone
                # The olefin is gone; the ER now owns the alpha configuration.
                records = [rec for rec in records if rec["offset"] > 2]
                sign = KR_SUBTYPE_ALPHA.get(_normalize_label(r.get("er_stereo_pred")))
                if sign is not None:
                    records.append(dict(kind="alpha", offset=1, sign=sign))

        chain = _apply_module_stereochemistry(chain, records, backbone)

        if _normalize_label(r.get("te_annotation_pred")) == "non-cyclic":
            released = _apply_non_cyclic_te_release(chain, backbone[0])
            if released is not None:
                chain = _apply_module_stereochemistry(released, records, backbone)

        by_module_smiles[module_num] = Chem.MolToSmiles(_without_backbone_labels(chain), isomericSmiles=True)

    return Chem.MolToSmiles(Chem.MolFromSmiles(starter_smiles), isomericSmiles=True), by_module_smiles


def _atom_has_carbonyl_oxygen(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    for bond in atom.GetBonds():
        nb = bond.GetOtherAtom(atom)
        if nb.GetAtomicNum() == 8 and bond.GetBondTypeAsDouble() == 2.0:
            return True
    return False


def _atom_has_single_bond_oxygen(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    for bond in atom.GetBonds():
        nb = bond.GetOtherAtom(atom)
        if nb.GetAtomicNum() == 8 and bond.GetBondTypeAsDouble() == 1.0:
            return True
    return False


def _classify_beta_state(mol, alpha_idx, beta_idx):
    if _atom_has_carbonyl_oxygen(mol, beta_idx):
        return "β keto"
    ab_bond = mol.GetBondBetweenAtoms(alpha_idx, beta_idx)
    if ab_bond is not None and ab_bond.GetBondTypeAsDouble() == 2.0:
        return "enoyl"
    if _atom_has_single_bond_oxygen(mol, beta_idx):
        return "β hydroxyl"
    if ab_bond is not None and ab_bond.GetBondTypeAsDouble() == 1.0:
        return "saturated"
    return "unknown"


def _infer_beta_hydroxy_kr_type(
    mol,
    alpha_idx,
    beta_idx,
    *,
    backbone=None,
    alpha_pos=None,
    beta_pos=None,
):
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    if backbone is not None and alpha_pos is not None and beta_pos is not None:
        backbone_set = set(backbone)
        oh_idx = _find_oh_oxygen(mol, beta_idx)
        if oh_idx is not None and beta_pos < len(backbone):
            beta_ch = _effective_chirality(
                mol,
                beta_idx,
                [backbone[beta_pos - 2], backbone[beta_pos], oh_idx],
            )
            if beta_ch != 0:
                family = "B" if beta_ch == 1 else "A"
                # The subtype digit describes the alpha substituent. A malonyl
                # module has none, so the digit is undefined there and inventing
                # one would present a default as evidence. Report the family alone.
                alpha_label = ""
                note = "backbone-aware β-hydroxyl stereochemistry supports the KR family"
                substituents = PR._alpha_substituents(mol, alpha_idx, backbone_set)
                if not substituents:
                    note += "; α carries no substituent so the subtype digit is undefined"
                elif len(substituents) > 1:
                    note += "; α carries several substituents so the subtype digit is omitted"
                elif 2 <= alpha_pos < len(backbone):
                    alpha_ch = _effective_chirality(
                        mol,
                        alpha_idx,
                        [backbone[alpha_pos - 2], backbone[alpha_pos], substituents[0]],
                    )
                    if alpha_ch != 0:
                        alpha_label = ALPHA_LABEL_BY_SIGN[alpha_ch]
                        note = "backbone-aware β-hydroxyl stereochemistry supports KR subtype"
                    else:
                        note += "; α configuration unresolved so the subtype digit is omitted"
                return f"{family}{alpha_label}", "high", note

    alpha_tag = str(mol.GetAtomWithIdx(alpha_idx).GetChiralTag())
    beta_tag = str(mol.GetAtomWithIdx(beta_idx).GetChiralTag())

    if beta_tag == "CHI_UNSPECIFIED":
        return "U", "low", "β-hydroxyl observed but β stereochemistry is unresolved"
    if beta_tag == "CHI_TETRAHEDRAL_CCW":
        if alpha_tag == "CHI_TETRAHEDRAL_CCW":
            return "A2", "high", "β CCW and α CCW match BioPKS A2 stereochemistry"
        if alpha_tag == "CHI_TETRAHEDRAL_CW":
            return "A1", "high", "β CCW and α CW match BioPKS A1 stereochemistry"
        return "A1", "medium", "β CCW supports A-family; α unresolved so defaulting to A1"
    if beta_tag == "CHI_TETRAHEDRAL_CW":
        if alpha_tag == "CHI_TETRAHEDRAL_CCW":
            return "B2", "high", "β CW and α CCW match BioPKS B2 stereochemistry"
        if alpha_tag == "CHI_TETRAHEDRAL_CW":
            return "B1", "high", "β CW and α CW match BioPKS B1 stereochemistry"
        return "B1", "medium", "β CW supports B-family; α unresolved so defaulting to B1"
    return "U", "low", "β stereochemistry tag is not interpretable"


def _infer_beta_keto_c_type(mol, alpha_idx, at_ann):
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    alpha_tag = str(mol.GetAtomWithIdx(alpha_idx).GetChiralTag())
    alpha_cip = _atom_cip_code(mol, alpha_idx)
    at_norm = _normalize_label(at_ann)

    if at_norm == "mmal":
        if alpha_cip == "R":
            return "C2", "active", "high", "β-keto + mmal with α R-CIP supports epimerizing C2"
        if alpha_cip == "S":
            return "C1", "active", "high", "β-keto + mmal with α S-CIP supports non-epimerizing C1"
        if alpha_tag == "CHI_TETRAHEDRAL_CCW":
            return "C2", "active", "medium", "β-keto + mmal with α CCW tag supports C2"
        if alpha_tag == "CHI_TETRAHEDRAL_CW":
            return "C1", "active", "medium", "β-keto + mmal with α CW tag supports C1"

    if at_norm == "mal":
        return "U", "inactive", "high", "β-keto excludes reducing KR/DH/ER; C1 vs no-KR is unresolved from product alone"

    if alpha_cip == "R":
        return "C2", "active", "medium", "β-keto with α R-CIP supports epimerizing C2"
    if alpha_cip == "S":
        return "C1", "active", "medium", "β-keto with α S-CIP supports non-epimerizing C1"
    if alpha_tag == "CHI_TETRAHEDRAL_CCW":
        return "C2", "active", "low", "β-keto with α CCW tag weakly supports C2"
    if alpha_tag == "CHI_TETRAHEDRAL_CW":
        return "C1", "active", "low", "β-keto with α CW tag weakly supports C1"
    return "U", "inactive", "high", "β-keto excludes reducing KR/DH/ER; C-type is unresolved from product alone"


def _infer_dh_geometry(mol, alpha_idx, beta_idx, backbone, alpha_pos, beta_pos):
    """Geometry of a surviving DH olefin, read off the observed product.

    Falls back to E, the near-universal outcome of PKS dehydratases, when the
    observed structure leaves the double bond unspecified.
    """
    if backbone is None or alpha_pos is None or beta_pos is None:
        return "E", "low", "no backbone context; defaulting to the E outcome of PKS DH domains"
    if alpha_pos < 2 or beta_pos >= len(backbone):
        return "E", "low", "olefin sits at a chain end; defaulting to the E outcome of PKS DH domains"

    relation = _read_alkene_relation(
        mol, alpha_idx, beta_idx, backbone[alpha_pos - 2], backbone[beta_pos]
    )
    if relation == "trans":
        return "E", "high", "observed olefin is E across the backbone"
    if relation == "cis":
        return "Z", "high", "observed olefin is Z across the backbone"
    return "E", "low", "observed olefin has no geometry; defaulting to the E outcome of PKS DH domains"


def _infer_alpha_config(mol, alpha_idx, backbone, alpha_pos):
    """Alpha-substituent configuration of a module, read off the observed product.

    Reported in the KR subtype convention: "1" unepimerized, "2" epimerized.
    A malonyl-derived module has no alpha substituent and reports "achiral";
    a module whose alpha centre the observed structure leaves open reports
    "unresolved", and the forward simulation then asserts nothing there.

    Used for every module regardless of which domain owns the centre: an ER in a
    full reductive loop, the KR itself in a KR-only module, or the epimerase
    half of a non-reducing C-type KR.
    """
    if backbone is None or alpha_pos is None or alpha_pos < 2 or alpha_pos >= len(backbone):
        return "unresolved", "low", "no backbone context for the alpha carbon"

    substituents = PR._alpha_substituents(mol, alpha_idx, set(backbone))
    if not substituents:
        return "achiral", "high", "alpha carbon carries no substituent, so the ER sets no stereocentre"
    if len(substituents) > 1:
        return "unresolved", "low", "alpha carbon carries several substituents; the single-substituent frame does not apply"

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    sign = _effective_chirality(
        mol, alpha_idx, [backbone[alpha_pos - 2], backbone[alpha_pos], substituents[0]]
    )
    if sign == 0:
        return "unresolved", "low", "alpha substituent present but its configuration is unspecified"
    return (
        ALPHA_LABEL_BY_SIGN[sign],
        "high",
        "backbone-aware alpha configuration recovered from the observed product",
    )


def _infer_module_activities(
    mol,
    alpha_idx,
    beta_idx,
    at_ann,
    *,
    backbone=None,
    alpha_pos=None,
    beta_pos=None,
):
    beta_state = _classify_beta_state(mol, alpha_idx, beta_idx)

    if beta_state == "β keto":
        kr_type, kr_act, conf, note = _infer_beta_keto_c_type(mol, alpha_idx, at_ann)
        alpha_cfg, alpha_conf, alpha_note = _infer_alpha_config(
            mol, alpha_idx, backbone, alpha_pos
        )
        # A C-type KR only epimerizes, so its subtype is the alpha configuration.
        # Prefer the backbone-frame reading over the CIP heuristic, which shifts
        # with the surrounding chain.
        if kr_type in {"C1", "C2"} and alpha_cfg in {"1", "2"}:
            kr_type = f"C{alpha_cfg}"
            conf = alpha_conf
            note = "backbone-aware alpha configuration supports the C-type subtype"
        return {
            "beta_state": beta_state,
            "kr_annotation_pred": f"type {kr_type}, {kr_act}" if kr_type != "U" else "type U, inactive",
            "kr_activity_pred": kr_act,
            "dh_activity_pred": "inactive",
            "er_activity_pred": "inactive",
            "kr_type_confidence": conf,
            "kr_note": note,
            "kr_subtype_evidence": "stereochemistry" if kr_type in {"C1", "C2"} else "product_only",
            "kr_subtype_confidence": conf,
            "kr_inference_basis": "β keto chemistry-first",
            "kr_alpha_pred": alpha_cfg,
            "kr_alpha_confidence": alpha_conf,
            "kr_alpha_note": alpha_note,
            "dh_geometry_pred": "",
            "dh_geometry_confidence": "",
            "dh_geometry_note": "",
            "er_stereo_pred": "",
            "er_stereo_confidence": "",
            "er_stereo_note": "",
        }

    if beta_state == "β hydroxyl":
        kr_type, conf, note = _infer_beta_hydroxy_kr_type(
            mol,
            alpha_idx,
            beta_idx,
            backbone=backbone,
            alpha_pos=alpha_pos,
            beta_pos=beta_pos,
        )
        return {
            "beta_state": beta_state,
            "kr_annotation_pred": f"type {kr_type}, active",
            "kr_activity_pred": "active",
            "dh_activity_pred": "inactive",
            "er_activity_pred": "inactive",
            "kr_type_confidence": conf,
            "kr_note": note,
            "kr_subtype_evidence": "stereochemistry" if kr_type != "U" else "product_only",
            "kr_subtype_confidence": conf,
            "kr_inference_basis": "β hydroxy chemistry-first",
            "kr_alpha_pred": "",
            "kr_alpha_confidence": "",
            "kr_alpha_note": "",
            "dh_geometry_pred": "",
            "dh_geometry_confidence": "",
            "dh_geometry_note": "",
            "er_stereo_pred": "",
            "er_stereo_confidence": "",
            "er_stereo_note": "",
        }

    if beta_state == "enoyl":
        dh_geom, dh_conf, dh_note = _infer_dh_geometry(
            mol, alpha_idx, beta_idx, backbone, alpha_pos, beta_pos
        )
        return {
            "beta_state": beta_state,
            "kr_annotation_pred": "type U, active",
            "kr_activity_pred": "active",
            "dh_activity_pred": "active",
            "er_activity_pred": "inactive",
            "kr_type_confidence": "low",
            "kr_note": "enoyl requires KR+DH; KR subtype is not recoverable from the end state",
            "kr_subtype_evidence": "product_only",
            "kr_subtype_confidence": "low",
            "kr_inference_basis": "enoyl chemistry-first",
            "kr_alpha_pred": "",
            "kr_alpha_confidence": "",
            "kr_alpha_note": "",
            "dh_geometry_pred": dh_geom,
            "dh_geometry_confidence": dh_conf,
            "dh_geometry_note": dh_note,
            "er_stereo_pred": "",
            "er_stereo_confidence": "",
            "er_stereo_note": "",
        }

    if beta_state == "saturated":
        er_stereo, er_conf, er_note = _infer_alpha_config(mol, alpha_idx, backbone, alpha_pos)
        return {
            "beta_state": beta_state,
            "kr_annotation_pred": "type U, active",
            "kr_activity_pred": "active",
            "dh_activity_pred": "active",
            "er_activity_pred": "active",
            "kr_type_confidence": "low",
            "kr_note": "saturated beta carbon requires KR+DH+ER; KR subtype is not recoverable from the end state",
            "kr_subtype_evidence": "product_only",
            "kr_subtype_confidence": "low",
            "kr_inference_basis": "saturated chemistry-first",
            "kr_alpha_pred": "",
            "kr_alpha_confidence": "",
            "kr_alpha_note": "",
            # The DH olefin is consumed here, so its geometry leaves no trace in
            # the product; the ER, not the KR, owns the surviving alpha centre.
            "dh_geometry_pred": "E",
            "dh_geometry_confidence": "low",
            "dh_geometry_note": "olefin consumed by the ER; geometry not observable in the product",
            "er_stereo_pred": er_stereo,
            "er_stereo_confidence": er_conf,
            "er_stereo_note": er_note,
        }

    return {
        "beta_state": beta_state,
        "kr_annotation_pred": "type U, inactive",
        "kr_activity_pred": "inactive",
        "dh_activity_pred": "inactive",
        "er_activity_pred": "inactive",
        "kr_type_confidence": "low",
        "kr_note": "beta state could not be classified",
        "kr_subtype_evidence": "product_only",
        "kr_subtype_confidence": "low",
        "kr_inference_basis": "unknown",
        "kr_alpha_pred": "",
        "kr_alpha_confidence": "",
        "kr_alpha_note": "",
        "dh_geometry_pred": "",
        "dh_geometry_confidence": "",
        "dh_geometry_note": "",
        "er_stereo_pred": "",
        "er_stereo_confidence": "",
        "er_stereo_note": "",
    }


def _predicted_domains(kr_activity_pred, dh_activity_pred, er_activity_pred, te_annotation_pred):
    parts = ["KS", "AT"]
    if dh_activity_pred == "active":
        parts.append("DH")
    if er_activity_pred == "active":
        parts.append("ER")
    if kr_activity_pred == "active":
        parts.append("KR")
    parts.append("ACP")
    if _normalize_label(te_annotation_pred):
        parts.append("TE")
    return ", ".join(parts)


def _to_raw_like_rows(detail_rows, module_at_map, smiles_by_bgc, starter_chain_by_bgc):
    by_bgc = {}
    for r in detail_rows:
        by_bgc.setdefault(r["bgc_id"], []).append(r)

    out = []
    for bgc_id, rows in by_bgc.items():
        rows = sorted(rows, key=lambda x: int(x["module_num"]))
        cluster_name = rows[0]["cluster_name"]
        at_map = module_at_map.get(str(bgc_id), {})
        starter_chain = starter_chain_by_bgc.get(str(bgc_id), "")

        out.append(
            {
                "bgc_id": bgc_id,
                "cluster_name": cluster_name,
                "module": f"{cluster_name} predicted module 0",
                "domains": "KS, AT, ACP",
                "kr_annotation": "",
                "at_annotation": _normalize_label(at_map.get(0, "")),
                "dh_annotation": "",
                "dh_geometry": "",
                "er_annotation": "",
                "er_stereo": "",
                "kr_alpha": "",
                "te_annotation": "",
                "intermediate_smiles": starter_chain,
            }
        )

        for r in rows:
            module_num = int(r["module_num"])
            out.append(
                {
                    "bgc_id": r["bgc_id"],
                    "cluster_name": r["cluster_name"],
                    "module": f"{cluster_name} predicted module {module_num}",
                    "domains": _predicted_domains(
                        r["kr_activity_pred"],
                        r["dh_activity_pred"],
                        r["er_activity_pred"],
                        r.get("te_annotation_pred", ""),
                    ),
                    "kr_annotation": r["kr_annotation_pred"],
                    "at_annotation": _normalize_label(at_map.get(module_num, r["at_annotation"])),
                    "dh_annotation": r["dh_activity_pred"],
                    "dh_geometry": (
                        r.get("dh_geometry_pred", "")
                        if r["dh_activity_pred"] == "active" and r["er_activity_pred"] != "active"
                        else ""
                    ),
                    "er_annotation": r["er_activity_pred"],
                    "er_stereo": (
                        r.get("er_stereo_pred", "") if r["er_activity_pred"] == "active" else ""
                    ),
                    "kr_alpha": r.get("kr_alpha_pred", ""),
                    "te_annotation": _normalize_label(r.get("te_annotation_pred", "")),
                    "intermediate_smiles": r.get("intermediate_smiles_pred", ""),
                }
            )

    out_df = pd.DataFrame(out)
    cols = [
        "bgc_id",
        "cluster_name",
        "module",
        "domains",
        "kr_annotation",
        "at_annotation",
        "dh_annotation",
        "dh_geometry",
        "er_annotation",
        "er_stereo",
        "kr_alpha",
        "te_annotation",
        "intermediate_smiles",
    ]
    if out_df.empty:
        return pd.DataFrame(columns=cols)
    return out_df[cols]



# Keiser et al. 2025 authentic Lip1-TE 3-hydroxyacid standards (Figure 1 / Materials).
LIP1_TE_STANDARDS = {
    "A1": "CC(C)[C@H](O)[C@@H](C)C(=O)O",  # (2R,3S)
    "A2": "CC(C)[C@H](O)[C@H](C)C(=O)O",   # (2S,3S)
    "B1": "CC(C)[C@@H](O)[C@@H](C)C(=O)O",  # (2R,3R)
    "B2": "CC(C)[C@@H](O)[C@H](C)C(=O)O",   # (2S,3R)
}

# SI Table S1: KR type and native-substrate SMILES for first-module 3-hydroxy thioesters.
TABLE_S1_STANDARDS = [
    ("Lip1", "A2", "CC(C)[C@H](O)[C@H](C)C(=O)[S]"),
    ("Pik1", "B2", "CC[C@@H](O)[C@H](C)C(=O)[S]"),
    ("Amp1", "A2", "C[C@H](O)[C@H](C)C(=O)[S]"),
    ("PM11", "A2", "CC[C@@H]([C@@H](C([S])=O)C)O"),
    ("Baf1", "B1", "CC(C)[C@@H](O)[C@@H](C)C(=O)[S]"),
    ("Tyl1", "B1", "CC[C@@H](O)[C@@H](C)C(=O)[S]"),
    ("Ave1", "B1", "CC(C)[C@H]([C@H](C([S])=O)C)O"),
    ("Cha1", "B1", "C[C@@H](O)[C@@H](C)C(=O)[S]"),
    ("Mei1", "B1", "C[C@@H](O)[C@@H](C)C(=O)[S]"),
    ("Debs", "B2", "CC[C@@H](O)[C@H](C)C(=O)[S]"),
    ("Meg1", "B2", "CC[C@@H](O)[C@H](C)C(=O)[S]"),
    ("Lan1", "B2", "CCC(C)[C@@H](O)[C@H](C)C(=O)[S]"),
    ("Acu1", "A1", "CC[C@H](O)[C@@H](C)C(=O)[S]"),
    ("Ang1", "A1", "CC[C@H](O)[C@@H](C)C(=O)[S]"),
]

_PR_STATE = {
    "β keto": "β-keto",
    "β hydroxyl": "β-hydroxy",
    "β-keto": "β-keto",
    "β-hydroxy": "β-hydroxy",
    "enoyl": "enoyl",
    "saturated": "saturated",
}


def _pad_malonyl_digit(subtype: str) -> str:
    """Paper A/B/C (no α-methyl) → ClusterCad A1/B1/C1."""
    if subtype in {"A", "B", "C"}:
        return f"{subtype}1"
    return subtype


def assign_kr_subtype(beta_state, pr_subtype, dh_geometry=""):
    """Map a module end-state onto a Keiser Figure 1 KR subtype.

    Returns (subtype, confidence, note, evidence).
    """
    state = _PR_STATE.get(str(beta_state).strip(), str(beta_state).strip())

    if state == "β-hydroxy":
        subtype = _pad_malonyl_digit(pr_subtype or "U")
        if subtype == "U":
            return (
                "U",
                "low",
                "β-OH present but its configuration is unspecified",
                "observed_stereo",
            )
        return (
            subtype,
            "high",
            "Keiser et al. 2025 Figure 1 / SI Table S1: β-OH and α-substituent "
            "read in the Lip1-TE backbone frame",
            "observed_stereo",
        )

    if state == "β-keto":
        subtype = _pad_malonyl_digit(pr_subtype or "C")
        if subtype == "U":
            subtype = "C1"
        return (
            subtype,
            "high",
            "Keiser et al. 2025: C-type KR, no NADPH reduction; C1 vs C2 is "
            "the α-substituent when one is present",
            "observed_stereo",
        )

    if state in {"enoyl", "saturated"}:
        return (
            "U",
            "high",
            "Keiser Figure 1 defines A1/A2/B1/B2 only while the β-OH remains; "
            "DH/ER have removed it so the KR subtype is not in the product",
            "not_readable",
        )

    return (
        "U",
        "low",
        "β oxidation state could not be classified",
        "product_only",
    )


def _stereo_backbone(mol, backbone, beta_pos):
    """Extend a truncated chain so toward-starter exists at the β-carbon.

    Acetyl-starter 3-hydroxy thioesters in SI Table S1 (Amp1, Cha1, Mei1) are
    four carbons; some backbone walks drop the terminal methyl, which makes
    the Lip1-TE neighbor order undefined.
    """
    if backbone is not None and beta_pos is not None and beta_pos < len(backbone):
        return backbone
    c1 = _acyl_c1_index(mol) if mol is not None else None
    if c1 is None:
        return backbone
    alt = _main_chain_from_c1(mol, c1)
    if alt and beta_pos is not None and beta_pos < len(alt):
        return alt
    return backbone


def infer_module_reduction(
    mol,
    alpha_idx,
    beta_idx,
    at_ann,
    *,
    backbone=None,
    alpha_pos=None,
    beta_pos=None,
):
    """Chemistry-first activities/geometry, with a Figure 1 KR subtype when readable."""
    pred = _infer_module_activities(
        mol,
        alpha_idx,
        beta_idx,
        at_ann,
        backbone=backbone,
        alpha_pos=alpha_pos,
        beta_pos=beta_pos,
    )

    pr_state = _PR_STATE.get(pred["beta_state"], pred["beta_state"])
    pr_subtype = "U"
    read_bb = _stereo_backbone(mol, backbone, beta_pos)
    if (
        mol is not None
        and read_bb is not None
        and alpha_pos is not None
        and beta_pos is not None
        and pr_state in {"β-keto", "β-hydroxy"}
    ):
        pr_subtype = PR._kr_subtype(
            mol, alpha_idx, beta_idx, pr_state, read_bb, alpha_pos, beta_pos
        )

    subtype, conf, note, evidence = assign_kr_subtype(
        pred["beta_state"],
        pr_subtype,
        pred.get("dh_geometry_pred", ""),
    )
    kr_act = pred["kr_activity_pred"]
    pred["kr_annotation_pred"] = f"type {subtype}, {kr_act}"
    pred["kr_type_confidence"] = conf
    pred["kr_note"] = note
    pred["kr_subtype_evidence"] = evidence
    pred["kr_subtype_confidence"] = conf
    pred["kr_inference_basis"] = note
    return pred


def _read_one_extension(smiles, at_ann="mmal"):
    mol, backbone, status = pe._core_backbone_from_smiles(smiles)
    if status != "ok" or mol is None or backbone is None or len(backbone) < 3:
        return None, f"backbone failed ({status})"
    pred = infer_module_reduction(
        mol, backbone[1], backbone[2], at_ann,
        backbone=backbone, alpha_pos=2, beta_pos=3,
    )
    return _parse_kr_type(pred["kr_annotation_pred"]).upper(), None


def _self_test():
    """Pin hydroxy reads to Keiser Figure 1 / SI Table S1; DH/ER stay unread."""
    failures = []
    for want, smiles in LIP1_TE_STANDARDS.items():
        got, err = _read_one_extension(smiles)
        if err:
            failures.append(f"Lip1-TE {want}: {err}")
        elif got != want:
            failures.append(f"Lip1-TE {want}: got type {got}")

    for name, want, smiles in TABLE_S1_STANDARDS:
        got, err = _read_one_extension(smiles)
        if err:
            failures.append(f"Table S1 {name}: {err}")
        elif got != want:
            failures.append(f"Table S1 {name} ({want}): got type {got}")

    observed = "C/C=C/C=C/C[C@H](C)C[C@H](C)C[C@@H](C)[C@H](O)CC(=O)O"
    mol, backbone, status = pe._core_backbone_from_smiles(observed)
    at_by_module = {1: "mal", 2: "mal", 3: "mmal", 4: "mmal", 5: "mmal", 6: "mal"}
    # Figure 1 is readable only on M6 (β-OH remains). M1–M5 are DH/ER.
    expect = {1: "U", 2: "U", 3: "U", 4: "U", 5: "U", 6: "A1"}
    n = max(at_by_module)
    for module_num, want in expect.items():
        alpha_pos = 2 * (n - module_num + 1)
        beta_pos = alpha_pos + 1
        pred = infer_module_reduction(
            mol,
            backbone[alpha_pos - 1],
            backbone[beta_pos - 1],
            at_by_module[module_num],
            backbone=backbone,
            alpha_pos=alpha_pos,
            beta_pos=beta_pos,
        )
        got = _parse_kr_type(pred["kr_annotation_pred"]).upper()
        if got != want:
            failures.append(
                f"Collins #12 M{module_num}: got {pred['kr_annotation_pred']}, expected type {want}"
            )
    return failures


def _starter_smi_candidates(cli_path: str) -> list[Path]:
    return [
        Path(cli_path),
        STAGE_DIR / "data" / "starters.smi",
        PROJECT_ROOT / "data" / "starters.smi",
        PROJECT_ROOT / "RetroTideV2" / "bcs" / "data" / "starters.smi",
        PROJECT_ROOT / "RetroTide" / "bcs" / "data" / "starters.smi",
        PROJECT_ROOT / "BioPKS-Pipeline" / "biopks_pipeline" / "retrotide" / "data" / "starters.smi",
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_csv", default=str(STAGE_DIR / "top20_genai_linearized.csv"))
    ap.add_argument("--smiles_col", default="linearized_smiles")
    ap.add_argument(
        "--extender_predictions_csv",
        default=str(STAGE_DIR / "top20_genai_extender_predictions.csv"),
    )
    ap.add_argument(
        "--output_csv",
        default=str(STAGE_DIR / "top20_genai_reduction_kr_subtype.csv"),
    )
    ap.add_argument("--max_visits", type=int, default=500000)
    ap.add_argument("--early_k", type=int, default=10)
    ap.add_argument("--starters_smi", default=str(RETROTIDE_BCS_DATA / "starters.smi"))
    ap.add_argument("--retrotide_extenders_smi", default=str(RETROTIDE_BCS_DATA / "extenders.smi"))
    ap.add_argument("--observed_smiles_csv", default=str(STAGE_DIR / "top20_genai_input.csv"))
    ap.add_argument("--observed_smiles_col", default="target_smiles")
    ap.add_argument(
        "--starter_whitelist",
        default="mal,mmal,prop,acetyl,isobut,2metbut,pyr,cyclopentene,trans,chc",
    )
    ap.add_argument(
        "--starter_carbons_mode",
        default="backbone_path",
        choices=["backbone_path", "carbon_count"],
    )
    ap.add_argument("--self-test", action="store_true", help="Run Lip1-TE and Collins #12 checks and exit")
    args = ap.parse_args()

    if args.self_test:
        failures = _self_test()
        if failures:
            for line in failures:
                print(f"FAIL  {line}")
            return 1
        print("ok    Lip1-TE standards and Collins #12 KR subtypes")
        return 0

    df = pd.read_csv(args.input_csv)
    module_at_map = _load_module_at_map(args.extender_predictions_csv)
    smiles_by_bgc = {
        str(r.get("bgc_id")): str(r.get(args.smiles_col, ""))
        for _, r in df.iterrows()
        if pd.notna(r.get("bgc_id"))
    }

    starter_smiles_map = {}
    extender_fragments = {}
    rxn_tpl = None
    for starter_path in _starter_smi_candidates(args.starters_smi):
        if starter_path.exists():
            starter_smiles_map = _load_starter_smiles(starter_path)
            ext_path = starter_path.with_name("extenders.smi")
            if args.retrotide_extenders_smi and Path(args.retrotide_extenders_smi).exists():
                ext_path = Path(args.retrotide_extenders_smi)
            extender_fragments = _load_extender_fragments(ext_path)
            rxn_tpl = _reaction_templates()
            break

    starter_carbons = {}
    starter_preference = []
    for starter_path in _starter_smi_candidates(args.starters_smi):
        if starter_path.exists():
            starter_carbons = pe._load_starter_carbon_counts(
                starter_path, carbon_mode=args.starter_carbons_mode
            )
            starter_carbons = pe._filter_starter_carbons(starter_carbons, args.starter_whitelist)
            starter_preference = pe._starter_preference_from_whitelist(
                args.starter_whitelist, starter_carbons
            )
            break

    observed_smiles_map = _load_observed_smiles_map(
        Path(args.observed_smiles_csv), smiles_col=args.observed_smiles_col
    )

    rows = []
    for _, row in df.iterrows():
        bgc_id = row.get("bgc_id")
        cluster_name = row.get("cluster_name")
        smiles = str(row.get(args.smiles_col, ""))
        if pd.isna(smiles) or not smiles or smiles == "nan":
            continue

        mol, backbone, status = pe._core_backbone_from_smiles(
            smiles, max_visits=args.max_visits, early_k=args.early_k
        )
        if status != "ok" or mol is None or not backbone:
            continue

        bgc_key = str(bgc_id)
        module_at = module_at_map.get(bgc_key, {})
        ext_by_module = {k: v for k, v in module_at.items() if int(k) >= 1}
        if not ext_by_module and starter_carbons:
            module_at = _infer_module_at_fallback(
                mol, backbone, starter_carbons, starter_preference
            )
            ext_by_module = {k: v for k, v in module_at.items() if int(k) >= 1}
        if not ext_by_module:
            continue
        if bgc_key not in module_at_map:
            module_at_map[bgc_key] = dict(module_at)

        extension_modules = max(ext_by_module.keys())
        observed_smiles = observed_smiles_map.get(bgc_key, "")
        te_annotation_pred, te_confidence_pred, te_note_pred = _infer_te_release(
            observed_smiles if observed_smiles else smiles
        )

        for module_num in range(1, extension_modules + 1):
            alpha_pos = 2 * (extension_modules - module_num + 1)
            beta_pos = alpha_pos + 1
            if beta_pos > len(backbone):
                continue
            at_ann = _normalize_label(
                module_at.get(module_num, ext_by_module.get(module_num, ""))
            )
            module_pred = infer_module_reduction(
                mol,
                backbone[alpha_pos - 1],
                backbone[beta_pos - 1],
                at_ann,
                backbone=backbone,
                alpha_pos=alpha_pos,
                beta_pos=beta_pos,
            )
            rows.append(
                {
                    "bgc_id": bgc_id,
                    "cluster_name": cluster_name,
                    "module": f"extender module {module_num}",
                    "module_num": int(module_num),
                    "at_annotation": at_ann,
                    **module_pred,
                    "te_annotation_pred": te_annotation_pred if module_num == extension_modules else "",
                    "te_confidence_pred": te_confidence_pred if module_num == extension_modules else "",
                    "te_note_pred": te_note_pred if module_num == extension_modules else "",
                }
            )

    rows_by_bgc = {}
    for r in rows:
        rows_by_bgc.setdefault(str(r["bgc_id"]), []).append(r)

    starter_chain_by_bgc = {}
    if starter_smiles_map and extender_fragments and rxn_tpl is not None:
        for bgc_key, module_rows in rows_by_bgc.items():
            starter_ann = _normalize_label(module_at_map.get(bgc_key, {}).get(0, ""))
            starter_chain, module_smiles = _simulate_bgc_intermediates(
                starter_ann,
                module_rows,
                starter_smiles_map,
                extender_fragments,
                rxn_tpl,
            )
            starter_chain_by_bgc[bgc_key] = starter_chain
            for r in module_rows:
                r["intermediate_smiles_pred"] = module_smiles.get(int(r["module_num"]), "")

    output_df = _to_raw_like_rows(rows, module_at_map, smiles_by_bgc, starter_chain_by_bgc)
    out_path = Path(args.output_csv)
    output_df.to_csv(out_path, index=False)
    print(f"[predict] generated module rows: {len(output_df)}")
    print(f"[predict] saved predictions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
