import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRIPT_DIR = Path(__file__).resolve().parent
RETROTIDE_BCS_DATA = (
    SCRIPT_DIR / "data"
    if (SCRIPT_DIR / "data" / "starters.smi").exists()
    else (
        DATA_DIR
        if (DATA_DIR / "starters.smi").exists()
        else (
            PROJECT_ROOT / "RetroTideV2" / "bcs" / "data"
            if (PROJECT_ROOT / "RetroTideV2" / "bcs" / "data").exists()
            else PROJECT_ROOT / "RetroTide" / "bcs" / "data"
        )
    )
)

def _import_carbon_backbone():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import carbon_backbone as cb  # type: ignore

    return cb


cb = _import_carbon_backbone()


def _core_backbone_from_smiles(
    smiles, *, exclude_sugar_rings=True, max_visits=500000, early_k=10
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, "invalid_smiles"

    _, c1_idx = cb.find_c1(mol)
    if c1_idx is None:
        return mol, None, "missing_c1"

    carbon_set = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}
    if exclude_sugar_rings:
        sugar_atoms = cb.NXBackboneAnalyzer._find_sugar_ring_atoms(mol)
        carbon_set -= sugar_atoms

    if c1_idx not in carbon_set:
        return mol, None, "c1_not_in_carbon_set"

    G = cb.NXBackboneAnalyzer._build_carbon_graph(mol, carbon_set)
    if c1_idx not in G:
        return mol, None, "c1_not_in_graph"

    reachable = cb.nx.node_connected_component(G, c1_idx)
    if not reachable:
        return mol, None, "no_reachable_carbons"

    G = G.subgraph(reachable).copy()

    backbone, _ = cb.NXBackboneAnalyzer._best_path_from_c1_dfs(
        G, c1_idx, max_visits=max_visits, early_k=early_k
    )
    if not backbone:
        return mol, None, "no_backbone"

    return mol, backbone, "ok"


def _core_backbone_candidates_from_smiles(
    smiles,
    *,
    exclude_sugar_rings=True,
    max_visits=500000,
    early_k=10,
    top_k=8,
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, "invalid_smiles"

    _, c1_idx = cb.find_c1(mol)
    if c1_idx is None:
        return mol, None, "missing_c1"

    candidates, _ = cb.find_pks_backbone_candidates(
        mol,
        c1_idx,
        exclude_sugar_rings=exclude_sugar_rings,
        max_visits=max_visits,
        early_k=early_k,
        top_k=top_k,
    )
    if not candidates:
        return mol, None, "no_backbone"
    return mol, candidates, "ok"


def _is_methyl(atom):
    return (
        atom.GetAtomicNum() == 6
        and atom.GetDegree() == 1
        and atom.GetTotalNumHs() == 3
    )


def _collect_branch_atoms(mol, start_idx, backbone_set, alpha_idx):
    visited = set()
    stack = [start_idx]
    while stack:
        idx = stack.pop()
        if idx in visited:
            continue
        visited.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx == alpha_idx or nb_idx in backbone_set:
                continue
            if nb_idx not in visited:
                stack.append(nb_idx)
    return visited


def _branch_stats(mol, branch_atoms):
    carbon_atoms = [
        idx
        for idx in branch_atoms
        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
    ]
    n_carbons = len(carbon_atoms)
    has_hetero = any(
        mol.GetAtomWithIdx(idx).GetAtomicNum() not in (1, 6) for idx in branch_atoms
    )
    has_halogen = any(
        mol.GetAtomWithIdx(idx).GetAtomicNum() in (9, 17, 35, 53)
        for idx in branch_atoms
    )
    has_aromatic = any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in branch_atoms)

    has_cc_double = False
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in branch_atoms and b in branch_atoms:
            if bond.GetBondTypeAsDouble() == 2.0:
                if (
                    mol.GetAtomWithIdx(a).GetAtomicNum() == 6
                    and mol.GetAtomWithIdx(b).GetAtomicNum() == 6
                ):
                    has_cc_double = True
                    break

    terminal_methyls = sum(
        1 for idx in carbon_atoms if _is_methyl(mol.GetAtomWithIdx(idx))
    )

    branch_points = 0
    for idx in carbon_atoms:
        atom = mol.GetAtomWithIdx(idx)
        carbon_neighbors = 0
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if (
                nb_idx in branch_atoms
                and mol.GetAtomWithIdx(nb_idx).GetAtomicNum() == 6
            ):
                carbon_neighbors += 1
        if carbon_neighbors >= 3:
            branch_points += 1

    return {
        "n_carbons": n_carbons,
        "has_hetero": has_hetero,
        "has_halogen": has_halogen,
        "has_aromatic": has_aromatic,
        "has_cc_double": has_cc_double,
        "terminal_methyls": terminal_methyls,
        "branch_points": branch_points,
    }


def _classify_alpha_substituent(mol, alpha_idx, backbone_set):
    alpha_atom = mol.GetAtomWithIdx(alpha_idx)

    non_backbone_neighbors = [
        nb for nb in alpha_atom.GetNeighbors() if nb.GetIdx() not in backbone_set
    ]
    if not non_backbone_neighbors:
        return "hydrogen", "mal", "mAT"

    # O-substituent logic: ether/acetal/ketal or O-CH3 => methoxy-derived branch; hydroxyl => hmal
    o_neighbors = [nb for nb in non_backbone_neighbors if nb.GetAtomicNum() == 8]
    if o_neighbors:
        for o_atom in o_neighbors:
            for nb2 in o_atom.GetNeighbors():
                if nb2.GetIdx() == alpha_idx:
                    continue
                if nb2.GetAtomicNum() == 6:
                    return "o_substituent", "mxmal_ACP", "mxmalAT"
        return "hydroxyl", "hmal", "hmalAT"

    branch_atoms = set()
    for nb in non_backbone_neighbors:
        branch_atoms |= _collect_branch_atoms(mol, nb.GetIdx(), backbone_set, alpha_idx)

    stats = _branch_stats(mol, branch_atoms)

    if stats["has_aromatic"] and stats["has_halogen"]:
        return "aromatic_halogen", "DCP", "DCPAT"

    n_carbons = stats["n_carbons"]

    if n_carbons == 1:
        return "methyl", "mmal", "mmAT"
    if n_carbons == 2:
        return "ethyl", "emal", "emAT"
    if n_carbons == 3 and stats["has_cc_double"]:
        return "allyl", "allylmal", "allylmalAT"
    if n_carbons == 3 and not stats["has_hetero"] and not stats["has_aromatic"]:
        if stats["terminal_methyls"] == 2:
            return "isopropyl", "isopropmal", "isopropmalAT"
        return "propyl", "propmal", "propmalAT"
    if n_carbons == 4:
        branch_root_degree = sum(nb.GetIdx() in branch_atoms
                                 for root in non_backbone_neighbors for nb in root.GetNeighbors())
        if stats["terminal_methyls"] >= 2 and branch_root_degree == 1:
            return "isobutyl", "isobutmal", "isobutmalAT"
        if stats["branch_points"] == 0 and stats["terminal_methyls"] == 1:
            return "butyl", "butmal", "butmalAT"
        return "branched_butyl", "2metbutmal", "2metbutmalAT"
    if n_carbons == 6:
        return "hexyl", "hexmal", "hexmalAT"
    if n_carbons >= 7 and stats["branch_points"] > 0 and stats["terminal_methyls"] >= 2:
        if not stats["has_hetero"]:
            return "branched_alkyl", "3-me-hexmal", "3-me-hexmalAT"

    return "other_carbon", "unknown", "unknown"


def _predict_extenders(mol, backbone):
    backbone_set = set(backbone)
    extenders = []
    for pos, idx in enumerate(backbone, start=1):
        if pos == 1:
            continue
        if pos % 2 == 0:
            _, extender, _ = _classify_alpha_substituent(mol, idx, backbone_set)
            extenders.append(extender)
    return extenders


def _predict_extenders_with_extension_boundary(mol, backbone, extension_modules):
    """
    Predict extenders only for alpha-carbons inside the extender-derived region.

    If E is known, extender-derived carbons are C1..C(2E). Therefore valid
    alpha positions are even positions <= 2E.
    """
    if extension_modules is None or extension_modules <= 0:
        return []

    backbone_set = set(backbone)
    extenders_from_c1 = []
    max_extender_pos = 2 * int(extension_modules)

    for pos, idx in enumerate(backbone, start=1):
        if pos == 1 or pos % 2 != 0:
            continue
        if pos > max_extender_pos:
            continue
        _, extender, _ = _classify_alpha_substituent(mol, idx, backbone_set)
        extenders_from_c1.append(extender)

    # C1->distal order is reverse module order; invert to module 1..E order.
    return list(reversed(extenders_from_c1))


def _enumerate_starter_hypotheses(backbone_len, starter_carbons):
    """
    Enumerate all stoichiometrically feasible (starter, E) hypotheses from:
        n = S + 2E
    where n is backbone_len, S is starter carbon count, and E is extension count.
    """
    if backbone_len is None or backbone_len <= 0:
        return []

    hypotheses = []
    for starter_unit, starter_len in sorted(starter_carbons.items()):
        if starter_len is None or starter_len <= 0:
            continue
        remainder = int(backbone_len) - int(starter_len)
        if remainder < 0:
            continue
        if remainder % 2 != 0:
            continue
        extension_modules = remainder // 2
        hypotheses.append(
            {
                "starter_unit": starter_unit,
                "starter_len": int(starter_len),
                "extension_modules": int(extension_modules),
            }
        )

    return hypotheses


def _score_extender_hypothesis(extenders):
    """
    Lower score is better.
    Primary objective: minimize unresolved alpha-site assignments.
    """
    unknown_count = sum(1 for x in extenders if _normalize_label(x) == "unknown")
    known_count = len(extenders) - unknown_count
    return (unknown_count, -known_count)


def _starter_fragment_features(mol, backbone, extension_modules):
    """
    Compute coarse structural features of the starter-derived fragment
    (backbone positions after 2E carbons from C1).
    """
    if extension_modules is None or extension_modules < 0:
        return {
            "branch_carbons": 0,
            "has_hetero": False,
            "has_aromatic": False,
            "has_ring": False,
            "has_cc_double": False,
        }

    backbone_set = set(backbone)
    starter_backbone = backbone[2 * int(extension_modules) :]
    starter_set = set(starter_backbone)

    branch_atoms = set()
    for idx in starter_backbone:
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in backbone_set:
                continue
            branch_atoms |= _collect_branch_atoms(mol, nb_idx, backbone_set, idx)

    if branch_atoms:
        stats = _branch_stats(mol, branch_atoms)
    else:
        stats = {
            "n_carbons": 0,
            "has_hetero": False,
            "has_aromatic": False,
            "has_cc_double": False,
        }

    has_ring = any(mol.GetAtomWithIdx(idx).IsInRing() for idx in starter_backbone)

    has_cc_double_backbone = False
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in starter_set and b in starter_set and bond.GetBondTypeAsDouble() == 2.0:
            if (
                mol.GetAtomWithIdx(a).GetAtomicNum() == 6
                and mol.GetAtomWithIdx(b).GetAtomicNum() == 6
            ):
                has_cc_double_backbone = True
                break

    has_aromatic_backbone = any(
        mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in starter_backbone
    )

    return {
        "branch_carbons": int(stats["n_carbons"]),
        "has_hetero": bool(stats["has_hetero"]),
        "has_aromatic": bool(stats["has_aromatic"] or has_aromatic_backbone),
        "has_ring": bool(has_ring),
        "has_cc_double": bool(stats["has_cc_double"] or has_cc_double_backbone),
    }


def _starter_compatibility_penalty(starter_unit, features):
    """
    Lower is better. Penalize starter assignments that conflict with obvious
    structural signatures in the inferred starter fragment.
    """
    penalty = 0

    # Simple short acyl starters are unlikely when the inferred starter
    # fragment is itself ring-embedded.
    if starter_unit in {"mal", "mmal", "prop", "acetyl"} and features["has_ring"]:
        penalty += 5

    # Branched starters should carry side-chain carbon evidence.
    if starter_unit in {"isobut", "2metbut"} and features["branch_carbons"] == 0:
        penalty += 2

    # Pyr starter should show aromatic starter-fragment character.
    if starter_unit == "pyr" and not features["has_aromatic"]:
        penalty += 6

    # Cyclopentene starter should show ring + unsaturation.
    if starter_unit == "cyclopentene":
        if not features["has_ring"]:
            penalty += 4
        if not features["has_cc_double"]:
            penalty += 3

    # Distinguish C7 pair (trans-1,2-CPDA vs CHC) by hetero signal.
    if starter_unit == "trans-1,2-cpda" and not features["has_hetero"]:
        penalty += 3
    if starter_unit == "chc" and features["has_hetero"]:
        penalty += 3

    return penalty


def _evaluate_starter_hypotheses(mol, backbone, hypotheses, starter_preference=None):
    if starter_preference is None:
        starter_preference = {}

    evaluated = []
    for hyp in hypotheses:
        extenders = _predict_extenders_with_extension_boundary(
            mol, backbone, hyp["extension_modules"]
        )
        score = _score_extender_hypothesis(extenders)
        starter_features = _starter_fragment_features(
            mol, backbone, hyp["extension_modules"]
        )
        starter_penalty = _starter_compatibility_penalty(
            hyp["starter_unit"], starter_features
        )
        evaluated.append(
            {
                "starter_unit": hyp["starter_unit"],
                "starter_len": hyp["starter_len"],
                "extension_modules": hyp["extension_modules"],
                "extenders": extenders,
                "score": score,
                "starter_penalty": starter_penalty,
                "starter_features": starter_features,
            }
        )

    # Deterministic ranking:
    # 1) fewer unknown extender calls
    # 2) lower starter-fragment incompatibility penalty
    # 3) more known extender calls
    # 4) prefer larger starter when equally plausible
    # 5) whitelist order preference
    # 6) lexical starter name tie-break
    evaluated.sort(
        key=lambda x: (
            x["score"][0],
            x["starter_penalty"],
            x["score"][1],
            -x["starter_len"],
            starter_preference.get(x["starter_unit"], 10**9),
            x["starter_unit"],
        )
    )
    return evaluated


def _backbone_branch_penalty(mol, backbone):
    """
    Penalize carbon branches hanging off the proposed backbone. Lower is better.
    """
    backbone_set = set(backbone)
    off_path = set()
    for idx in backbone:
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb.GetAtomicNum() != 6 or nb_idx in backbone_set:
                continue
            off_path.add(nb_idx)
    return len(off_path)


def _rank_joint_candidate(candidate, starter_preference):
    best = candidate["best_hypothesis"]
    return (
        best["score"][0],
        best["starter_penalty"],
        candidate["backbone_rank"],
        candidate["backbone_branch_penalty"],
        best["score"][1],
        -best["starter_len"],
        starter_preference.get(best["starter_unit"], 10**9),
        best["starter_unit"],
        tuple(candidate["backbone"]),
    )


def _choose_best_backbone_hypothesis(
    mol,
    backbones,
    starter_carbons,
    starter_preference=None,
):
    if starter_preference is None:
        starter_preference = {}

    joint_candidates = []
    for backbone_rank, backbone in enumerate(backbones, start=1):
        hypotheses = _enumerate_starter_hypotheses(len(backbone), starter_carbons)
        if not hypotheses:
            continue
        evaluated = _evaluate_starter_hypotheses(
            mol, backbone, hypotheses, starter_preference=starter_preference
        )
        best = _apply_c3_branch_signature_override(evaluated)
        best = _apply_mal_vs_2metbut_override(best, evaluated)
        best = _apply_mmal_vs_mal_offbyone_override(best, mol, backbone, starter_carbons)
        if not best:
            continue
        joint_candidates.append(
            {
                "backbone_rank": backbone_rank,
                "backbone": backbone,
                "backbone_len": len(backbone),
                "backbone_branch_penalty": _backbone_branch_penalty(mol, backbone),
                "best_hypothesis": best,
                "evaluated": evaluated,
            }
        )

    if not joint_candidates:
        return None, []

    joint_candidates.sort(key=lambda x: _rank_joint_candidate(x, starter_preference))
    chosen = joint_candidates[0]

    candidate_rows = []
    for joint_rank, candidate in enumerate(joint_candidates, start=1):
        for hyp_rank, hyp in enumerate(candidate["evaluated"], start=1):
            enriched = dict(hyp)
            enriched["joint_rank"] = joint_rank
            enriched["backbone_rank"] = candidate["backbone_rank"]
            enriched["backbone_len"] = candidate["backbone_len"]
            enriched["backbone_branch_penalty"] = candidate["backbone_branch_penalty"]
            candidate_rows.append((hyp_rank, enriched))

    return chosen, candidate_rows


def _apply_c3_branch_signature_override(evaluated):
    """
    Resolve a common C3 tie mode:
      - mmal/prop often outrank isobut only because they imply one extra extender.
      - if the inferred C3 starter fragment has branch-carbon evidence and the
        first extender looks methylmalonyl-like, prefer isobut.
    """
    if not evaluated:
        return None

    best = evaluated[0]
    if best["starter_unit"] not in {"mmal", "prop"}:
        return best

    features = best.get("starter_features", {})
    branch_carbons = int(features.get("branch_carbons", 0))
    if branch_carbons <= 0:
        return best

    extenders = best.get("extenders", [])
    ext1 = extenders[0] if extenders else None
    if ext1 not in {"mmal", "mxmal_ACP"}:
        return best

    for hyp in evaluated:
        if hyp["starter_unit"] == "isobut":
            return hyp
    return best


def _leading_label_run(values, label):
    run = 0
    for x in values:
        if x == label:
            run += 1
        else:
            break
    return run


def _apply_mal_vs_2metbut_override(best, evaluated):
    """
    Resolve a persistent mal vs 2metbut boundary degeneracy:
    when mal captures one extra leading mmal-like module, a branched
    2metbut starter can be more plausible for the same product scaffold.
    """
    if not best or best["starter_unit"] != "mal":
        return best

    by_name = {h["starter_unit"]: h for h in evaluated}
    alt = by_name.get("2metbut")
    if alt is None:
        return best

    ext = best.get("extenders", [])
    if not ext:
        return best

    run_mmal = _leading_label_run(ext, "mmal")
    mmal_ratio = sum(1 for x in ext if x == "mmal") / len(ext)

    alt_feat = alt.get("starter_features", {})
    cond_common = (
        run_mmal >= 3
        and int(alt_feat.get("branch_carbons", 0)) == 1
        and bool(alt_feat.get("has_hetero", False))
    )

    # Case A: fully mmal-leading mal hypothesis with enough modules.
    cond_a = all(x == "mmal" for x in ext) and len(ext) >= 7

    # Case B: mixed profile where mmal occupancy is near half of modules.
    cond_b = (
        (not all(x == "mmal" for x in ext))
        and len(ext) >= 12
        and 0.45 <= mmal_ratio <= 0.55
    )

    if cond_common and (cond_a or cond_b):
        return alt
    return best


def _apply_mmal_vs_mal_offbyone_override(best, mol, backbone, starter_carbons):
    """
    Correct rare odd-length backbone cases where mmal is selected only because
    mal is one carbon off by stoichiometry, but both chains are fully mal-like.
    """
    if not best or best["starter_unit"] != "mmal":
        return best

    ext = best.get("extenders", [])
    feat = best.get("starter_features", {})
    if not ext or not all(x == "mal" for x in ext):
        return best
    if int(feat.get("branch_carbons", 0)) <= 0:
        return best

    mal_len = starter_carbons.get("mal")
    if mal_len is None:
        return best

    remainder = (len(backbone) + 1) - int(mal_len)
    if remainder < 0 or remainder % 2 != 0:
        return best
    extension_modules = remainder // 2

    mal_extenders = _predict_extenders_with_extension_boundary(
        mol, backbone, extension_modules
    )
    if not mal_extenders or not all(x == "mal" for x in mal_extenders):
        return best

    mal_features = _starter_fragment_features(mol, backbone, extension_modules)
    return {
        "starter_unit": "mal",
        "starter_len": int(mal_len),
        "extension_modules": int(extension_modules),
        "extenders": mal_extenders,
        "score": _score_extender_hypothesis(mal_extenders),
        "starter_penalty": _starter_compatibility_penalty("mal", mal_features),
        "starter_features": mal_features,
    }


def _normalize_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    aliases = {
        "acetyl-coa": "acetyl",
        "chc-coa": "chc",
        "trans-1,2-cpda": "trans",
    }
    return aliases.get(s.lower(), s.lower())


def _starter_backbone_path_len(starter_smiles):
    """
    For starter thioesters, count carbon atoms on the longest carbon path
    starting from the starter carbonyl carbon [CX3](=O)[S].
    """
    mol = Chem.MolFromSmiles(str(starter_smiles))
    if mol is None:
        return None

    patt = Chem.MolFromSmarts("[CX3](=O)[S]")
    matches = mol.GetSubstructMatches(patt)
    if not matches:
        return None
    c1 = matches[0][0]

    carbon_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    carbon_set = set(carbon_idxs)
    if c1 not in carbon_set:
        return None

    adj = {idx: set() for idx in carbon_idxs}
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in carbon_set and b in carbon_set:
            adj[a].add(b)
            adj[b].add(a)

    best = 1
    stack = [(c1, {c1})]
    while stack:
        node, visited = stack.pop()
        if len(visited) > best:
            best = len(visited)
        for nb in adj.get(node, []):
            if nb not in visited:
                stack.append((nb, visited | {nb}))
    return best


def _load_starter_carbon_counts(path, carbon_mode="backbone_path"):
    """
    Load starter shortName -> effective carbon count from starters.smi-style table.

    carbon_mode:
      - backbone_path: use longest carbon path from [CX3](=O)[S] carbonyl.
      - carbon_count: use provided carbonCount column.
    """
    sdf = pd.read_csv(path, sep="\t")
    required = {"shortName", "smiles"}
    if carbon_mode == "carbon_count":
        required.add("carbonCount")
    if not required.issubset(set(sdf.columns)):
        missing = ", ".join(sorted(required - set(sdf.columns)))
        raise ValueError(f"Missing required columns in starters file: {missing}")

    out = {}
    for _, r in sdf.iterrows():
        name = _normalize_label(r.get("shortName"))
        if not name:
            continue

        c = None
        if carbon_mode == "backbone_path":
            c = _starter_backbone_path_len(r.get("smiles"))
            if c is None:
                try:
                    c = int(r.get("carbonCount"))
                except Exception:
                    c = None
        elif carbon_mode == "carbon_count":
            try:
                c = int(r.get("carbonCount"))
            except Exception:
                c = None
        else:
            raise ValueError(
                "Invalid --starter_carbons_mode. Use 'backbone_path' or 'carbon_count'."
            )

        if c is None:
            continue
        out[name] = c
    return out


def _filter_starter_carbons(starter_carbons, starter_whitelist):
    if starter_whitelist == "all":
        return dict(starter_carbons)

    allowed = {
        _normalize_label(x)
        for x in str(starter_whitelist).split(",")
        if _normalize_label(x)
    }
    out = {k: v for k, v in starter_carbons.items() if k in allowed}
    if not out:
        raise ValueError(
            "No valid starters remain after applying whitelist. "
            "Use --starter_whitelist all or provide valid shortNames."
        )
    return out


def _starter_preference_from_whitelist(starter_whitelist, starter_carbons):
    """
    Build deterministic starter preference ranks from whitelist order.
    Lower rank means higher priority in exact-score ties.
    """
    if starter_whitelist == "all":
        ordered = sorted(starter_carbons.keys())
    else:
        ordered = [
            _normalize_label(x)
            for x in str(starter_whitelist).split(",")
            if _normalize_label(x) in starter_carbons
        ]
    return {name: i for i, name in enumerate(ordered)}


def _list_to_str(values):
    if not values:
        return ""
    return ";".join(str(v) for v in values)


def _is_missing_smiles(smiles):
    return pd.isna(smiles) or not smiles or smiles == "nan"


def _make_prediction_failed_row(bgc_id, cluster_name):
    return {
        "bgc_id": bgc_id,
        "cluster_name": cluster_name,
        "module": "prediction_failed",
        "module_num": -1,
        "at_annotation": "unknown",
    }


def _make_starter_row(bgc_id, cluster_name, starter_unit):
    return {
        "bgc_id": bgc_id,
        "cluster_name": cluster_name,
        "module": "starter module 0",
        "module_num": 0,
        "at_annotation": starter_unit,
    }


def _make_extender_row(bgc_id, cluster_name, module_num, extender_unit):
    return {
        "bgc_id": bgc_id,
        "cluster_name": cluster_name,
        "module": f"extender module {module_num}",
        "module_num": module_num,
        "at_annotation": extender_unit,
    }


def _make_candidate_row(bgc_id, cluster_name, rank, hypothesis):
    return {
        "bgc_id": bgc_id,
        "cluster_name": cluster_name,
        "rank": rank,
        "joint_rank": hypothesis.get("joint_rank"),
        "backbone_rank": hypothesis.get("backbone_rank"),
        "backbone_len": hypothesis.get("backbone_len"),
        "backbone_branch_penalty": hypothesis.get("backbone_branch_penalty"),
        "starter_unit": hypothesis["starter_unit"],
        "starter_carbon_count": hypothesis["starter_len"],
        "extension_modules": hypothesis["extension_modules"],
        "unknown_extender_count": hypothesis["score"][0],
        "starter_penalty": hypothesis["starter_penalty"],
        "extender_units": _list_to_str(hypothesis["extenders"]),
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Predict starter/extender AT units from linearized SMILES and write "
            "one final module-level prediction per BGC."
        )
    )
    ap.add_argument(
        "--input_csv",
        default=str(DATA_DIR / "test_set_linearized_networkx.csv"),
        help="Input CSV containing linearized SMILES.",
    )
    ap.add_argument(
        "--smiles_col",
        default="linearized_smiles",
        help="Column containing linearized SMILES.",
    )
    ap.add_argument(
        "--output_csv",
        default=str(DATA_DIR / "test_set_extender_predictions.csv"),
        help="Output CSV path for final module-level predictions.",
    )
    ap.add_argument(
        "--max_visits",
        type=int,
        default=500000,
        help="Max DFS node visits for path enumeration.",
    )
    ap.add_argument(
        "--early_k",
        type=int,
        default=10,
        help="Early-path window for scoring penalties.",
    )
    ap.add_argument(
        "--starters_smi",
        default=str(RETROTIDE_BCS_DATA / "starters.smi"),
        help="Starter library TSV with shortName and carbonCount.",
    )
    ap.add_argument(
        "--candidates_output_csv",
        default="",
        help=(
            "Optional CSV path to save all starter hypotheses (diagnostics). "
            "If empty, only final predictions are written."
        ),
    )
    ap.add_argument(
        "--starter_whitelist",
        default="mal,mmal,prop,acetyl,isobut,2metbut,pyr,cyclopentene,trans,chc",
        help=(
            "Comma-separated starter shortNames to consider, or 'all'. "
            "Default uses a cis-AT-focused whitelist."
        ),
    )
    ap.add_argument(
        "--starter_carbons_mode",
        default="backbone_path",
        choices=["backbone_path", "carbon_count"],
        help=(
            "How to convert starter structures to stoichiometric starter carbon counts. "
            "'backbone_path' is default and more robust for branched starters."
        ),
    )
    ap.add_argument(
        "--backbone_top_k",
        type=int,
        default=8,
        help="Number of top C1-anchored backbone candidates to jointly score.",
    )
    args = ap.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    df = pd.read_csv(input_path)
    starter_carbons = _load_starter_carbon_counts(
        Path(args.starters_smi), carbon_mode=args.starter_carbons_mode
    )
    starter_carbons = _filter_starter_carbons(starter_carbons, args.starter_whitelist)
    starter_preference = _starter_preference_from_whitelist(
        args.starter_whitelist, starter_carbons
    )

    rows = []
    candidate_rows = []
    for _, row in df.iterrows():
        smiles = str(row.get(args.smiles_col, ""))
        bgc_id = row.get("bgc_id")
        cluster_name = row.get("cluster_name")

        if _is_missing_smiles(smiles):
            rows.append(_make_prediction_failed_row(bgc_id, cluster_name))
            continue

        mol, backbones, status = _core_backbone_candidates_from_smiles(
            smiles,
            max_visits=args.max_visits,
            early_k=args.early_k,
            top_k=args.backbone_top_k,
        )
        if status != "ok":
            rows.append(_make_prediction_failed_row(bgc_id, cluster_name))
            continue

        chosen, joint_candidate_rows = _choose_best_backbone_hypothesis(
            mol,
            backbones,
            starter_carbons,
            starter_preference=starter_preference,
        )
        if chosen is None:
            rows.append(_make_prediction_failed_row(bgc_id, cluster_name))
            continue

        backbone = chosen["backbone"]
        best = chosen["best_hypothesis"]
        starter_unit = best["starter_unit"]
        extender_units = best["extenders"]

        rows.append(_make_starter_row(bgc_id, cluster_name, starter_unit))

        for idx, ext in enumerate(extender_units, start=1):
            rows.append(_make_extender_row(bgc_id, cluster_name, idx, ext))

        for rank, hyp in joint_candidate_rows:
            candidate_rows.append(_make_candidate_row(bgc_id, cluster_name, rank, hyp))

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["bgc_id", "module_num"], kind="stable")
    out_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    if args.candidates_output_csv:
        cand_df = pd.DataFrame(candidate_rows)
        cand_path = Path(args.candidates_output_csv)
        cand_df.to_csv(cand_path, index=False)
        print(f"Saved candidate diagnostics to {cand_path}")


if __name__ == "__main__":
    main()
