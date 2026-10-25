"""Deterministic acceptance and module readout for PKS retrosynthesis termini.

The pathway model removes tailoring reactions.  Once a remaining scaffold can
be explained as a modular PKS assembly product, this module hands it to the
existing deterministic backbone/extender/reduction code and exposes a small,
model-free API for the search engine.

Unlike :mod:`deterministic_stage.predict_pipeline`, terminal validation does
not need the separate atom-attribution GAT.  Search candidates are already
single PKS-scaffold hypotheses, so using the complete molecule prevents a
second learned model from becoming part of the stopping rule.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from functools import lru_cache
from itertools import product
import hashlib
import json
from pathlib import Path
import sys
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

try:
    from pks_training_workflow import execution_rules as execution
except ImportError:
    class _FallbackExecutionRules:
        @staticmethod
        def identity(mol_or_smiles, *, stereo: bool = True) -> str:
            mol = (
                Chem.MolFromSmiles(mol_or_smiles)
                if isinstance(mol_or_smiles, str)
                else Chem.Mol(mol_or_smiles)
            )
            if mol is None:
                return ""
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            if not stereo:
                Chem.RemoveStereochemistry(mol)
            return Chem.MolToSmiles(mol, isomericSmiles=stereo)

        @staticmethod
        def separate_inorganic_components(smiles: str):
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                return smiles, ()
            frags = Chem.GetMolFrags(mol, asMols=True)
            organic, spectators = [], []
            for frag in frags:
                has_carbon = any(a.GetAtomicNum() == 6 for a in frag.GetAtoms())
                smi = Chem.MolToSmiles(frag, isomericSmiles=True)
                if has_carbon:
                    organic.append(smi)
                else:
                    spectators.append(smi)
            return ".".join(organic), tuple(spectators)

        @staticmethod
        def stereo_assignment_difference(candidate_smiles: str, reference_smiles: str):
            cand = Chem.MolFromSmiles(candidate_smiles) if candidate_smiles else None
            ref = Chem.MolFromSmiles(reference_smiles) if reference_smiles else None
            if cand is None or ref is None:
                return None
            cand_iso = _FallbackExecutionRules.identity(cand, stereo=True)
            ref_iso = _FallbackExecutionRules.identity(ref, stereo=True)
            cand_const = _FallbackExecutionRules.identity(cand, stereo=False)
            ref_const = _FallbackExecutionRules.identity(ref, stereo=False)
            if cand_const != ref_const:
                return None
            if cand_iso == ref_iso:
                return {"relation": "exact", "candidate_smiles": cand_iso, "reference_smiles": ref_iso}
            # Check if candidate only adds stereo to unassigned sites in reference while preserving all assigned sites
            match = ref.GetSubstructMatch(cand, useChirality=True)
            ref_chiral = sum(
                1 for info in Chem.FindPotentialStereo(ref)
                if info.specified == Chem.StereoSpecified.Specified
            )
            cand_chiral = sum(
                1 for info in Chem.FindPotentialStereo(cand)
                if info.specified == Chem.StereoSpecified.Specified
            )
            if match and cand_chiral >= ref_chiral:
                return {
                    "relation": "adds_unassigned_stereo",
                    "candidate_smiles": cand_iso,
                    "reference_smiles": ref_iso,
                }
            return {
                "relation": "stereo_differs",
                "candidate_smiles": cand_iso,
                "reference_smiles": ref_iso,
            }

    execution = _FallbackExecutionRules()

STAGE_DIR = Path(__file__).resolve().parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

try:
    from . import carbon_backbone as cb
    from . import predict_extenders as pe
    from . import predict_reduction as pr
    from . import ringbreaker
except ImportError:
    import carbon_backbone as cb
    import predict_extenders as pe
    import predict_reduction as pr
    import ringbreaker


# Effective C1-anchored carbon-path lengths produced by
# predict_extenders._load_starter_carbon_counts(..., "backbone_path") for the
# cis-AT starter whitelist used by predict_pipeline.py.  Keeping these small
# deterministic values here makes terminal testing self-contained; the full
# pipeline can still load the corresponding RetroTide structures when it needs
# to simulate chemistry.
DEFAULT_STARTER_CARBONS = {
    "mmal": 3,
    "mal": 2,
    "acetyl": 2,
    "prop": 3,
    "isobut": 3,
    "2metbut": 4,
    "chc": 7,
    "trans": 7,
    "cyclopentene": 6,
    "pyr": 5,
}
DEFAULT_STARTER_ORDER = (
    "mal",
    "mmal",
    "prop",
    "acetyl",
    "isobut",
    "2metbut",
    "pyr",
    "cyclopentene",
    "trans",
    "chc",
)
DEFAULT_STARTER_PREFERENCE = {
    name: index for index, name in enumerate(DEFAULT_STARTER_ORDER)
}


@lru_cache(maxsize=1)
def _assembly_resources():
    """Use the checked-in substrate structures, without importing BCS or torch."""
    chemistry = pr
    root = Path(__file__).resolve().parents[1]
    for candidate_dir in (
        STAGE_DIR / "data",
        root / "data",
        root / "RetroTideV2" / "bcs" / "data",
        root / "RetroTide" / "bcs" / "data",
        root / "BioPKS-Pipeline" / "biopks_pipeline" / "retrotide" / "data",
        root / "practice" / "data",
    ):
        if (candidate_dir / "starters.smi").exists():
            data = candidate_dir
            break
    else:
        raise FileNotFoundError("Could not locate starters.smi / extenders.smi")

    starters = chemistry._load_starter_smiles(data / "starters.smi")
    return (chemistry, starters, chemistry._load_extender_fragments(data / "extenders.smi"),
            chemistry._reaction_templates())


@lru_cache(maxsize=1)
def assembly_starter_carbons():
    """All executable checked-in starters, for structure-only assembly search.

    The frozen modular benchmark retains its original whitelist. Deployment
    also needs aromatic/NRPS-like loading substrates already in starters.smi.
    Whole-molecule forward reconstruction remains mandatory for every proposal.
    """
    _, starters, _, _ = _assembly_resources()
    return {name: length for name, smiles in starters.items()
            if (length := pe._starter_backbone_path_len(smiles)) is not None}


@lru_cache(maxsize=2)
def _supported_heteroatom_classes(extended_starters=False):
    """Necessary atom classes from the actual assembly substrate whitelist."""
    _, starters, fragments, _ = _assembly_resources()
    names = assembly_starter_carbons() if extended_starters else DEFAULT_STARTER_CARBONS
    substrates = [Chem.MolFromSmiles(starters[k]) for k in names if k in starters]
    substrates.extend(fragments.values())
    return {(a.GetAtomicNum(), a.GetIsAromatic()) for mol in substrates for a in mol.GetAtoms()
            if a.GetAtomicNum() not in (1, 6, 8, 16)}


def unsupported_atom_classes(mol, *, extended_starters=False):
    """Heteroatom classes in ``mol`` that no supported assembly substrate carries.

    KR/DH/ER and supported release operations do not create new nitrogen,
    halogen, etc. atom classes. Carbon, oxygen and carrier sulfur still require
    full reconstruction. Returns ``["N", "n", "Cl", ...]`` (lower case = aromatic).
    """
    supported = _supported_heteroatom_classes(extended_starters)
    return sorted({a.GetSymbol().lower() if a.GetIsAromatic() else a.GetSymbol() for a in mol.GetAtoms()
                   if a.GetAtomicNum() not in (1, 6, 8, 16)
                   and (a.GetAtomicNum(), a.GetIsAromatic()) not in supported})


def can_match_assembly_elements(smiles):
    """Cheap necessary check, never a terminal certificate.

    For example, an aliphatic amino sugar cannot originate from the supported
    aromatic nitrogen-containing substrates (see :func:`unsupported_atom_classes`).
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and not unsupported_atom_classes(mol)


def normalize_protonation(smiles):
    """Neutralize acid/base conjugates; returns ``(smiles, operations)``.

    A carboxylate, ammonium or phenolate is a protonation state of the same covalent structure, so
    the assembler reads the neutral form (RDKit ``Uncharger``). Charge-separated groups that are not
    acid/base conjugates (nitro, N-oxide, quaternary ammonium) are left unchanged. Only hydrogen
    counts and formal charges change; connectivity and stereo are untouched. The caller's string is
    returned as-is when nothing changes. ``operations`` lists ``[element, charge_before, charge_after]``
    per atom changed, in input atom order; it records input handling, not a chemical claim.
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or not any(a.GetFormalCharge() for a in mol.GetAtoms()):
        return smiles, ()
    neutral = rdMolStandardize.Uncharger(canonicalOrder=True).uncharge(Chem.Mol(mol))
    operations = tuple([a.GetSymbol(), a.GetFormalCharge(), b.GetFormalCharge()]
                       for a, b in zip(mol.GetAtoms(), neutral.GetAtoms())
                       if a.GetFormalCharge() != b.GetFormalCharge())
    return (Chem.MolToSmiles(neutral), operations) if operations else (smiles, ())


def _comparison_smiles(mol, *, stereo=True):
    """Compare free acid and terminal carrier placeholders on the same basis.

    Only terminal acyl sulfur is normalized. Other sulfur and all molecular
    components remain part of the identity check.
    """
    copy = Chem.Mol(mol)
    for match in copy.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[S;D1]")):
        atom = copy.GetAtomWithIdx(match[2])
        atom.SetAtomicNum(8)
        atom.SetNumRadicalElectrons(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.SanitizeMol(copy)
    if any(a.GetFormalCharge() for a in copy.GetAtoms()):
        copy = rdMolStandardize.Uncharger(canonicalOrder=True).uncharge(copy)
    if not stereo:
        Chem.RemoveStereochemistry(copy)
    return Chem.MolToSmiles(copy, isomericSmiles=stereo)


@lru_cache(maxsize=128)
def _starter_geometry_candidates(smiles):
    """Finite E/Z hypotheses for unspecified C=C bonds of a fixed substrate.

    This uses only the dictionary substrate, never atoms or geometry copied
    from the query. Keep the original unspecified form and all assigned stereo;
    tetrahedral centres and non-carbon double bonds are not enumerated.
    """
    mol = Chem.MolFromSmiles(smiles)
    sites = []
    for info in Chem.FindPotentialStereo(mol):
        if (info.type != Chem.StereoType.Bond_Double or
                info.specified != Chem.StereoSpecified.Unspecified):
            continue
        bond = mol.GetBondWithIdx(info.centeredOn)
        if bond.GetBeginAtom().GetAtomicNum() == bond.GetEndAtom().GetAtomicNum() == 6:
            sites.append(info.centeredOn)
    original = Chem.MolToSmiles(mol, isomericSmiles=True)
    candidates = {original}
    for choices in product((Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ), repeat=len(sites)):
        variant = Chem.Mol(mol)
        for index, stereo in zip(sites, choices):
            bond = variant.GetBondWithIdx(index)
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            left = next(a.GetIdx() for a in bond.GetBeginAtom().GetNeighbors() if a.GetIdx() != end)
            right = next(a.GetIdx() for a in bond.GetEndAtom().GetNeighbors() if a.GetIdx() != begin)
            bond.SetStereoAtoms(left, right)
            bond.SetStereo(stereo)
        Chem.SetDoubleBondNeighborDirections(variant)
        candidate = Chem.MolToSmiles(variant, isomericSmiles=True)
        difference = execution.stereo_assignment_difference(candidate, original)
        if difference and difference["relation"] in ("exact", "adds_unassigned_stereo"):
            candidates.add(candidate)
    return (original, *sorted(candidates - {original}))


def _validate_assembly_candidates(mol, backbone, best):
    """Reject incompatible constitutions once, then verify each starter geometry."""
    original = _validate_assembly(mol, backbone, best)
    yield original
    if not original.get("constitution_match"):
        return
    dictionary_smiles = _assembly_resources()[1][best["starter_unit"]]
    geometries = _starter_geometry_candidates(dictionary_smiles)
    for smiles in geometries[1:]:
        validation = _validate_assembly(mol, backbone, best, starter_smiles=smiles)
        validation["starter_geometry"] = {
            "dictionary_smiles": dictionary_smiles,
            "selected_smiles": smiles,
            "enumerated_assigned_candidates": len(geometries) - 1,
            "scope": "input-compatible assembly hypothesis; native starter use not established",
        }
        yield validation


def _validate_assembly(mol, backbone, best, *, starter_smiles=None):
    """Rebuild the WHOLE linearized molecule from actual substrates and modules.

    A carbon-path annotation alone is never a terminal certificate. An exact
    match is evidence under this assembly grammar, not enzyme activity evidence.
    """
    chemistry, starters, fragments, templates = _assembly_resources()
    if best["starter_unit"] not in starters:
        return {"status": "unsupported_starter"}
    if starter_smiles is not None:
        starters = {**starters, best["starter_unit"]: starter_smiles}
    if any(chemistry._normalize_extender_key(x) not in fragments for x in best["extenders"]):
        return {"status": "unsupported_extender"}
    carbon_count = lambda m: sum(a.GetAtomicNum() == 6 for a in m.GetAtoms())
    expected_carbons = carbon_count(Chem.MolFromSmiles(starters[best["starter_unit"]])) + sum(
        carbon_count(fragments[chemistry._normalize_extender_key(x)]) - 1 for x in best["extenders"]
    )
    observed_carbons = carbon_count(mol)
    if expected_carbons != observed_carbons:
        return {"status": "carbon_count_mismatch", "expected_carbons": expected_carbons,
                "observed_carbons": observed_carbons,
                "unexplained_carbons": observed_carbons - expected_carbons}
    n = best["extension_modules"]
    program = []
    for num, extender in enumerate(best["extenders"], 1):
        alpha_pos = 2 * (n - num + 1)
        beta_pos = alpha_pos + 1
        if beta_pos >= len(backbone):
            return {"status": "incomplete_backbone"}
        pred = chemistry.infer_module_reduction(
            mol, backbone[alpha_pos - 1], backbone[beta_pos - 1], extender,
            backbone=backbone, alpha_pos=alpha_pos, beta_pos=beta_pos,
        )
        program.append({"module_num": num, "at_annotation": extender, **pred})
    _, intermediates = chemistry._simulate_bgc_intermediates(
        best["starter_unit"], program, starters, fragments, templates,
    )
    if len(intermediates) != n or any(not s for s in intermediates.values()):
        return {"status": "assembly_failed"}
    generated = Chem.MolFromSmiles(intermediates[n])
    if generated is None:
        return {"status": "assembly_failed"}
    observed = _comparison_smiles(mol)
    rebuilt = _comparison_smiles(generated)
    constitution_match = _comparison_smiles(mol, stereo=False) == _comparison_smiles(generated, stereo=False)
    return {
        "status": "ok" if observed == rebuilt else (
            "stereo_mismatch" if constitution_match else "assembly_mismatch"),
        "constitution_match": constitution_match,
        "stereo_match": observed == rebuilt,
        "reconstructed_smiles": rebuilt,
        "comparison_smiles": observed,
        "forward_program": {"starter": best["starter_unit"], "modules": program,
                            **({"starter_smiles": starter_smiles} if starter_smiles is not None else {})},
        "evidence": "forward_assembly_under_supported_grammar",
    }


FAILURE_LAYERS = ("invalid_input", "unsupported_function", "candidate_limit", "structure_mismatch",
                  "internal_error")
# Release classes enumerate_backbone_precursors has no hypothesis for, counted on the input.
_UNRELEASED_GROUPS = {"acyclic_ester": "[CX3](=O)!@[OX2H0]", "acyclic_amide": "[CX3](=O)!@[NX3]",
                      "ring_acyl_heteroatom": "[CX3;R](=O)@[OX2H0,NX3;R]", "ketone": "[#6][CX3](=O)[#6]"}


def failure_cause(layer, reason, **evidence):
    return {"layer": layer, "reason": reason, "evidence": evidence}


def failure_report(causes, **limits):
    """One failure record: the first cause names ``layer``/``reason``; ``causes`` lists all, in order."""
    if not causes or any(c["layer"] not in FAILURE_LAYERS for c in causes):
        raise ValueError(f"failure layers must be in {FAILURE_LAYERS}: {causes}")
    return {"layer": causes[0]["layer"], "reason": causes[0]["reason"], "causes": causes, "limits": limits}


def _structure_difference(observed_smiles, rebuilt_smiles):
    """What the input has beyond the rebuilt structure (observed minus rebuilt), as counts."""
    observed, rebuilt = (Chem.MolFromSmiles(s) for s in (observed_smiles, rebuilt_smiles))
    count = lambda m: Counter(a.GetSymbol() for a in Chem.AddHs(m).GetAtoms())
    a, b = count(observed), count(rebuilt)
    return {"elements": {e: a[e] - b[e] for e in sorted(set(a) | set(b)) if a[e] != b[e]},
            "rings": rdMolDescriptors.CalcNumRings(observed) - rdMolDescriptors.CalcNumRings(rebuilt),
            "aromatic_rings": (rdMolDescriptors.CalcNumAromaticRings(observed) -
                               rdMolDescriptors.CalcNumAromaticRings(rebuilt))}


def _difference_rank(difference):
    """Closeness of a rebuilt structure: atoms it builds beyond the input first, then the input's surplus."""
    heavy = {e: n for e, n in difference["elements"].items() if e != "H"}
    return (sum(-n for n in heavy.values() if n < 0) + max(-difference["rings"], 0),
            sum(n for n in heavy.values() if n > 0), abs(difference["rings"]),
            abs(difference["elements"].get("H", 0)))


def _mismatch_reason(difference):
    """Reason for a constitution that differs from the rebuilt one (carbon counts agree).

    Signs are input minus rebuilt: ``extra_*_in_input`` names atoms or rings the input has beyond the
    rebuilt structure; ``hypothesis_exceeds_input`` means the rebuilt structure has atoms or rings
    the input lacks. These are graph differences, not reaction centres.
    """
    elements, rings = difference["elements"], difference["rings"]
    if any(n < 0 for e, n in elements.items() if e != "H") or rings < 0:
        return "hypothesis_exceeds_input"
    if any(n > 0 for e, n in elements.items() if e not in ("C", "H", "O")):
        return "extra_heteroatoms_in_input"
    if rings > 0:
        return "extra_rings_in_input"
    if elements.get("O", 0) > 0:
        return "extra_oxygen_in_input"
    if "H" in elements:
        return "unsaturation_differs"
    return "connectivity_differs"


def _candidate_cause(metadata, candidates):
    """Cause from the candidates that reached validation, judged on the best-ranked status."""
    validation = metadata["assembly_validation"]
    status = validation["status"]
    same = [c["assembly_validation"] for c in candidates if c["assembly_validation"]["status"] == status]
    if status == "unsupported_extender":
        chemistry, _, fragments, _ = _assembly_resources()
        labels = sorted({x for c in candidates if c["assembly_validation"]["status"] == status
                         for x in c["extenders"] if chemistry._normalize_extender_key(x) not in fragments})
        return failure_cause("unsupported_function", "extender_not_in_dictionary", labels=labels,
                             candidates=len(same))
    if status == "carbon_count_mismatch":
        # unexplained_carbons = input carbons minus the carbons the hypothesis builds.
        values = [v["unexplained_carbons"] for v in same]
        closest = min(values, key=lambda n: (abs(n), -n))
        return failure_cause("structure_mismatch",
                             "extra_carbons_in_input" if closest > 0 else "missing_carbons_in_input",
                             unexplained_carbons=closest, observed_carbons=same[0]["observed_carbons"],
                             candidates=len(same), unexplained_range=[min(values), max(values)])
    if status == "assembly_mismatch":
        differences = [(_structure_difference(v["comparison_smiles"], v["reconstructed_smiles"]), v) for v in same]
        difference, closest = min(differences, key=lambda d: _difference_rank(d[0]))
        return failure_cause("structure_mismatch", _mismatch_reason(difference), difference=difference,
                             observed_smiles=closest["comparison_smiles"],
                             rebuilt_smiles=closest["reconstructed_smiles"], candidates=len(same))
    if status == "stereo_mismatch":
        return failure_cause("structure_mismatch", "stereo_assignment_differs",
                             relation=(validation.get("endpoint_stereo") or {}).get("relation"),
                             candidates=len(same))
    if status == "release_mismatch":
        return failure_cause("structure_mismatch", "release_roundtrip_differs", candidates=len(same))
    if status == "incomplete_backbone":
        return failure_cause("structure_mismatch", "backbone_shorter_than_modules", candidates=len(same))
    if status == "unsupported_starter":
        return failure_cause("unsupported_function", "starter_not_in_dictionary", candidates=len(same))
    if status == "assembly_failed":
        return failure_cause("unsupported_function", "forward_assembly_failed", candidates=len(same))
    return failure_cause("internal_error", f"unmapped_validation_status:{status}")


def _explain_failure(metadata, mol, *, max_visits, backbone_top_k, extended_starters):
    """Name the layer that stopped a non-``ok`` decomposition; ``mol`` is the (normalized) input."""
    status = metadata["status"]
    if status == "disconnected_input":
        # The search separates inorganic spectators before this module sees the structure.
        _, spectators = execution.separate_inorganic_components(Chem.MolToSmiles(mol))
        return failure_report([failure_cause(
            "invalid_input", "inorganic_spectator_present" if spectators else "multiple_organic_components",
            components=len(Chem.GetMolFrags(mol)), spectators=spectators)])
    if status in ("missing_smiles", "invalid_smiles"):
        return failure_report([failure_cause("invalid_input", status)])
    causes = []
    if status == "missing_supported_release":
        causes.append(failure_cause("unsupported_function", "no_supported_release", groups={
            name: len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
            for name, smarts in _UNRELEASED_GROUPS.items()}))
    classes = unsupported_atom_classes(mol, extended_starters=extended_starters)
    if classes:
        causes.append(failure_cause("unsupported_function", "atom_class_outside_substrates", classes=classes))
    if status == "no_valid_hypothesis":
        causes.append(failure_cause("structure_mismatch", "no_backbone_hypothesis",
                                    releases=metadata["release_candidates"]))
    elif metadata["candidates"]:
        causes.append(_candidate_cause(metadata, metadata["candidates"]))
    if not causes:
        causes.append(failure_cause("internal_error", f"unexplained_status:{status}"))
    return failure_report(
        causes, backbone_top_k=backbone_top_k, max_visits=max_visits,
        backbone_candidate_limit_reached=metadata["backbone_candidate_limit_reached"],
        search_budget_exhausted=metadata["search_budget_exhausted"],
        omitted_candidates=metadata.get("omitted_candidate_count", 0))


def _failure(metadata: dict, status: str, mol=None, **options):
    result = dict(metadata)
    result["status"] = status
    result["failure"] = _explain_failure(result, mol, **options)
    return (), result


def _module_rows(mol, backbone, best):
    extension_modules = int(best["extension_modules"])
    rows = [
        {
            "module": "starter module 0",
            "module_num": 0,
            "at_annotation": best["starter_unit"],
            "alpha_carbon": None,
            "alpha_atom_index": None,
            "beta_carbon": None,
            "beta_atom_index": None,
            "beta_state": "",
            "kr_subtype": "",
            "kr_activity": "",
            "dh_activity": "",
            "er_activity": "",
            "domains": pr._build_domain_string(
                best["starter_unit"], "", "", is_starter=True, is_terminal=False
            ),
        }
    ]
    for module_num in range(1, extension_modules + 1):
        # Backbones are C1-outward.  The last extension is closest to C1, while
        # module 1 occupies the highest numbered alpha/beta pair.
        alpha_pos = 2 * (extension_modules - module_num + 1)
        beta_pos = alpha_pos + 1
        if beta_pos > len(backbone):
            raise ValueError("incomplete backbone")
        alpha_idx = int(backbone[alpha_pos - 1])
        beta_idx = int(backbone[beta_pos - 1])
        at_annotation = best["extenders"][module_num - 1]
        beta_state = pr._classify_beta_state(mol, alpha_idx, beta_idx)
        kr_subtype = pr._kr_subtype(
            mol,
            alpha_idx,
            beta_idx,
            beta_state,
            backbone,
            alpha_pos,
            beta_pos,
        )
        kr_activity, dh_activity, er_activity = {
            "β-keto": ("inactive", "inactive", "inactive"),
            "β keto": ("inactive", "inactive", "inactive"),
            "β-hydroxy": ("active", "inactive", "inactive"),
            "β hydroxyl": ("active", "inactive", "inactive"),
            "enoyl": ("active", "active", "inactive"),
            "saturated": ("active", "active", "active"),
        }.get(beta_state, ("inactive", "inactive", "inactive"))
        rows.append(
            {
                "module": f"extender module {module_num}",
                "module_num": module_num,
                "at_annotation": at_annotation,
                "alpha_carbon": f"C{alpha_pos}",
                "alpha_atom_index": alpha_idx,
                "beta_carbon": f"C{beta_pos}",
                "beta_atom_index": beta_idx,
                "beta_state": beta_state,
                "kr_subtype": kr_subtype,
                "kr_activity": kr_activity,
                "dh_activity": dh_activity,
                "er_activity": er_activity,
                "domains": pr._build_domain_string(
                    at_annotation,
                    beta_state,
                    kr_subtype,
                    is_starter=False,
                    is_terminal=module_num == extension_modules,
                ),
            }
        )

    return tuple(rows)


def _candidate_order(candidate):
    validation = candidate["assembly_validation"]
    status_order = {"ok": 0, "stereo_mismatch": 1, "assembly_mismatch": 2,
                    "carbon_count_mismatch": 3}
    return (status_order.get(validation["status"], 4),
            abs(validation.get("unexplained_carbons", 0)),
            -candidate["backbone_len"],
            DEFAULT_STARTER_PREFERENCE.get(candidate["starter_unit"], 999),
            candidate["release"]["mechanism"],
            tuple(candidate["release"]["cut_source_atom_indices"] or []),
            tuple(candidate["backbone_atom_indices"]))


@lru_cache(maxsize=512)
def _decompose_pks_terminal_cached(
    smiles: str,
    *,
    max_visits: int = 500_000,
    early_k: int = 10,
    backbone_top_k: int = 8,
    extended_starters: bool = False,
) -> tuple[tuple[dict, ...], dict]:
    """Validate release/backbone/starter alternatives before choosing a terminus."""
    if max_visits < 1 or backbone_top_k < 1:
        raise ValueError("max_visits and backbone_top_k must be positive")
    metadata = {"status": "no_valid_hypothesis", "input_smiles": smiles,
                "linearized_smiles": None, "terminus_type": None,
                "backbone_len": None, "extension_modules": 0,
                "search_budget_exhausted": False, "backbone_candidate_limit_reached": False,
                "candidates": []}
    options = {"max_visits": max_visits, "backbone_top_k": backbone_top_k, "extended_starters": extended_starters}
    if not smiles or str(smiles).strip().lower() == "nan":
        return _failure(metadata, "missing_smiles", **options)
    mol = source = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _failure(metadata, "invalid_smiles", **options)
    if len(Chem.GetMolFrags(mol)) != 1:
        return _failure(metadata, "disconnected_input", mol, **options)
    releases = ringbreaker.enumerate_backbone_precursors(smiles)
    metadata["release_candidates"] = len(releases)
    if not releases:
        return _failure(metadata, "missing_supported_release", source, **options)
    candidates = []
    accepted = {}
    for release_index, release in enumerate(releases):
        mol = release["mol"]
        release_valid = ringbreaker.verify_release_roundtrip(release)
        backbones, exhausted = cb.find_pks_backbone_candidates(
            mol, release["c1_idx"], max_visits=max_visits,
            early_k=early_k, top_k=backbone_top_k, diverse=True,
        )
        metadata["search_budget_exhausted"] |= bool(exhausted)
        metadata["search_budget_exhausted"] |= release["normalization_limit_reached"]
        metadata["backbone_candidate_limit_reached"] |= len(backbones) >= backbone_top_k
        for backbone in backbones:
            hypotheses = pe._enumerate_starter_hypotheses(
                len(backbone), assembly_starter_carbons() if extended_starters else DEFAULT_STARTER_CARBONS)
            hypotheses = [h for h in hypotheses if h["extension_modules"] > 0]
            evaluated = pe._evaluate_starter_hypotheses(
                mol, backbone, hypotheses, starter_preference=DEFAULT_STARTER_PREFERENCE,
            )
            for best in evaluated:
                for validation in _validate_assembly_candidates(mol, backbone, best):
                    validation["release_roundtrip"] = release_valid
                    if validation.get("constitution_match"):
                        endpoint = ringbreaker.release_forward_product(release, validation["reconstructed_smiles"])
                        validation["reconstructed_endpoint_smiles"] = endpoint
                        validation["endpoint_stereo"] = (execution.stereo_assignment_difference(
                            endpoint, release["source_smiles"]) if endpoint else None)
                        if not validation["endpoint_stereo"]:
                            validation["status"] = "release_mismatch"
                            validation["release_roundtrip"] = False
                        else:
                            validation["stereo_match"] = validation["endpoint_stereo"]["relation"] in (
                                "exact",
                                "adds_unassigned_stereo",
                            )
                            validation["status"] = "ok" if validation["stereo_match"] else "stereo_mismatch"
                    if validation["status"] == "ok" and not release_valid:
                        validation["status"] = "release_mismatch"
                    candidate = {
                        "candidate_id": len(candidates), "release_index": release_index,
                        "linearized_smiles": release["smiles"],
                        "terminus_type": release["mechanism"],
                        "backbone_len": len(backbone), "backbone_atom_indices": backbone,
                        "backbone_source_atom_indices": [release["source_atom_indices"][i] for i in backbone],
                        "extension_modules": best["extension_modules"],
                        "starter_unit": best["starter_unit"], "extenders": best["extenders"],
                        "release": {k: release[k] for k in ("mechanism", "cut_source_atom_indices",
                                   "c1_source_atom_index", "source_smiles", "normalizations", "ring_alkene_constraints")},
                        "assembly_validation": validation,
                    }
                    # The readout depends on the structure only, so constitution-matching candidates
                    # keep it even when the assigned stereochemistry disagrees (not a terminal).
                    if validation.get("constitution_match"):
                        candidate["module_rows"] = _module_rows(mol, backbone, best)
                        if release["mechanism"] == "reductive_aldehyde":
                            candidate["module_rows"][-1]["domains"] = candidate["module_rows"][-1]["domains"].replace(
                                " - TE", " - R")
                    candidates.append(candidate)
                    if validation["status"] == "ok":
                        accepted[candidate["candidate_id"]] = candidate["module_rows"]
    candidates.sort(key=_candidate_order)
    metadata["candidate_count"] = len(candidates)
    metadata["accepted_candidate_count"] = len(accepted)
    # Keep diagnostics bounded while showing which hypotheses were examined.
    metadata["candidates"] = candidates[:64]
    metadata["omitted_candidate_count"] = max(0, len(candidates) - 64)
    if not candidates:
        return _failure(metadata, "no_valid_hypothesis", source, **options)
    chosen = candidates[0]
    metadata.update({k: v for k, v in chosen.items() if k not in ("candidate_id", "module_rows")})
    metadata["status"] = chosen["assembly_validation"]["status"]
    metadata["validation_scope"] = "supported_modular_type_i_assembly; biochemical route not established"
    metadata["failure"] = (None if metadata["status"] == "ok"
                           else _explain_failure(metadata, source, **options))
    return accepted.get(chosen["candidate_id"], ()), metadata


def decompose_pks_terminal(
    smiles: str,
    *,
    max_visits: int = 500_000,
    early_k: int = 10,
    backbone_top_k: int = 8,
    extended_starters: bool = False,
) -> tuple[tuple[dict, ...], dict]:
    """Safely decompose a possible terminus; ordinary failures never abort search.

    The input is read in its neutral protonation state (:func:`normalize_protonation`). ``input_smiles``
    keeps the caller's string and ``input_normalizations`` lists what changed; every other SMILES in
    the result refers to the neutral form. A non-``ok`` status carries ``failure`` (see
    :func:`failure_report`).
    """
    try:
        normalized, operations = normalize_protonation(smiles)
        rows, metadata = deepcopy(_decompose_pks_terminal_cached(
            normalized,
            max_visits=max_visits,
            early_k=early_k,
            backbone_top_k=backbone_top_k,
            extended_starters=extended_starters,
        ))
        metadata.update(input_smiles=smiles, input_normalizations=list(operations))
        return rows, metadata
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return (), {
            "status": "decomposition_error",
            "input_smiles": smiles,
            "error": error,
            "failure": failure_report([failure_cause("internal_error", "decomposition_error", error=error)]),
        }


def is_deterministic_pks_terminal(smiles: str) -> bool:
    """Whether ``smiles`` has a complete deterministic PKS decomposition."""
    rows, metadata = decompose_pks_terminal(smiles)
    return bool(rows) and metadata["status"] == "ok"


# ---------------------------------------------------------------------------
# Structure-only assembly proposal
#
# Input: one structure. Output: forward programs whose execution by the existing
# executors (modular rebuild above, eukaryotic_pks.execute_variants,
# generate.execute_program) gives the same constitution. Nothing about the
# structure's origin (core ID, generation history, family label) is an input.
# A reproducing program is a forward-reproduction hypothesis under the declared
# space; it is not evidence that the enzymes accept the substrates.
# ---------------------------------------------------------------------------

ASSEMBLY_STATUSES = ("reproduced", "exhausted_none", "budget_exhausted", "unsupported")
DEFAULT_PROGRAM_BUDGET = 20_000  # forward executions per enumerated family
DEFAULT_BACKBONE_TOP_K = 8  # backbones per release hypothesis; equals decompose_pks_terminal's default
ENUMERATED_FAMILIES = ("fungal_nr", "fungal_pr", "plant_type_iii", "type_ii", "type_iii")
UNSEARCHED_FAMILIES = {
    "fungal_hr": "HR cycle states are not read from the structure yet; exhaustive cycle-state "
                 "enumeration exceeds 5e7 programs on each of 11 held-out HR cores"}
# Bounds of the generator run cores/family_100_20260929/pks_run.json: NR/PR/plant 2..6 extensions,
# at most two C-methylations, 'screen' starter pools. Type II/III follow generate.screening_programs.
EXTENSION_RANGE = (2, 6)
MAX_METHYLATIONS = 2
CARBON_CHANGE = {"decarboxylate": -1, "deacetylate": -2}
BETA_STATE_REDUCTION = {"β-keto": "keto", "β-hydroxy": "hydroxy", "enoyl": "enoyl", "saturated": "alkyl"}
RING_CLOSING_STEPS = ("aldol_aromatize", "hemiketal", "claisen_aromatize", "pyrone")
# Where an alpha-methyl comes from: a methylmalonyl extender (modular AT) or SAM through the cMT domain (fungal iPKS).
EXTENDER_METHYL, CMT_METHYL = "methylmalonyl_extender", "cMT_SAM"
SCOPE = ("Forward-reproduction hypotheses inside the declared space; enzyme acceptance, "
         "biological feasibility and the historical program are not established")


@lru_cache(maxsize=1)
def _executors():
    from practice import eukaryotic_pks, generate
    # generate.screening_programs imports eukaryotic_pks by its script-level name.
    sys.modules.setdefault("eukaryotic_pks", eukaryotic_pks)
    return eukaryotic_pks, generate


def _carbons(mol):
    return sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms())


def _extra_classes(mol, *, starter=False):
    """(element, aromatic) classes beyond C, O and H; sulfur is only the carrier placeholder of a starter."""
    skip = {1, 6, 8, 16} if starter else {1, 6, 8}
    return {(a.GetAtomicNum(), a.GetIsAromatic()) for a in mol.GetAtoms() if a.GetAtomicNum() not in skip}


def _starter_classes(smiles_list):
    return set().union(*(_extra_classes(Chem.MolFromSmiles(s), starter=True) for s in smiles_list))


@lru_cache(maxsize=1)
def _enumerated_space():
    """Eukaryotic recipe frames, Type II/III programs and starter carbon counts, built once."""
    euk, gen = _executors()
    space = euk.Space(kinds=euk.KINDS, modes=("NR", "PR"), min_extensions=EXTENSION_RANGE[0],
                      max_extensions=EXTENSION_RANGE[1],
                      family_extensions=(("NR", *EXTENSION_RANGE), ("PR", *EXTENSION_RANGE)),
                      max_methylations=MAX_METHYLATIONS, starter_preset="screen", selection="exhaustive")
    pools = euk.normalize_pools(space)
    starters = {kind: {} for kind in pools}
    for kind, pool in pools.items():
        for name, smiles in pool:
            starters[kind].setdefault(_carbons(Chem.MolFromSmiles(smiles)), []).append((name, smiles))
    frames = []
    for kind, n, recipe, options, _ in euk.recipe_batches(space, pools):
        family = "plant_type_iii" if kind == "plant_type_iii" else "fungal_" + recipe.mode.lower()
        change = 2 * n + sum(CARBON_CHANGE.get(f.kind, 0) for f in (*recipe.folds, *recipe.finishing))
        for cycles in euk.schedules(options, space.max_methylations):
            frames.append((family, kind, recipe, cycles, change + sum(c.methyl for c in cycles)))
    special = tuple(euk.special_plant_programs(space))
    iterative, iterative_smiles = {}, {"type_ii": [], "type_iii": []}
    for program in gen.screening_programs(gen.builtin_programs(), {}, {}):
        carbons = (_carbons(Chem.MolFromSmiles(program.starter_smiles)) + 2 * program.extensions
                   + sum(CARBON_CHANGE.get(step.operation, 0) for step in program.steps))
        iterative.setdefault(program.pks_class, {}).setdefault(carbons, []).append(program)
        iterative_smiles[program.pks_class].append(program.starter_smiles)
    fungal = [smiles for pool in starters["fungal_ipks"].values() for _, smiles in pool]
    plant = [smiles for pool in starters["plant_type_iii"].values() for _, smiles in pool]
    # Special programs rewrite their anthranilate nitrogen into the ring, so their classes are read from the executed products.
    special_products = [Chem.MolFromSmiles(_forward_smiles("plant_type_iii", p)) for p in special]
    classes = {"fungal_nr": _starter_classes(fungal), "fungal_pr": _starter_classes(fungal),
               "plant_type_iii": _starter_classes(plant).union(*(_extra_classes(m) for m in special_products)),
               "type_ii": _starter_classes(iterative_smiles["type_ii"]),
               "type_iii": _starter_classes(iterative_smiles["type_iii"])}
    return {"frames": tuple(frames), "starters": starters, "special": special, "iterative": iterative,
            "classes": classes, "pool_sizes": {kind: sum(map(len, g.values())) for kind, g in starters.items()}}


def _canonical_starter(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _fold_record(fold):
    return {"kind": fold.kind, "positions": list(fold.positions), "eliminate_alcohols": list(fold.eliminate_alcohols)}


def summarize_forward_program(family, program):
    """The assembly-defining fields of one enumerated program, in the same form for every family."""
    if hasattr(program, "cycles"):  # eukaryotic_pks.Program
        # Every eukaryotic cycle extends with malonyl-CoA; an alpha-methyl comes from the cMT domain (SAM),
        # not from a methylmalonyl extender, so the two are different programs.
        return {"family": family, "executor": "eukaryotic_pks.execute_variants",
                "starter": {"name": program.starter_name, "smiles": program.starter_smiles},
                "extensions": len(program.cycles),
                "cycles": [{"extension": i, "extender": "malonyl", "reduction": c.reduction,
                            "methyl_source": CMT_METHYL if c.methyl else None}
                           for i, c in enumerate(program.cycles, 1)],
                "ring_closures": [_fold_record(f) for f in program.folds],
                "release": {"kind": program.release.kind, "position": program.release.position,
                            "reagent_smiles": program.release.reagent_smiles},
                "finishing": [_fold_record(f) for f in program.finishing],
                "program": asdict(program)}
    _, generate = _executors()
    return {"family": family, "executor": "generate.execute_program",
            "starter": {"name": program.starter_name, "smiles": program.starter_smiles},
            "extensions": program.extensions,
            "cycles": [{"extension": i, "extender": "malonyl", "reduction": "keto", "methyl_source": None}
                       for i in range(1, program.extensions + 1)],
            "ring_closures": [{"kind": s.operation, "positions": list(s.positions)}
                              for s in program.steps if s.operation in RING_CLOSING_STEPS],
            "release": {"kind": next(s.operation for s in program.steps if s.operation in generate.RELEASE_OPERATIONS),
                        "position": 0, "reagent_smiles": ""},
            "finishing": [{"kind": s.operation, "positions": list(s.positions)}
                          for s in program.steps if s.operation not in RING_CLOSING_STEPS
                          and s.operation not in generate.RELEASE_OPERATIONS],
            "program": asdict(program)}


def assembly_program_key(summary):
    """Identity of a program: names, evidence labels, references and KR stereo labels do not enter."""
    fields = {"family": summary["family"], "starter": _canonical_starter(summary["starter"]["smiles"]),
              "cycles": [[c["extender"], c["reduction"], c["methyl_source"]] for c in summary["cycles"]],
              "ring_closures": summary["ring_closures"], "release": summary["release"],
              "finishing": summary["finishing"]}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def _forward_smiles(family, program):
    euk, generate = _executors()
    if family in ("type_ii", "type_iii"):
        return generate.execute_program(program)["smiles"]
    return euk.execute_variants(program, detail="program")[0]["smiles"]


def _enumerated_candidates(family, carbons):
    """Every program of the declared space whose product has exactly ``carbons`` carbon atoms.

    The count is exact because starter carbons, 2 per extension, one per methyl branch and the
    carbons removed by decarboxylation or deacetylation are the only carbon changes in the space
    (pinned by a test that executes sampled candidates).
    """
    euk, _ = _executors()
    space = _enumerated_space()
    if family in ("type_ii", "type_iii"):
        return list(space["iterative"].get(family, {}).get(carbons, ()))
    candidates = [euk.program_from_recipe(kind, recipe, name, smiles, cycles)
                  for fam, kind, recipe, cycles, change in space["frames"] if fam == family
                  for name, smiles in space["starters"][kind].get(carbons - change, ())]
    if family == "plant_type_iii":
        candidates += space["special"]
    return candidates


def _search_enumerated(family, target, budget):
    result = {"status": None, "reason": None, "candidate_programs": 0, "executed": 0,
              "execution_errors": 0, "searched_fully": True, "programs": []}
    outside = target["classes"] - _enumerated_space()["classes"][family]
    if outside:
        names = sorted(Chem.GetPeriodicTable().GetElementSymbol(z) + (" (aromatic)" if aromatic else "")
                       for z, aromatic in outside)
        return {**result, "status": "unsupported", "reason": "atom classes outside the starter pools: " + ", ".join(names)}
    candidates = _enumerated_candidates(family, target["carbons"])
    result["candidate_programs"] = len(candidates)
    for program in candidates[:budget]:
        result["executed"] += 1
        try:
            product = _forward_smiles(family, program)
            if execution.identity(product, stereo=False) != target["constitution"]:
                continue
            stereo_match = execution.identity(product) == target["isomeric"]
            stereo = {**execution.stereo_assignment_difference(product, target["isomeric"]), "compared_with": "input"}
        except (ValueError, RuntimeError):
            result["execution_errors"] += 1
            continue
        summary = summarize_forward_program(family, program)
        summary["program_id"] = assembly_program_key(summary)
        summary["product"] = {"smiles": product, "constitution_match": True, "stereo_match": stereo_match,
                              "stereo": stereo, "form": "released_product"}
        result["programs"].append(summary)
    result["searched_fully"] = result["executed"] == len(candidates)
    result["status"] = ("reproduced" if result["programs"] else
                        "exhausted_none" if result["searched_fully"] else "budget_exhausted")
    return result


def _search_modular(smiles, backbone_top_k):
    result = {"status": None, "reason": None, "searched_fully": True, "programs": []}
    if not can_match_assembly_elements(smiles):
        return {**result, "status": "unsupported", "reason": "atom classes outside the modular substrate pool"}
    chemistry, starters, _, _ = _assembly_resources()
    _, metadata = decompose_pks_terminal(smiles, backbone_top_k=backbone_top_k)
    if metadata["status"] in ("missing_smiles", "invalid_smiles", "disconnected_input", "decomposition_error"):
        return {**result, "status": "unsupported", "reason": metadata.get("error") or metadata["status"]}
    for candidate in metadata.get("candidates", ()):
        validation = candidate["assembly_validation"]
        if not (validation.get("constitution_match") and validation.get("release_roundtrip")):
            continue
        rows = [r for r in candidate["module_rows"] if r["module_num"] > 0]
        summary = {
            "family": "modular_type_i",
            "executor": "predict_reduction_kr_subtype._simulate_bgc_intermediates (whole-molecule rebuild)",
            "starter": {"name": candidate["starter_unit"],
                        "smiles": validation.get("forward_program", {}).get("starter_smiles",
                                   starters[candidate["starter_unit"]])},
            "extensions": len(rows),
            "cycles": [{"extension": r["module_num"], "extender": chemistry._normalize_extender_key(r["at_annotation"]),
                        "reduction": BETA_STATE_REDUCTION.get(r["beta_state"], r["beta_state"]),
                        "kr_subtype": r["kr_subtype"],
                        "methyl_source": (EXTENDER_METHYL if chemistry._normalize_extender_key(r["at_annotation"]) == "mmal"
                                          else None)} for r in rows],
            "ring_closures": [], "release": {"kind": candidate["release"]["mechanism"]}, "finishing": [],
            "release_normalizations": candidate["release"]["normalizations"],
            "program": {"backbone_len": candidate["backbone_len"], "candidate_id": candidate["candidate_id"]},
            # The rebuilt chain is the linearized precursor; the ring closure is checked by the
            # reversible release cut (release_roundtrip), not executed forward.
            "product": {"smiles": validation["reconstructed_smiles"], "constitution_match": True,
                        "stereo_match": validation["stereo_match"], "form": "linearized_precursor",
                        "stereo": {**execution.stereo_assignment_difference(
                            validation["reconstructed_smiles"], validation["comparison_smiles"]),
                            "compared_with": "linearized_input"}}}
        summary["program_id"] = assembly_program_key(summary)
        if summary["program_id"] not in {p["program_id"] for p in result["programs"]}:
            result["programs"].append(summary)
    limits = [name for name, hit in (("decomposer search budget", metadata.get("search_budget_exhausted")),
                                     ("backbone candidate limit", metadata.get("backbone_candidate_limit_reached")),
                                     ("omitted release candidates", metadata.get("omitted_candidate_count", 0) > 0))
              if hit]
    result["searched_fully"] = not limits
    result["reason"] = "stopped at: " + ", ".join(limits) if limits else None
    result["status"] = ("reproduced" if result["programs"] else
                        "exhausted_none" if result["searched_fully"] else "budget_exhausted")
    return result


def assembly_search_space(budget=DEFAULT_PROGRAM_BUDGET, backbone_top_k=DEFAULT_BACKBONE_TOP_K):
    """The declared bounds an ``exhausted_none`` result is relative to."""
    return {"extensions": list(EXTENSION_RANGE), "max_methylations": MAX_METHYLATIONS,
            "program_budget_per_family": budget, "starter_pool_sizes": _enumerated_space()["pool_sizes"],
            "type_ii_type_iii": "generate.screening_programs(builtin_programs())",
            "modular": "ringbreaker releases, carbon-backbone candidates, whole-molecule rebuild",
            "modular_backbones_per_release": backbone_top_k}


def propose_assembly_programs(smiles, *, budget=DEFAULT_PROGRAM_BUDGET, backbone_top_k=DEFAULT_BACKBONE_TOP_K):
    """Forward programs that reproduce the constitution of ``smiles``, for every supported family.

    ``status`` is the best outcome over the searched families: ``reproduced`` (a program was found),
    ``budget_exhausted`` (none found and some family stopped at its limit), ``exhausted_none`` (every
    searched family examined its whole declared space and found none) or ``unsupported`` (the input
    cannot be represented by any family). ``families`` keeps each family's own status; families that
    are not searched are listed in ``families_not_searched`` and never count toward ``exhausted_none``.
    Programs are listed per family, so one structure explained by several families keeps all of them.
    """
    if budget < 1 or backbone_top_k < 1:
        raise ValueError("budget and backbone_top_k must be positive")
    result = {"input_smiles": smiles, "status": None, "families": {}, "programs": [],
              "families_not_searched": sorted(UNSEARCHED_FAMILIES), "scope": SCOPE,
              "search_space": assembly_search_space(budget, backbone_top_k)}
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    reason = ("invalid SMILES" if mol is None else "disconnected input" if len(Chem.GetMolFrags(mol)) != 1 else
              "charged or radical atoms" if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms())
              else None)
    if reason:
        unsupported = {"status": "unsupported", "reason": reason, "searched_fully": False, "programs": []}
        result["families"] = {family: dict(unsupported) for family in (*ENUMERATED_FAMILIES, "modular_type_i")}
    else:
        target = {"carbons": _carbons(mol), "classes": _extra_classes(mol),
                  "constitution": execution.identity(mol, stereo=False), "isomeric": execution.identity(mol)}
        result["families"] = {family: _search_enumerated(family, target, budget) for family in ENUMERATED_FAMILIES}
        result["families"]["modular_type_i"] = _search_modular(smiles, backbone_top_k)
    for family, reason in UNSEARCHED_FAMILIES.items():
        result["families"][family] = {"status": "unsupported", "reason": reason, "searched_fully": False, "programs": []}
    searched = [row["status"] for family, row in result["families"].items() if family not in UNSEARCHED_FAMILIES]
    result["programs"] = [program for row in result["families"].values() for program in row["programs"]]
    result["status"] = ("reproduced" if "reproduced" in searched else
                        "budget_exhausted" if "budget_exhausted" in searched else
                        "exhausted_none" if "exhausted_none" in searched else "unsupported")
    return result
