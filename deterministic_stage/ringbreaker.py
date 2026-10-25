import argparse
import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def _is_benzene_ring(mol, ring):
    """Return True for a six-membered all-carbon aromatic ring."""
    if len(ring) != 6:
        return False
    return all(
        mol.GetAtomWithIdx(i).GetIsAromatic() and mol.GetAtomWithIdx(i).GetAtomicNum() == 6
        for i in ring
    )

def _is_protected_ring(mol, ring, protect_aromatic):
    if protect_aromatic == "none":
        return False
    if protect_aromatic == "benzene":
        return _is_benzene_ring(mol, ring)
    if protect_aromatic == "any":
        return all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    raise ValueError(f"Unknown protect_aromatic setting: {protect_aromatic}")

def _is_valid_macrolactone_cut(mol, carbonyl_c, ester_o, ring_info, min_ring_size, protect_aromatic):
    """Return True if the ester bond sits in a valid target ring."""
    bond = mol.GetBondBetweenAtoms(carbonyl_c, ester_o)
    if bond is None or not bond.IsInRing():
        return False
    for ring in ring_info.AtomRings():
        if _is_protected_ring(mol, ring, protect_aromatic):
            continue
        if len(ring) >= min_ring_size and carbonyl_c in ring and ester_o in ring:
            return True
    return False

def _longest_carbon_chain_from(atom_idx, mol, visited=None):
    """Return the longest carbon-only path length starting at ``atom_idx``."""
    if visited is None:
        visited = set()
    visited.add(atom_idx)
    atom = mol.GetAtomWithIdx(atom_idx)
    max_len = 0
    for nbr in atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in visited:
            continue
        if nbr.GetAtomicNum() != 6:
            continue
        max_len = max(
            max_len,
            1 + _longest_carbon_chain_from(nbr_idx, mol, visited.copy())
        )
    return max_len

def _has_large_unprotected_ring(mol, ring_info, min_ring_size, protect_aromatic):
    """Return True if any unprotected ring meets the minimum size."""
    for ring in ring_info.AtomRings():
        if _is_protected_ring(mol, ring, protect_aromatic):
            continue
        if len(ring) >= min_ring_size:
            return True
    return False

def _count_carbons(mol):
    """Count carbon atoms in an RDKit molecule."""
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)


def _small_ring_alkenes(mol):
    """Implicit cis relations between ring neighbours of small-ring olefins.

    RDKit omits E/Z tags inside rings smaller than eight atoms. Opening such a
    ring must make that geometrical constraint explicit in the acyclic chain.
    The relation is between ring neighbours, not necessarily CIP Z.
    """
    constraints = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) >= 8:
            continue
        for position, a in enumerate(ring):
            b = ring[(position + 1) % len(ring)]
            bond = mol.GetBondBetweenAtoms(a, b)
            if (bond.GetBondType() != Chem.BondType.DOUBLE or
                    any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in (a, b))):
                continue
            constraints.append({"source_atom_indices": [ring[position - 1], a, b,
                                                          ring[(position + 2) % len(ring)]],
                                "relation": "cis", "ring_size": len(ring),
                                "evidence": "small_ring_topology"})
    return constraints


def _alkene_matches_ring_constraint(mol, indices):
    left, a, b, right = indices
    bond = mol.GetBondBetweenAtoms(a, b)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return False
    stereo = bond.GetStereo()
    if stereo not in (Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOZ,
                      Chem.BondStereo.STEREOTRANS, Chem.BondStereo.STEREOE):
        return False
    refs = [left, right] if bond.GetBeginAtomIdx() == a else [right, left]
    flips = sum(x != y for x, y in zip(refs, bond.GetStereoAtoms()))
    cis = stereo in (Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOZ)
    return cis != bool(flips % 2)


def enumerate_backbone_precursors(smiles, *, min_ring_size=4, open_lactams=True):
    """Retain release hypotheses with explicit anchors and atom correspondence.

    Source indices refer to the canonical, unmapped input, so alternate SMILES
    spellings do not change tie breaking. Maps in each returned SMILES identify
    source atoms (index + 1); an introduced carrier sulfur has map zero.
    This enumerates formal release chemistry, not enzyme substrate acceptance.
    Four-member cuts are restricted to beta-lactones (C(=O)-C-C-O);
    beta-lactams, cyclic carbonates and three-member rings are not implied.
    """
    original = Chem.MolFromSmiles(smiles)
    if original is None or len(Chem.GetMolFrags(original)) != 1:
        return []
    for atom in original.GetAtoms():
        atom.SetAtomMapNum(0)
    source_smiles = Chem.MolToSmiles(original, isomericSmiles=True)
    original = Chem.MolFromSmiles(source_smiles)
    for atom in original.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    candidates = []
    ring_alkenes = _small_ring_alkenes(original)

    def add(mol, c1, mechanism, cut=None, normalizations=()):
        mol = Chem.Mol(mol)
        by_source = {a.GetAtomMapNum() - 1: a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}
        constraints = []
        for constraint in ring_alkenes:
            left, a, b, right = (by_source[i] for i in constraint["source_atom_indices"])
            bond = mol.GetBondBetweenAtoms(a, b)
            if any(len(ring) < 8 and bond.GetIdx() in ring for ring in mol.GetRingInfo().BondRings()):
                continue  # the small-ring constraint is still implicit in this precursor
            refs = (left, right) if bond.GetBeginAtomIdx() == a else (right, left)
            bond.SetStereoAtoms(*refs)
            bond.SetStereo(Chem.BondStereo.STEREOCIS)
            constraints.append(constraint)
        Chem.SetDoubleBondNeighborDirections(mol)
        mapped = Chem.MolToSmiles(mol, isomericSmiles=True)
        parsed = Chem.MolFromSmiles(mapped)
        anchor = next(a.GetIdx() for a in parsed.GetAtoms() if a.GetAtomMapNum() == c1 + 1)
        candidates.append({
            "mol": parsed, "smiles": mapped, "c1_idx": anchor,
            "c1_source_atom_index": c1, "source_smiles": source_smiles,
            "mechanism": mechanism, "cut_source_atom_indices": list(cut) if cut else None,
            "normalizations": list(normalizations),
            "ring_alkene_constraints": constraints,
            "normalization_limit_reached": False,
            "source_atom_indices": [a.GetAtomMapNum() - 1 if a.GetAtomMapNum() else None
                                    for a in parsed.GetAtoms()],
        })

    # A native terminus is also a hypothesis when the molecule contains other
    # rings. Do not silently choose one of several acid groups as C1.
    for kind, pattern in (("thioester", "[CX3](=O)[S;D1]"),
                          ("carboxylic_acid", "[CX3](=O)[OX2H1]")):
        for match in original.GetSubstructMatches(Chem.MolFromSmarts(pattern)):
            add(original, match[0], kind)

    # A terminal reductase can release the carrier-bound acyl chain as an
    # aldehyde. Restore only the carrier bond; the entire proposed chain still
    # has to be rebuilt from the substrate whitelist by the assembly validator.
    for carbonyl, _ in original.GetSubstructMatches(Chem.MolFromSmarts("[CX3H1](=O)")):
        editable = Chem.RWMol(original)
        atom = editable.GetAtomWithIdx(carbonyl)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        sulfur = editable.AddAtom(Chem.Atom(16))
        editable.AddBond(carbonyl, sulfur, Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(editable)
        except (ValueError, RuntimeError):
            continue
        add(editable.GetMol(), carbonyl, "reductive_aldehyde")

    patterns = [("lactonization", "[CX3](=O)[OX2H0]")]
    if open_lactams:
        patterns.append(("lactamization", "[CX3](=O)[NX3]"))
    for mechanism, pattern in patterns:
        for carbonyl, _, hetero in original.GetSubstructMatches(Chem.MolFromSmarts(pattern)):
            bond = original.GetBondBetweenAtoms(carbonyl, hetero)
            if not bond.IsInRing():
                continue
            editable = Chem.RWMol(original)
            editable.RemoveBond(carbonyl, hetero)
            # Use the remaining bond path, not the SSSR ring basis: fused and
            # bridged rings can have several valid cycle representations.
            path = Chem.GetShortestPath(editable, carbonyl, hetero)
            if len(path) < max(4, min_ring_size):
                continue
            if len(path) == 4 and (
                mechanism != "lactonization"
                or [original.GetAtomWithIdx(i).GetAtomicNum() for i in path] != [6, 6, 6, 8]
                or any(original.GetBondBetweenAtoms(a, b).GetBondType() != Chem.BondType.SINGLE
                       for a, b in zip(path, path[1:]))
            ):
                continue
            sulfur = editable.AddAtom(Chem.Atom(16))
            editable.AddBond(carbonyl, sulfur, Chem.BondType.SINGLE)
            atom = editable.GetAtomWithIdx(hetero)
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            try:
                Chem.SanitizeMol(editable)
            except (ValueError, RuntimeError):
                continue
            if len(Chem.GetMolFrags(editable)) == 1:
                add(editable.GetMol(), carbonyl, mechanism, (carbonyl, hetero))
    # A cyclic hemiketal masks a PKS beta ketone. Retain both interpretations;
    # opening it is an explicitly recorded equilibrium normalization, not an
    # additional enzyme/tailoring step. Limit combinations, never silently prune.
    hemiketal = Chem.MolFromSmarts("[C;X4;H0]([O;H1;D1])([O;R;H0;D2])")
    for base in list(candidates):
        queue, seen = [base], {base["smiles"]}
        for current in queue:
            for carbon, hydroxyl, ether in current["mol"].GetSubstructMatches(hemiketal):
                c_atom = current["mol"].GetAtomWithIdx(carbon)
                if sum(a.GetAtomicNum() == 6 for a in c_atom.GetNeighbors()) != 2:
                    continue
                if not current["mol"].GetBondBetweenAtoms(carbon, ether).IsInRing():
                    continue
                editable = Chem.RWMol(current["mol"])
                editable.RemoveBond(carbon, ether)
                editable.GetBondBetweenAtoms(carbon, hydroxyl).SetBondType(Chem.BondType.DOUBLE)
                editable.GetAtomWithIdx(carbon).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                editable.GetAtomWithIdx(hydroxyl).SetNumExplicitHs(0)
                editable.GetAtomWithIdx(ether).SetNumExplicitHs(1)
                try:
                    Chem.SanitizeMol(editable)
                except (ValueError, RuntimeError):
                    continue
                key = Chem.MolToSmiles(editable, isomericSmiles=True)
                if key in seen:
                    continue
                if len(seen) >= 16:
                    for item in queue:
                        item["normalization_limit_reached"] = True
                    continue
                seen.add(key)
                operation = {"operation": "hemiketal_opening",
                             "source_atom_indices": [current["mol"].GetAtomWithIdx(i).GetAtomMapNum() - 1
                                                     for i in (carbon, hydroxyl, ether)],
                             "closure_stereo": "observed_ring_hypothesis"}
                add(editable, base["c1_source_atom_index"], base["mechanism"],
                    base["cut_source_atom_indices"], current["normalizations"] + [operation])
                queue.append(candidates[-1])
    return candidates


def release_forward_product(candidate, reconstructed_smiles=None):
    """Execute the recorded release, optionally on an independently rebuilt chain.

    ``reconstructed_smiles`` uses the assembly validator's acid-normalized carrier
    representation. Its stereo is retained through atom correspondence; the
    observed source supplies only release anchors and an explicit hemiketal
    closure hypothesis, never the rest of the rebuilt product.
    """
    mol = Chem.RWMol(candidate["mol"])
    if reconstructed_smiles is not None:
        rebuilt = Chem.MolFromSmiles(reconstructed_smiles)
        if rebuilt is None:
            return None
        query = Chem.Mol(mol)
        sulfur_indices = set()
        for match in query.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[S;D1]")):
            sulfur_indices.add(match[2])
            atom = query.GetAtomWithIdx(match[2])
            atom.SetAtomicNum(8)
            atom.SetNumRadicalElectrons(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
        for atom in query.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.SanitizeMol(query)
        match = query.GetSubstructMatch(rebuilt, useChirality=False)
        if len(match) != query.GetNumAtoms() or rebuilt.GetNumAtoms() != query.GetNumAtoms():
            return None
        mapped = Chem.RWMol(rebuilt)
        for index, source_index in enumerate(match):
            source_atom = mol.GetAtomWithIdx(source_index)
            if source_index in sulfur_indices:
                mapped.ReplaceAtom(index, Chem.Atom(source_atom))
            else:
                mapped.GetAtomWithIdx(index).SetAtomMapNum(source_atom.GetAtomMapNum())
        mol = mapped
    by_map = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}
    # Reject a trans (or unspecified) precursor before small-ring closure can
    # cause RDKit serialization to silently discard its alkene stereo tag.
    for constraint in candidate.get("ring_alkene_constraints", []):
        indices = [by_map[i + 1] for i in constraint["source_atom_indices"]]
        if not _alkene_matches_ring_constraint(mol, indices):
            return None
    source = Chem.MolFromSmiles(candidate["source_smiles"])
    for operation in reversed(candidate["normalizations"]):
        carbon, hydroxyl, ether = operation["source_atom_indices"]
        c, o, e = (by_map[i + 1] for i in (carbon, hydroxyl, ether))
        mol.GetBondBetweenAtoms(c, o).SetBondType(Chem.BondType.SINGLE)
        mol.GetAtomWithIdx(o).SetNumExplicitHs(1)
        mol.GetAtomWithIdx(e).SetNumExplicitHs(0)
        mol.AddBond(c, e, Chem.BondType.SINGLE)
        # The newly formed hemiketal centre belongs to this closure hypothesis;
        # it is not an inferred KR stereocentre. Preserve all other stereo.
        old = source.GetAtomWithIdx(carbon)
        old_order = [a.GetIdx() for a in old.GetNeighbors()]
        new_order = [a.GetAtomMapNum() - 1 for a in mol.GetAtomWithIdx(c).GetNeighbors()]
        order = [old_order.index(i) for i in new_order]
        inversions = sum(order[i] > order[j] for i in range(len(order)) for j in range(i + 1, len(order)))
        mol.GetAtomWithIdx(c).SetChiralTag(old.GetChiralTag())
        if inversions % 2:
            mol.GetAtomWithIdx(c).InvertChirality()
    if candidate["mechanism"] == "reductive_aldehyde":
        c1 = by_map[candidate["c1_source_atom_index"] + 1]
        sulfur = [a.GetIdx() for a in mol.GetAtomWithIdx(c1).GetNeighbors()
                  if a.GetAtomicNum() == 16 and a.GetAtomMapNum() == 0]
        if len(sulfur) != 1:
            return None
        mol.GetAtomWithIdx(c1).SetNumExplicitHs(1)
        mol.RemoveAtom(sulfur[0])
    elif candidate["cut_source_atom_indices"] is not None:
        carbonyl, hetero = candidate["cut_source_atom_indices"]
        c1, nucleophile = by_map[carbonyl + 1], by_map[hetero + 1]
        sulfur = [a.GetIdx() for a in mol.GetAtomWithIdx(c1).GetNeighbors()
                  if a.GetAtomicNum() == 16 and a.GetAtomMapNum() == 0]
        if len(sulfur) != 1:
            return None
        mol.AddBond(c1, nucleophile, Chem.BondType.SINGLE)
        atom = mol.GetAtomWithIdx(nucleophile)
        atom.SetNumExplicitHs(max(0, atom.GetNumExplicitHs() - 1))
        mol.RemoveAtom(sulfur[0])
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except (ValueError, RuntimeError):
        return None


def verify_release_roundtrip(candidate):
    """Check that the recorded release restores the source, including stereo."""
    return release_forward_product(candidate) == candidate["source_smiles"]

def prepare_for_backbone_analysis(
    smiles,
    name="Unknown",
    min_ring_size=7,
    protect_aromatic="benzene",
    warn_on_unbroken=False,
    open_lactams=False,
):
    """Linearize macrocyclic releases for downstream backbone analysis.

    Macrolactone opening remains the default behavior. ``open_lactams`` is an
    opt-in used by retrosynthesis terminal validation, where a released
    macrolactam must be mapped back to its pre-TE carrier thioester.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "mol": None,
            "smiles": None,
        }
    
    if min_ring_size < 3:
        raise ValueError("Minimum ring size must be greater than or equal to 3")

    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() > 0:
        ester_pattern = Chem.MolFromSmarts("[#6][CX3](=O)[OX2H0][#6]")
        matches = mol.GetSubstructMatches(ester_pattern) if ester_pattern else []
        candidates = []
        for match in matches:
            _, carbonyl_c, _, ester_o, _ = match
            if _is_valid_macrolactone_cut(
                mol, carbonyl_c, ester_o, ring_info, min_ring_size, protect_aromatic
            ):
                candidates.append((carbonyl_c, ester_o))

        # Select the cut that linearizes the ring while maximizing downstream carbon continuity.
        best_mol = None
        best_score = -1
        orig_rings = ring_info.NumRings()
        for carbonyl_c, ester_o in candidates:
            try:
                em = Chem.RWMol(mol)
                em.RemoveBond(carbonyl_c, ester_o)
                s_idx = em.AddAtom(Chem.Atom(16))
                em.AddBond(carbonyl_c, s_idx, Chem.BondType.SINGLE)
                Chem.SanitizeMol(em)
                trial_mol = em.GetMol()
            except Exception:
                continue

            new_rings = trial_mol.GetRingInfo().NumRings()
            frags = Chem.GetMolFrags(trial_mol, asMols=False)
            n_frags = len(frags)
            if new_rings >= orig_rings:
                continue
            if n_frags > 1:
                warnings.warn(
                    f"Ring count decreased but produced {n_frags} fragments "
                    f"(macrolactone cut C{carbonyl_c}-O{ester_o}) - skipping",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            score = _longest_carbon_chain_from(carbonyl_c, trial_mol)
            if score > best_score:
                best_mol = trial_mol
                best_score = score
        if best_mol is not None:
            mol = best_mol

    if open_lactams and _has_large_unprotected_ring(
        mol, mol.GetRingInfo(), min_ring_size, protect_aromatic
    ):
        ring_info = mol.GetRingInfo()
        amide_pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        candidates = set()
        for match in mol.GetSubstructMatches(amide_pattern) if amide_pattern else []:
            carbonyl_c, _, amide_n = match
            bond = mol.GetBondBetweenAtoms(carbonyl_c, amide_n)
            if bond is None or not bond.IsInRing():
                continue
            if any(
                len(ring) >= min_ring_size
                and carbonyl_c in ring
                and amide_n in ring
                and not _is_protected_ring(mol, ring, protect_aromatic)
                for ring in ring_info.AtomRings()
            ):
                candidates.add((carbonyl_c, amide_n))

        best_mol = None
        best_score = -1
        original_rings = ring_info.NumRings()
        for carbonyl_c, amide_n in sorted(candidates):
            try:
                editable = Chem.RWMol(mol)
                editable.RemoveBond(carbonyl_c, amide_n)
                sulfur_idx = editable.AddAtom(Chem.Atom(16))
                editable.AddBond(carbonyl_c, sulfur_idx, Chem.BondType.SINGLE)
                Chem.SanitizeMol(editable)
                trial_mol = editable.GetMol()
            except Exception:
                continue
            if trial_mol.GetRingInfo().NumRings() >= original_rings:
                continue
            if len(Chem.GetMolFrags(trial_mol, asMols=False)) > 1:
                continue
            score = _longest_carbon_chain_from(carbonyl_c, trial_mol)
            if score > best_score:
                best_mol = trial_mol
                best_score = score
        if best_mol is not None:
            mol = best_mol

    ring_info = mol.GetRingInfo()
    remaining_large_ring = _has_large_unprotected_ring(mol, ring_info, min_ring_size, protect_aromatic)
    if warn_on_unbroken and remaining_large_ring:
        warnings.warn(
            f"{name}: large ring remains unbroken (min_ring_size={min_ring_size})",
            UserWarning,
            stacklevel=2,
        )

    mol.SetProp("_Name", f"{name}_linearized")
    
    return {
        "mol": mol,
        "smiles": Chem.MolToSmiles(mol),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ring-break macrocyclic PKS products into linearized thioesters."
    )
    parser.add_argument("--input_csv", default=str(DATA_DIR / "test_set.csv"))
    parser.add_argument("--output_csv", default=str(DATA_DIR / "test_set_linearized.csv"))
    parser.add_argument("--min_ring_size", type=int, default=7)
    parser.add_argument(
        "--protect_aromatic",
        choices=["benzene", "any", "none"],
        default="benzene",
        help="Aromatic ring protection mode when evaluating cuts.",
    )
    parser.add_argument(
        "--warn_on_unbroken",
        action="store_true",
        help="Warn when a large ring remains unbroken after scanning.",
    )
    args = parser.parse_args()

    test_df = pd.read_csv(args.input_csv)

    results = []
    for _, row in test_df.iterrows():
        name = row["cluster_name"]
        smiles = str(row["target_smiles"])
        result = prepare_for_backbone_analysis(
            smiles,
            name,
            min_ring_size=args.min_ring_size,
            protect_aromatic=args.protect_aromatic,
            warn_on_unbroken=args.warn_on_unbroken,
        )
        results.append(result)
    
    output_data = []
    for idx, row in test_df.iterrows():
        bgc_id = row["bgc_id"]
        result = results[idx]

        orig_smiles = str(row["target_smiles"])
        orig_mol = Chem.MolFromSmiles(orig_smiles)
        orig_carbon_count = _count_carbons(orig_mol)

        lin_carbon_count = 0
        if result.get("mol"):
            lin_carbon_count = _count_carbons(result["mol"])
        elif result.get("smiles"):
            lin_mol = Chem.MolFromSmiles(result["smiles"])
            lin_carbon_count = _count_carbons(lin_mol)

        output_row = {
            "bgc_id": bgc_id,
            "cluster_name": row["cluster_name"],
            "linearized_smiles": result["smiles"] if result["smiles"] else "",
            "original_carbon_count": orig_carbon_count,
            "linearized_carbon_count": lin_carbon_count,
        }

        output_data.append(output_row)
    
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(args.output_csv, index=False)
    
    print("Macrolactone rings linearized")
