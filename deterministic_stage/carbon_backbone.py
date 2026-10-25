import argparse
import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def _import_networkx():
    try:
        import networkx as nx  # type: ignore
        return nx
    except ImportError:
        nx_repo = PROJECT_ROOT / "networkx"
        if nx_repo.exists():
            sys.path.insert(0, str(nx_repo))
        import networkx as nx  # type: ignore
        return nx

nx = _import_networkx()

C1_ANCHOR_PATTERNS = [
    ("thioester", "[CX3](=O)[S]"),  # Primary C1 anchor (native or introduced during linearization)
    ("carboxylic_acid", "[CX3](=O)[OX2H1]"),  # Secondary C1 anchor when no thioester is present
]


class NXBackboneAnalyzer:
    """PKS backbone extraction utilities based on RDKit + NetworkX."""

    C1_ANCHOR_PATTERNS = C1_ANCHOR_PATTERNS

    @staticmethod
    def _carbon_neighbor_count(mol, atom_idx):
        atom = mol.GetAtomWithIdx(atom_idx)
        return sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 6)

    @staticmethod
    def _thioester_terminal_bonus(mol, atom_idx):
        atom = mol.GetAtomWithIdx(atom_idx)
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() == 16 and nb.GetDegree() == 1:
                return 1
        return 0

    @classmethod
    def find_c1(cls, mol):
        """Return deterministic C1 anchor as (terminus_type, atom_index)."""
        for name, smarts in cls.C1_ANCHOR_PATTERNS:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                continue

            candidates = []
            for match in matches:
                c1_idx = match[0]
                c_nbrs = cls._carbon_neighbor_count(mol, c1_idx)
                terminal_like = 1 if c_nbrs == 1 else 0
                bonus = cls._thioester_terminal_bonus(mol, c1_idx) if name == "thioester" else 0
                candidates.append((bonus, terminal_like, -c_nbrs, c1_idx))

            _, _, _, best_idx = max(candidates)
            return name, best_idx

        return None, None

    @staticmethod
    def _is_carbonyl_carbon(atom):
        """True if carbon has a double-bonded oxygen neighbor (C=O)."""
        if atom.GetAtomicNum() != 6:
            return False
        for b in atom.GetBonds():
            if b.GetBondTypeAsDouble() == 2.0:
                other = b.GetOtherAtom(atom)
                if other.GetAtomicNum() == 8:
                    return True
        return False

    @staticmethod
    def _find_sugar_ring_atoms(mol):
        """Return atom indices for sugar-like rings to optionally exclude."""
        sugar_atoms = set()
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if len(ring) not in (5, 6):
                continue

            atoms = [mol.GetAtomWithIdx(i) for i in ring]
            hetero = [a for a in atoms if a.GetAtomicNum() in (7, 8, 16)]
            o_count = sum(1 for a in atoms if a.GetAtomicNum() == 8)

            if o_count < 1:
                continue
            if len(hetero) < 2:
                continue

            ring_set = set(ring)
            has_exocyclic_o = False
            for a in atoms:
                for nb in a.GetNeighbors():
                    if nb.GetAtomicNum() == 8 and nb.GetIdx() not in ring_set:
                        has_exocyclic_o = True
                        break
                if has_exocyclic_o:
                    break

            if has_exocyclic_o:
                sugar_atoms.update(ring)

        return sugar_atoms

    @staticmethod
    def _is_methyl_branch_atom(atom):
        """Return True for terminal methyl-like carbon atoms."""
        if atom.GetAtomicNum() != 6:
            return False
        if atom.GetDegree() != 1:
            return False
        return atom.GetTotalNumHs() == 3

    @classmethod
    def _build_carbon_graph(cls, mol, allowed_indices):
        """Build carbon-only graph with node annotations used by path scoring."""
        G = nx.Graph()
        allowed = set(allowed_indices)
        for idx in allowed:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() != 6:
                continue
            G.add_node(
                idx,
                atom_idx=idx,
                is_methyl_branch=cls._is_methyl_branch_atom(atom),
                is_aromatic=atom.GetIsAromatic(),
                is_carbonyl=cls._is_carbonyl_carbon(atom),
                degree=atom.GetDegree(),
                carbon_degree=sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 6),
            )

        for bond in mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            if a in allowed and b in allowed:
                G.add_edge(a, b, bond_type=bond.GetBondTypeAsDouble())

        return G

    @classmethod
    def _score_path_lexicographic(cls, G, path, *, early_k=10):
        """
        Deterministic lexicographic score tuple:
          1) methyl_early:   # methyl-branch atoms in early path (minimize)
          2) aromatic_early: # aromatic atoms in early path (minimize)
          3) β score:     soft PKS parity plausibility (maximize)
          4) length:         path length (maximize; tie-break)
        """
        # 1) methyl early
        methyl_early = 0
        aromatic_early = 0
        for idx in path[1 : 1 + early_k]:
            node = G.nodes[idx]
            if node.get("is_methyl_branch"):
                methyl_early += 1
            if node.get("is_aromatic"):
                aromatic_early += 1

        # 2) soft beta-parity plausibility
        even_carbonyl = 0
        odd_carbonyl = 0
        for pos, idx in enumerate(path):
            if G.nodes[idx].get("is_carbonyl"):
                if pos % 2 == 0:
                    even_carbonyl += 1
                else:
                    odd_carbonyl += 1
        beta_score = 5 * even_carbonyl - 1 * odd_carbonyl

        return (methyl_early, aromatic_early, -beta_score, -len(path))

    @classmethod
    def _best_paths_from_c1_dfs(
        cls,
        G,
        c1_idx,
        *,
        max_visits=500000,
        early_k=10,
        top_k=8,
        diverse=False,
    ):
        """
        Enumerate simple C1-anchored paths and retain the top-K unique paths by
        the lexicographic PKS-consistency score.
        """
        visits = 0
        path = [c1_idx]
        visited = {c1_idx}
        best_items = []
        longest_items = []

        def neighbor_key(v):
            node = G.nodes[v]
            return (node.get("is_methyl_branch", False), node.get("is_aromatic", False), v)

        def maybe_record(current_path):
            nonlocal best_items, longest_items
            path_tuple = tuple(current_path)
            score = cls._score_path_lexicographic(G, current_path, early_k=early_k)
            if diverse:
                longest_items.append((score, path_tuple))
                longest_items.sort(key=lambda x: (-len(x[1]), x[0], x[1]))
                longest_items = longest_items[:max(1, int(top_k))]

            for i, (existing_score, existing_path) in enumerate(best_items):
                if existing_path == path_tuple:
                    if score < existing_score:
                        best_items[i] = (score, path_tuple)
                        best_items.sort(key=lambda x: (x[0], x[1]))
                    return

            best_items.append((score, path_tuple))
            best_items.sort(key=lambda x: (x[0], x[1]))
            if len(best_items) > max(1, int(top_k)):
                best_items = best_items[: max(1, int(top_k))]

        def dfs(u):
            nonlocal visits
            visits += 1
            if visits > max_visits:
                return

            maybe_record(path)

            for v in sorted(G.neighbors(u), key=neighbor_key):
                if v in visited:
                    continue
                visited.add(v)
                path.append(v)
                dfs(v)
                path.pop()
                visited.remove(v)

        dfs(c1_idx)
        exhausted_budget = visits > max_visits
        if diverse:
            merged = []
            for rank in range(max(len(best_items), len(longest_items))):
                for items in (best_items, longest_items):
                    if rank < len(items) and items[rank][1] not in merged:
                        merged.append(items[rank][1])
            return [list(p) for p in merged[:max(1, int(top_k))]], exhausted_budget
        return [list(p) for _, p in best_items], exhausted_budget

    @classmethod
    def _best_path_from_c1_dfs(cls, G, c1_idx, *, max_visits=500000, early_k=10):
        """
        Enumerate simple paths starting at C1 with a visit budget.
        Choose best path by lexicographic score (not longest).
        """
        paths, exhausted_budget = cls._best_paths_from_c1_dfs(
            G,
            c1_idx,
            max_visits=max_visits,
            early_k=early_k,
            top_k=1,
        )
        if not paths:
            return [c1_idx], exhausted_budget
        return paths[0], exhausted_budget

    @staticmethod
    def _correct_gem_dimethyl_terminus(mol, backbone):
        """Append second methyl if terminal carbon is part of a gem-dimethyl."""
        if len(backbone) < 3:
            return backbone
        terminal = backbone[-1]
        parent = backbone[-2]

        term_atom = mol.GetAtomWithIdx(terminal)
        parent_atom = mol.GetAtomWithIdx(parent)

        carbon_neighbors = [nb.GetIdx() for nb in term_atom.GetNeighbors() if nb.GetAtomicNum() == 6]
        if len(carbon_neighbors) != 1 or term_atom.GetTotalNumHs() < 2:
            return backbone

        backbone_set = set(backbone)
        for nb in parent_atom.GetNeighbors():
            if nb.GetAtomicNum() != 6:
                continue
            nb_idx = nb.GetIdx()
            if nb_idx in backbone_set:
                continue
            c_nbrs = sum(1 for x in nb.GetNeighbors() if x.GetAtomicNum() == 6)
            if c_nbrs == 1 and nb.GetTotalNumHs() >= 2:
                return backbone + [nb_idx]
        return backbone

    @classmethod
    def _extend_backbone_with_rings(
        cls,
        mol,
        backbone,
        *,
        ring_sizes=None,
    ):
        """
        Extend backbone to include ring carbons and terminal exocyclic carbonyls
        attached to any ring connected to the backbone.
        """
        if not backbone:
            return backbone

        backbone_set = set(backbone)
        pos_map = {idx: pos for pos, idx in enumerate(backbone)}
        ring_info = mol.GetRingInfo()
        sugar_atoms = cls._find_sugar_ring_atoms(mol)

        extensions_by_anchor = {}
        added_global = set()

        rings = list(ring_info.AtomRings())
        rings.sort(key=lambda r: (min(r), len(r)))

        for ring in rings:
            if ring_sizes is not None and len(ring) not in ring_sizes:
                continue

            ring_list = list(ring)
            ring_set = set(ring_list)
            if ring_set & sugar_atoms:
                continue

            attach_positions = []
            ring_backbone_atoms = []
            for r_idx in ring_set:
                if r_idx in pos_map:
                    attach_positions.append(pos_map[r_idx])
                    ring_backbone_atoms.append(r_idx)
                for nb in mol.GetAtomWithIdx(r_idx).GetNeighbors():
                    nb_idx = nb.GetIdx()
                    if nb_idx in pos_map:
                        attach_positions.append(pos_map[nb_idx])

            if not attach_positions:
                continue

            anchor_pos = max(attach_positions)

            if ring_backbone_atoms:
                entry = max(ring_backbone_atoms, key=lambda i: pos_map[i])
            else:
                entry = None
                anchor_atom = backbone[anchor_pos]
                ring_neighbors = [
                    r_idx
                    for r_idx in ring_set
                    if any(
                        nb.GetIdx() == anchor_atom
                        for nb in mol.GetAtomWithIdx(r_idx).GetNeighbors()
                    )
                ]
                if ring_neighbors:
                    entry = min(ring_neighbors)
                else:
                    entry = min(ring_set)

            if entry in ring_list:
                start_idx = ring_list.index(entry)
                ring_order = ring_list[start_idx:] + ring_list[:start_idx]
            else:
                ring_order = ring_list

            extension = []
            extension_set = set()
            for idx in ring_order:
                if idx in backbone_set:
                    continue
                if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6 and idx not in extension_set:
                    extension.append(idx)
                    extension_set.add(idx)

            # Add terminal exocyclic carbonyls attached to ring
            for r_idx in ring_set:
                for nb in mol.GetAtomWithIdx(r_idx).GetNeighbors():
                    nb_idx = nb.GetIdx()
                    if nb_idx in backbone_set or nb_idx in ring_set or nb_idx in extension_set:
                        continue
                    nb_atom = mol.GetAtomWithIdx(nb_idx)
                    if nb_atom.GetAtomicNum() != 6:
                        continue
                    if not cls._is_carbonyl_carbon(nb_atom):
                        continue
                    carbon_neighbors = sum(1 for x in nb_atom.GetNeighbors() if x.GetAtomicNum() == 6)
                    if carbon_neighbors <= 1:
                        extension.append(nb_idx)
                        extension_set.add(nb_idx)

            if extension:
                extensions_by_anchor.setdefault(anchor_pos, []).append(extension)

        if not extensions_by_anchor:
            return backbone

        new_backbone = []
        for pos, atom_idx in enumerate(backbone):
            new_backbone.append(atom_idx)
            if pos in extensions_by_anchor:
                for extension in extensions_by_anchor[pos]:
                    for idx in extension:
                        if idx in backbone_set or idx in added_global:
                            continue
                        new_backbone.append(idx)
                        added_global.add(idx)

        return new_backbone

    @classmethod
    def find_pks_backbone(
        cls,
        mol,
        c1_idx,
        *,
        exclude_sugar_rings=True,
        max_visits=500000,
        early_k=10,
        include_ring_extensions=False,
        allowed_atom_indices=None,
    ):
        """
        Backbone selection grounded in PKS stoichiometry (deterministic):
          1) Build carbon-only NetworkX graph; optionally drop sugar-like rings only.
          2) Restrict to carbon component reachable from C1.
          3) Enumerate C1-anchored simple paths with DFS (visit budget).
          4) Select best path by lexicographic PKS-consistency score.
          5) Return the strict core PKS path only.
        """
        if c1_idx is None:
            return [], False
        c1_atom = mol.GetAtomWithIdx(c1_idx)
        if c1_atom.GetAtomicNum() != 6:
            return [], False

        carbon_set = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}

        if exclude_sugar_rings:
            sugar_atoms = cls._find_sugar_ring_atoms(mol)
            carbon_set -= sugar_atoms

        if allowed_atom_indices is not None:
            carbon_set &= set(allowed_atom_indices)
            carbon_set.add(c1_idx)

        if c1_idx not in carbon_set:
            return [], False

        G = cls._build_carbon_graph(mol, carbon_set)
        if c1_idx not in G:
            return [], False

        reachable = nx.node_connected_component(G, c1_idx)
        if not reachable:
            return [], False

        G = G.subgraph(reachable).copy()

        backbone, exhausted = cls._best_path_from_c1_dfs(
            G, c1_idx, max_visits=max_visits, early_k=early_k
        )
        # Strict stoichiometric PKS core only: do not append ring carbons or
        # gem-dimethyl side branches after core path selection.
        return backbone, exhausted

    @classmethod
    def find_pks_backbone_candidates(
        cls,
        mol,
        c1_idx,
        *,
        exclude_sugar_rings=True,
        max_visits=500000,
        early_k=10,
        top_k=8,
        allowed_atom_indices=None,
        diverse=False,
    ):
        """
        Return top-K strict C1-anchored PKS core backbone candidates.
        """
        if c1_idx is None:
            return [], False
        c1_atom = mol.GetAtomWithIdx(c1_idx)
        if c1_atom.GetAtomicNum() != 6:
            return [], False

        carbon_set = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}
        if exclude_sugar_rings:
            sugar_atoms = cls._find_sugar_ring_atoms(mol)
            carbon_set -= sugar_atoms
        if allowed_atom_indices is not None:
            carbon_set &= set(allowed_atom_indices)
            carbon_set.add(c1_idx)
        if c1_idx not in carbon_set:
            return [], False

        G = cls._build_carbon_graph(mol, carbon_set)
        if c1_idx not in G:
            return [], False

        reachable = nx.node_connected_component(G, c1_idx)
        if not reachable:
            return [], False

        G = G.subgraph(reachable).copy()
        return cls._best_paths_from_c1_dfs(
            G,
            c1_idx,
            max_visits=max_visits,
            early_k=early_k,
            top_k=top_k,
            diverse=diverse,
        )

    @staticmethod
    def build_malonyl_chain(n_carbons, terminus="thioester"):
        """
        Build unreduced polyketide backbone with beta-keto groups.
        """
        if n_carbons < 1:
            return None

        if terminus == "thioester":
            chain = "SC(=O)"  # C1
        else:
            chain = "OC(=O)"  # C1 (carboxylic acid)

        for i in range(2, n_carbons + 1):
            if i % 2 == 0:
                chain += "C"
            else:
                chain += "C(=O)"

        return chain


def find_c1(mol):
    """Compatibility wrapper around ``NXBackboneAnalyzer.find_c1``."""
    return NXBackboneAnalyzer.find_c1(mol)


def find_pks_backbone(
    mol,
    c1_idx,
    *,
    exclude_sugar_rings=True,
    max_visits=500000,
    early_k=10,
    include_ring_extensions=False,
    allowed_atom_indices=None,
):
    """Compatibility wrapper around ``NXBackboneAnalyzer.find_pks_backbone``."""
    return NXBackboneAnalyzer.find_pks_backbone(
        mol,
        c1_idx,
        exclude_sugar_rings=exclude_sugar_rings,
        max_visits=max_visits,
        early_k=early_k,
        include_ring_extensions=include_ring_extensions,
        allowed_atom_indices=allowed_atom_indices,
    )


def find_pks_backbone_candidates(
    mol,
    c1_idx,
    *,
    exclude_sugar_rings=True,
    max_visits=500000,
    early_k=10,
    top_k=8,
    allowed_atom_indices=None,
    diverse=False,
):
    """Compatibility wrapper around ``NXBackboneAnalyzer.find_pks_backbone_candidates``."""
    return NXBackboneAnalyzer.find_pks_backbone_candidates(
        mol,
        c1_idx,
        exclude_sugar_rings=exclude_sugar_rings,
        max_visits=max_visits,
        early_k=early_k,
        top_k=top_k,
        allowed_atom_indices=allowed_atom_indices,
        diverse=diverse,
    )


def build_malonyl_chain(n_carbons, terminus="thioester"):
    """Compatibility wrapper around ``NXBackboneAnalyzer.build_malonyl_chain``."""
    return NXBackboneAnalyzer.build_malonyl_chain(n_carbons, terminus=terminus)


def _alpha_beta_indices(backbone):
    """Split backbone indices into alpha/even and beta/odd positions (from C1)."""
    alpha = []
    beta = []
    for pos, idx in enumerate(backbone, start=1):
        if pos == 1:
            continue
        if pos % 2 == 0:
            alpha.append(idx)
        else:
            beta.append(idx)
    return alpha, beta


def _indices_to_str(indices):
    """Serialize atom indices as semicolon-separated text."""
    if not indices:
        return ""
    return ";".join(str(i) for i in indices)


def main():
    ap = argparse.ArgumentParser(
        description="Compute PKS backbone length using NetworkX traversal (core only)."
    )
    ap.add_argument(
        "--input_csv",
        default=str(DATA_DIR / "test_set_linearized.csv"),
        help="Input CSV with linearized_smiles.",
    )
    ap.add_argument(
        "--output_csv",
        default=str(DATA_DIR / "test_set_linearized_networkx.csv"),
        help="Output CSV path (default: data/test_set_linearized_networkx.csv).",
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
        "--include_sugar_rings",
        action="store_true",
        help="Include sugar-like rings in the carbon graph.",
    )
    ap.add_argument(
        "--include_indices",
        action="store_true",
        help="Include backbone atom indices in output.",
    )
    ap.add_argument(
        "--include_alpha_beta",
        action="store_true",
        help="Include alpha/beta atom indices from the core backbone path.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print summary of skipped rows and budget exhaustion.",
    )
    args = ap.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    exclude_sugar_rings = not args.include_sugar_rings

    test_df = pd.read_csv(input_path)
    print(f"Loaded {len(test_df)} BGCs from {input_path}")

    malonyl_chains = []
    backbone_lengths = []
    backbone_indices = []
    alpha_indices = []
    beta_indices = []

    def append_skipped_row():
        malonyl_chains.append(None)
        backbone_lengths.append(None)
        backbone_indices.append(None)
        if args.include_alpha_beta:
            alpha_indices.append(None)
            beta_indices.append(None)

    skip_counts = {
        "missing_smiles": 0,
        "invalid_mol": 0,
        "missing_c1": 0,
        "no_backbone": 0,
        "exhausted_budget": 0,
    }

    for _, row in test_df.iterrows():
        linearized_smiles = str(row["linearized_smiles"])

        if pd.isna(linearized_smiles) or not linearized_smiles or linearized_smiles == "nan":
            skip_counts["missing_smiles"] += 1
            append_skipped_row()
            continue

        mol = Chem.MolFromSmiles(linearized_smiles)
        if mol is None:
            skip_counts["invalid_mol"] += 1
            append_skipped_row()
            continue

        terminus_type, c1_idx = find_c1(mol)
        if c1_idx is None:
            skip_counts["missing_c1"] += 1
            append_skipped_row()
            continue

        backbone, exhausted = find_pks_backbone(
            mol,
            c1_idx,
            exclude_sugar_rings=exclude_sugar_rings,
            max_visits=args.max_visits,
            early_k=args.early_k,
        )
        if exhausted:
            skip_counts["exhausted_budget"] += 1

        if not backbone:
            skip_counts["no_backbone"] += 1
            append_skipped_row()
            continue

        backbone_length = len(backbone)
        backbone_lengths.append(backbone_length)
        backbone_indices.append(";".join(str(i) for i in backbone))

        terminus = "thioester" if terminus_type == "thioester" else "acid"
        malonyl_smiles = build_malonyl_chain(backbone_length, terminus=terminus)
        malonyl_chains.append(malonyl_smiles)

        if args.include_alpha_beta:
            core_backbone, _ = find_pks_backbone(
                mol,
                c1_idx,
                exclude_sugar_rings=exclude_sugar_rings,
                max_visits=args.max_visits,
                early_k=args.early_k,
                include_ring_extensions=False,
            )
            if not core_backbone:
                alpha_indices.append(None)
                beta_indices.append(None)
            else:
                alpha, beta = _alpha_beta_indices(core_backbone)
                alpha_indices.append(_indices_to_str(alpha))
                beta_indices.append(_indices_to_str(beta))

    test_df["backbone_carbon_count_nx"] = backbone_lengths
    test_df["malonyl_chain_smiles_nx"] = malonyl_chains
    if args.include_indices:
        test_df["backbone_indices_nx"] = backbone_indices
    if args.include_alpha_beta:
        test_df["alpha_indices_nx"] = alpha_indices
        test_df["beta_indices_nx"] = beta_indices

    test_df.to_csv(output_path, index=False)
    print(f"\nSaved updated CSV to {output_path}")
    print("Added columns: backbone_carbon_count_nx, malonyl_chain_smiles_nx")
    if args.include_indices:
        print("Added columns: backbone_indices_nx")
    if args.include_alpha_beta:
        print("Added columns: alpha_indices_nx, beta_indices_nx")
    if args.verbose:
        print("\nSkipped rows:")
        for key, val in skip_counts.items():
            print(f"  - {key}: {val}")


if __name__ == "__main__":
    main()
