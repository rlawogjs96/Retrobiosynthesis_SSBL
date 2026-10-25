"""
Create leak-free train/val/test splits for the Reverse-Peeling GNN & SupCon pipeline.

Uses connected-component splitting (Union-Find) over entire trajectory rollout trees
rooted at each nascent PKS core M_0:
- Unions `root_pks_smiles` -> `parent_smiles` -> `smiles` across all trajectory steps M_0, M_1, ..., M_k
  and core-corrupted dead-end negatives M_dead.
- Guarantees zero scaffold or intermediate SMILES leakage across train, val, and test splits.

Input:
  - data/processed/pks_peeling_trajectory_states.parquet (multi-step trajectory states)
  - data/processed/pks_augmentation_pairs.parquet (triplet format)
Output:
  - data/{train,val,test}/peeling_trajectory_{train,val,test}.parquet (full 3-head supervision)
  - data/{train,val,test}/supcon_{train,val,test}.parquet (compatible flat format)

Usage:
    python scripts/03_create_train_val_test_splits.py
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


class UnionFind:
    """Union-Find data structure for connected-component detection."""

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def build_trajectory_components(
    df_states: pd.DataFrame,
    df_triplets: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build connected components across all trajectory steps M_0 -> M_1 -> ... -> M_k
    and dead-end states M_dead so that any shared intermediate or root nascent PKS core
    is assigned to the exact same split.
    """
    uf = UnionFind()

    for _, row in df_states.iterrows():
        root = str(row["root_pks_smiles"])
        parent = str(row["parent_smiles"])
        smi = str(row["smiles"])
        uf.union(root, parent)
        uf.union(root, smi)

    for _, row in df_triplets.iterrows():
        pks = str(row["pks_smiles"])
        uf.union(pks, str(row["enzymatic_aug_smiles"]))
        uf.union(pks, str(row["synthetic_aug_smiles"]))

    df_states = df_states.copy()
    df_states["component_id"] = df_states["root_pks_smiles"].apply(uf.find)

    df_triplets = df_triplets.copy()
    df_triplets["component_id"] = df_triplets["pks_smiles"].apply(uf.find)

    return df_states, df_triplets


def split_components_greedy(
    df: pd.DataFrame,
    target_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Dict[str, str]:
    """
    Assign connected components to splits using greedy bin packing.
    """
    component_sizes = df.groupby("component_id").size().to_dict()
    total = sum(component_sizes.values())

    targets = {
        "train": total * target_ratios[0],
        "val": total * target_ratios[1],
        "test": total * target_ratios[2],
    }
    current = {"train": 0, "val": 0, "test": 0}
    assignments: Dict[str, str] = {}

    for comp_id, size in sorted(component_sizes.items(), key=lambda x: x[1], reverse=True):
        gaps = {k: targets[k] - current[k] for k in targets}
        best_split = max(gaps, key=gaps.get)
        assignments[comp_id] = best_split
        current[best_split] += size

    return assignments


def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """Verify no SMILES overlap across splits."""
    train_smiles = set(train_df["smiles"])
    val_smiles = set(val_df["smiles"])
    test_smiles = set(test_df["smiles"])

    train_val = train_smiles & val_smiles
    train_test = train_smiles & test_smiles
    val_test = val_smiles & test_smiles

    if train_val or train_test or val_test:
        print("LEAKAGE DETECTED:")
        print(f"  Train & Val overlap: {len(train_val)}")
        print(f"  Train & Test overlap: {len(train_test)}")
        print(f"  Val & Test overlap: {len(val_test)}")
        return False
    return True


def main():
    project_root = Path(__file__).resolve().parents[1]
    states_path = project_root / "data" / "processed" / "pks_peeling_trajectory_states.parquet"
    triplets_path = project_root / "data" / "processed" / "pks_augmentation_pairs.parquet"

    train_dir = project_root / "data" / "train"
    val_dir = project_root / "data" / "val"
    test_dir = project_root / "data" / "test"
    stats_path = project_root / "data" / "processed" / "supcon_split_stats.txt"

    print(f"Loading trajectory states from {states_path}")
    df_states = pd.read_parquet(states_path)
    print(f"Loaded {len(df_states):,} trajectory states")

    print(f"Loading triplets from {triplets_path}")
    df_triplets = pd.read_parquet(triplets_path)
    print(f"Loaded {len(df_triplets):,} triplets")

    # Build connected components over all trajectories
    print("Building trajectory connected components...")
    df_states, df_triplets = build_trajectory_components(df_states, df_triplets)

    comp_sizes = df_states.groupby("component_id").size()
    print(f"Found {len(comp_sizes):,} independent trajectory components")
    print(f"  Largest component: {int(comp_sizes.max())} states")
    print(f"  Median component:  {int(comp_sizes.median())} states")

    # Split components (80/10/10)
    assignments = split_components_greedy(df_states)
    df_states["split"] = df_states["component_id"].map(assignments)
    df_triplets["split"] = df_triplets["component_id"].map(assignments)

    # Deduplicate identical SMILES within each split while keeping the lowest distance_to_pks
    df_states = (
        df_states.sort_values(
            by=["split", "step"],
            key=lambda col: col if col.name != "step" else col.replace(-1, 999),
        )
        .drop_duplicates(subset=["split", "smiles"], keep="first")
        .reset_index(drop=True)
    )

    train_df = df_states[df_states["split"] == "train"].drop(columns=["split", "component_id"]).reset_index(drop=True)
    val_df = df_states[df_states["split"] == "val"].drop(columns=["split", "component_id"]).reset_index(drop=True)
    test_df = df_states[df_states["split"] == "test"].drop(columns=["split", "component_id"]).reset_index(drop=True)

    print("\nVerifying zero SMILES leakage across splits...")
    no_leakage = verify_no_leakage(train_df, val_df, test_df)
    if not no_leakage:
        raise RuntimeError("SMILES leakage detected across train/val/test splits!")
    print("  Verified: 0 overlapping SMILES between train, val, and test!")

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Save both the trajectory parquets and `supcon_{train,val,test}.parquet` (with all 3-head columns preserved)
    for split_name, split_dir, df_s in [
        ("train", train_dir, train_df),
        ("val", val_dir, val_df),
        ("test", test_dir, test_df),
    ]:
        traj_out = split_dir / f"peeling_trajectory_{split_name}.parquet"
        supcon_out = split_dir / f"supcon_{split_name}.parquet"
        df_s.to_parquet(traj_out, index=False)
        df_s.to_parquet(supcon_out, index=False)
        pks_ratio = (df_s["label"] == 1).mean() if len(df_s) > 0 else 0.0
        print(
            f"  {split_name.capitalize():5s}: {len(df_s):5,d} states "
            f"(Nascent PKS Core ratio: {100*pks_ratio:.1f}%) -> {supcon_out}"
        )

    with open(stats_path, "w") as f:
        f.write("=== Reverse-Peeling Trajectory Split Statistics ===\n\n")
        f.write(f"Total unique trajectory states: {len(df_states):,}\n")
        f.write(f"Total connected components:     {len(comp_sizes):,}\n")
        f.write(f"Train states:                   {len(train_df):,}\n")
        f.write(f"Val states:                     {len(val_df):,}\n")
        f.write(f"Test states:                    {len(test_df):,}\n")
        f.write(f"No SMILES leakage detected:     {'Yes' if no_leakage else 'FAILED'}\n")

    print(f"\nSaved split statistics to {stats_path}")


if __name__ == "__main__":
    main()
