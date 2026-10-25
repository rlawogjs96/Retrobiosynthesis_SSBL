"""
Tiered Multi-Step Enzymatic Trajectory & Dead-End Generator for Reverse-Peeling GNN

Replaces 1-step stereo-stripped DORAnet augmentation with a 4-phase biologically
constrained post-PKS tailoring simulator powered by SynKit ITS/DPO rules + MITE + Rhea/BKMS
+ Pericyclase (Diels-Alder / spiroketal) reaction templates.

Following `retrobiosynthesis.txt`, this script:
1. Builds a standardized, cofactor-rebalanced, atom-mapped enzymatic Rule Bank (16 canonical
   post-PKS tailoring + pericyclase + Rhea/BKMS classes + parsed MITE entries) validated via
   SynKit's Imaginary Transition State (`rsmi_to_its`) and `SynRule`.
2. Applies a 4-Phase Tiered Biological Grammar to stereo-preserved nascent PKS cores (M_0):
   - Phase 1 (Early Scaffold Cyclization & Pericyclases): Intramolecular [4+2] Diels-Alder
     (SpnF/LovB), intramolecular oxa-Michael tetrahydropyran ring formation, spiroketalization.
   - Phase 2 (Primary C-H Functionalization & Backbone Editing): Cytochrome P450 C-H
     hydroxylation, alkene epoxidation, FADH2/alpha-KG C-H chlorination/bromination, and
     Baeyer-Villiger (BVMO) lactone/ester oxygen insertion.
   - Phase 3 (Secondary Redox & Group Transfer): SAM O-methylation, SAM C-methylation,
     alcohol dehydrogenase oxidation, O-acetylation, epoxide hydrolysis.
   - Phase 4 (Bulky Terminal Decorations): Deoxysugar O-glycosylation, aromatic/enolic
     C-prenylation, O-carbamoylation.
3. Tracks exact atom/bond provenance across all forward steps M_0 -> M_1 -> ... -> M_k:
   - `is_pks_core_atom`: ground-truth mask for Head 1 (Core vs. Decoration Segmentation)
   - `atom_register_role`: ('C1', 'alpha', 'beta', 'starter_backbone', 'alpha_branch',
     'beta_oxygen', 'tailoring_decoration')
   - `bridge_or_cleavable_bonds`: bonds formed by tailoring or cyclization bridges
4. Computes Causal Exposed-Leaf Modifications C_exposed(M_t) for Head 2 (Leaf-Peeling Policy):
   - Only leaf modifications that are NOT capped by downstream non-commuting steps (e.g., an
     O-methylated P450 hydroxyl only exposes O-demethylation first, NOT dehydroxylation!)
     are marked as positive exposed reaction centers.
5. Generates Core-Corrupted Dead-End Negatives (M_dead) for Head 3 (Distance-to-PKS Value Network):
   - Applies reverse tailoring rules to genuine inline PKS functional groups (e.g. stripping
     an inline KR beta-OH or inline methylmalonyl alpha-methyl), verified by
     `deterministic_stage/pks_terminal.py` to break the PKS core register (labeled V(G) = 0, d = inf).
6. Visualizes sample multi-step trajectories and SynKit ITS reaction centers to `plots/`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, Draw
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "SynKit") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "SynKit"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synkit.IO import rsmi_to_its
from synkit.Rule import SynRule
from deterministic_stage import pks_terminal as pt

DEFAULT_INPUT = str(PROJECT_ROOT / "data" / "processed" / "pks_cores_with_stereo_provenance.parquet")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "data" / "processed" / "pks_augmentation_pairs.parquet")
DEFAULT_STATS = str(PROJECT_ROOT / "data" / "processed" / "pks_augmentation_stats.json")
DEFAULT_RULEBANK_JSON = str(PROJECT_ROOT / "data" / "processed" / "synkit_mite_rhea_rulebank.json")
PLOTS_DIR = str(PROJECT_ROOT / "plots")


# =============================================================================
# 1. SYNKIT + MITE + RHEA/BKMS + PERICYCLASE ENZYMATIC RULE BANK
# =============================================================================

@dataclass
class EnzymaticTailoringRule:
    """A biologically constrained enzymatic rule with forward/reverse templates and SynKit ITS metadata."""
    rule_id: int
    name: str
    phase: int  # 1: Cyclization/Pericyclase, 2: Primary C-H/Backbone, 3: Secondary Capping/Redox, 4: Terminal Bulky
    enzyme_family: str  # e.g., 'Pericyclase', 'P450', 'BVMO', 'Halogenase', 'Methyltransferase', 'Glycosyltransferase', 'Prenyltransferase', 'Acyltransferase', 'Oxidoreductase', 'Hydrolase'
    ec_class: str
    source_db: str  # 'MITE', 'Rhea/BKMS', 'MITE+Rhea'
    forward_smarts: str
    reverse_smarts: str
    balanced_its_rsmi: str  # Cofactor-balanced atom-mapped reaction SMILES for SynKit ITS construction
    is_bridge_forming: bool  # True for Diels-Alder, oxa-Michael, spiroketal, BVMO (edits topology without simple pendant removal)
    caps_parent_oxygen: bool  # True if this step caps a hydroxyl oxygen (O-MT, O-GT, O-Acyl, O-Carbamoyl, Alcohol Ox)
    candidate_enzymes: str = ""  # Concrete characterized enzymes catalyzing this transformation
    uniprot_ids: str = ""        # Representative UniProt / MITE accessions
    pfam_domains: str = ""       # Pfam / InterPro HMM signatures for BGC gene matching
    cofactors: str = ""          # Required cofactors / cosubstrates
    synkit_rc_nodes: int = 0
    synkit_rc_edges: int = 0


ENZYME_CATALOG_BY_RULE_ID: Dict[int, Dict[str, str]] = {
    0: {
        "candidate_enzymes": "SpnF (Spinosad [4+2] Diels-Alderase), LovB/LovC (Lovastatin nonaketide Diels-Alderase), VstJ",
        "uniprot_ids": "Q9AL95 (SpnF), Q9Y8A5 (LovB)",
        "pfam_domains": "PF08241 (Methyltransf_11-like pericyclase fold), PF12697",
        "cofactors": "None (SAM-structural or cofactor-independent pericyclase)",
    },
    1: {
        "candidate_enzymes": "AmbDH3 (Ambruticin THP pyran cyclase), PedI/PksJ PS-DH domain, SalBIII (Salinomycin)",
        "uniprot_ids": "Q108P4 (AmbDH3), A0A0F7V0Y5 (SalBIII)",
        "pfam_domains": "PF14765 (PS-DH / Pyran-synthase dehydratase-like domain)",
        "cofactors": "None (His/Asp general acid-base catalysis)",
    },
    2: {
        "candidate_enzymes": "RevJ (Reveromycin spiroketalase), AveC (Avermectin spiroketalase), MonBI/MonBII (Monensin)",
        "uniprot_ids": "G4V4H2 (RevJ), Q9S0R6 (AveC), Q846W9 (MonBI)",
        "pfam_domains": "PF12680 (SnoaL_2 / Spiroketalase / Epoxide cyclase)",
        "cofactors": "None",
    },
    3: {
        "candidate_enzymes": "EryF / CYP107A1 (6-DEB 6S-hydroxylase), EryK (C12-hydroxylase), PikC / CYP107L1 (Pikromycin), TylI",
        "uniprot_ids": "Q00441 (EryF), P93703 (EryK), O87605 (PikC)",
        "pfam_domains": "PF00067 (Cytochrome P450)",
        "cofactors": "Heme, O2, Ferredoxin / Ferredoxin Reductase, NAD(P)H",
    },
    4: {
        "candidate_enzymes": "AbyE (Abyssomicin C11,C12-epoxidase), MonCI (Monensin flavin epoxidase), EpoK (Epothilone), PimD (Pimaricin)",
        "uniprot_ids": "F4F7G8 (AbyE), Q846W8 (MonCI), Q9KIZ4 (EpoK)",
        "pfam_domains": "PF00067 (Cytochrome P450) or PF01494 (FAD_binding_3 epoxidase)",
        "cofactors": "Heme or FAD, O2, NAD(P)H",
    },
    5: {
        "candidate_enzymes": "MtmOIV (Mithramycin BVMO), PtlE (Pentalenolactone BVMO), CcsB (Cytochalasin BVMO)",
        "uniprot_ids": "Q194P4 (MtmOIV), Q82IY8 (PtlE), K0E4D4 (CcsB)",
        "pfam_domains": "PF00743 (FMO-like) / PF13738 (Pyr_redox_3 Baeyer-Villiger Monooxygenase)",
        "cofactors": "FAD, O2, NADPH",
    },
    6: {
        "candidate_enzymes": "SyrB2 / CytC3 (Non-heme Fe(II)/2-OG aliphatic halogenase), WelO5, BesD",
        "uniprot_ids": "Q9FB02 (SyrB2), Q194R8 (CytC3)",
        "pfam_domains": "PF05721 (PhyH / Fe(II)-alpha-ketoglutarate halogenase)",
        "cofactors": "Fe(II), 2-oxoglutarate, Cl-, O2",
    },
    7: {
        "candidate_enzymes": "RadH (Radicicol FADH2 halogenase), PltA (Pyoluteorin), ChlB4 (Chlorothricin)",
        "uniprot_ids": "C5H885 (RadH), Q4K423 (PltA), Q0R4N2 (ChlB4)",
        "pfam_domains": "PF04820 (Trp_halogenase / Flavin-dependent halogenase)",
        "cofactors": "FADH2, Br-/Cl-, O2",
    },
    8: {
        "candidate_enzymes": "AviG2 (Avilamycin O-MT, MITE0000001), EryG (Erythromycin 3''-O-MT), MonE (Monensin C3-O-MT), TylF",
        "uniprot_ids": "Q93KW1 (AviG2 / MITE0000001), P33996 (EryG), Q846W4 (MonE)",
        "pfam_domains": "PF08241 (Methyltransf_11) / PF00891 (Methyltransf_2)",
        "cofactors": "S-adenosyl-L-methionine (SAM)",
    },
    9: {
        "candidate_enzymes": "NovO / CouO (C-methyltransferase), MtmMII (Mithramycin C9-MT), CurJ cMT",
        "uniprot_ids": "Q9F8U1 (NovO), Q9LCX3 (MtmMII)",
        "pfam_domains": "PF08242 (Methyltransf_12) / PF02353 (CMAS)",
        "cofactors": "S-adenosyl-L-methionine (SAM)",
    },
    10: {
        "candidate_enzymes": "AveF (Avermectin C5-ketoreductase/dehydrogenase), LanV, ActIII",
        "uniprot_ids": "Q79ZM6 (AveF), P16544 (ActIII)",
        "pfam_domains": "PF00106 (adh_short / SDR oxidoreductase)",
        "cofactors": "NAD(P)+ / NAD(P)H",
    },
    11: {
        "candidate_enzymes": "MdmB (Midecamycin 3-O-acyltransferase), AcyA (Carbomycin O-AT), LovD",
        "uniprot_ids": "Q00719 (MdmB), Q9Y7D1 (LovD)",
        "pfam_domains": "PF02458 (BAHD acyltransferase) / PF00698 (Acyl_transf_1)",
        "cofactors": "Acetyl-CoA / Acyl-ACP",
    },
    12: {
        "candidate_enzymes": "EpoF (Epothilone epoxide hydrolase), MonBI/MonBII (Monensin epoxide hydrolase), NanI",
        "uniprot_ids": "Q846W9 (MonBI), Q846W7 (MonBII)",
        "pfam_domains": "PF00561 (Abhydrolase_1) / PF12680 (SnoaL_2 epoxide hydrolase)",
        "cofactors": "H2O",
    },
    13: {
        "candidate_enzymes": "DesVII/DesVIII (Pikromycin D-desosaminyltransferase), EryBV (L-mycarosyltransferase), EryCIII",
        "uniprot_ids": "O87603 (DesVII), O33939 (EryBV), A4F7P3 (EryCIII)",
        "pfam_domains": "PF00201 (UDPGT / GT1 Macrolide Glycosyltransferase)",
        "cofactors": "dTDP-/UDP-6-deoxysugar",
    },
    14: {
        "candidate_enzymes": "NphB / CloQ (ABBA aromatic/polyketide prenyltransferase), Fura2, XanB2",
        "uniprot_ids": "Q4WYG3 (NphB), Q9F8T9 (CloQ)",
        "pfam_domains": "PF11991 (Trp_DMAT / ABBA prenyltransferase)",
        "cofactors": "Dimethylallyl diphosphate (DMAPP), Mg2+",
    },
    15: {
        "candidate_enzymes": "Asm21 (Ansamitocin O-carbamoyltransferase), NovN (Novobiocin), ConI (Concanamycin)",
        "uniprot_ids": "Q846Y7 (Asm21), Q9L9F5 (NovN)",
        "pfam_domains": "PF02543 (CmcH_NodU / O-Carbamoyltransferase)",
        "cofactors": "Carbamoyl phosphate, ATP",
    },
    16: {
        "candidate_enzymes": "AbyA1 + AbyA2 + AbyA3 (Abyssomicin FabH-like glyceryl-ACP synthase + tetronate cyclase/eliminase), ChlM/ChlD1-D4 (Chlorothricin), RkD/RkE",
        "uniprot_ids": "F4F7G2 (AbyA1), F4F7G3 (AbyA2), F4F7G4 (AbyA3)",
        "pfam_domains": "PF08545 (ACP_syn_III / FabH-like tetronate synthase) + PF00561 (Hydrolase) + PF00501",
        "cofactors": "D-1,3-bisphosphoglycerate, Holo-ACP (AbyA3), Acetyl-CoA",
    },
    17: {
        "candidate_enzymes": "AbyU (Abyssomicin spirotetronate [4+2] Diels-Alderase, PDB: 5DYV), ChlJ (Chlorothricin), PyrI4 (Pyrroindomycin), VstJ",
        "uniprot_ids": "F4F7H5 (AbyU), Q0R4M6 (ChlJ), A0A089QRB9 (PyrI4)",
        "pfam_domains": "PF13577 (SnoaL_4 / Beta-barrel spirotetronate Diels-Alderase)",
        "cofactors": "None (Cofactor-independent beta-barrel pericyclase)",
    },
    18: {
        "candidate_enzymes": "AbyZ / AbyA4 (Abyssomicin epoxide-opening ether-bridge cyclase), MonBI/MonBII (Monensin polyether cyclase), Lsd19 (Lasalocid)",
        "uniprot_ids": "F4F7G5 (AbyA4), F4F7I0 (AbyZ), Q846W9 (MonBI), B6ZK66 (Lsd19)",
        "pfam_domains": "PF12680 (SnoaL_2 / Epoxide-opening ether cyclase) or PF00561",
        "cofactors": "None (Brønsted acid-base epoxide activation)",
    },
    19: {
        "candidate_enzymes": "MonBI + MonBII (Monensin polyether epoxide-opening zipper cyclase), Lsd19 (Lasalocid A THF/THP cyclase), SalBIII, NigBI",
        "uniprot_ids": "Q846W9 (MonBI), Q846W7 (MonBII), B6ZK66 (Lsd19)",
        "pfam_domains": "PF12680 (SnoaL_2 polyether epoxide cyclase)",
        "cofactors": "None (Acid-base epoxide-opening zipper cascade)",
    },
    20: {
        "candidate_enzymes": "RapP / FkbP / MerP (PKS-to-NRPS C-A-PCP condensation-adenylation module), TmcA (Tirandamycin), Chondrochloren CndF",
        "uniprot_ids": "P60208 (RapP), Q9KID9 (FkbP), D6B6R8 (TmcA)",
        "pfam_domains": "PF00668 (Condensation domain) + PF00501 (AMP-binding A domain) + PF00550 (PP-binding PCP)",
        "cofactors": "ATP, Mg2+, L-Amino Acid / L-Pipecolate, 4'-Phosphopantetheine",
    },
    21: {
        "candidate_enzymes": "EpoB (Epothilone NRPS heterocyclization Cy + FMN oxidase Ox), CtaB (Cystothiazole), BlmIII (Bleomycin)",
        "uniprot_ids": "Q9KIZ9 (EpoB), Q9F635 (CtaB), Q9FB18 (BlmIII)",
        "pfam_domains": "PF00668 (Cy Heterocyclization domain) + PF00881 (Nitroreductase / FMN Oxidase)",
        "cofactors": "FMN, O2, ATP",
    },
    22: {
        "candidate_enzymes": "TmcA / LipPks1 / IkaA (Dieckmann R0 / TE tetramate offloading cyclase), HSAF PKS-NRPS",
        "uniprot_ids": "D6B6R8 (TmcA), G9M9B6 (IkaA)",
        "pfam_domains": "PF07993 (Male_sterile / Dieckmann R0 reductase-like cyclase) or PF00975 (Thioesterase)",
        "cofactors": "None (Intramolecular Dieckmann C-acylation offloading)",
    },
    23: {
        "candidate_enzymes": "Inline NRPS nMT domain (Cyclosporin/Anabaenopeptin N-MT),RapM, DesVI (N,N-dimethyltransferase)",
        "uniprot_ids": "Q09164 (SimA nMT), O87602 (DesVI)",
        "pfam_domains": "PF08241 (Methyltransf_11) / PF13649 (Methyltransf_25)",
        "cofactors": "S-adenosyl-L-methionine (SAM)",
    },
    24: {
        "candidate_enzymes": "RapP (Rapamycin L-pipecolate NRPS C-A-PCP module), FkbP (FK520/Tacrolimus), MerP (Meridamycin)",
        "uniprot_ids": "P60208 (RapP), Q9KID9 (FkbP)",
        "pfam_domains": "PF00668 (Condensation) + PF00501 (AMP-binding) + PF00550 (PP-binding)",
        "cofactors": "ATP, Mg2+, L-Proline / L-Pipecolate",
    },
    25: {
        "candidate_enzymes": "OzmP / VirG / MgsH (NRPS Oxazole Cy + FMN Oxidase module), PksJ (Bacillaene), DisD (Disorazol)",
        "uniprot_ids": "Q1EMV2 (OzmP), Q9R9J1 (PksJ)",
        "pfam_domains": "PF00668 (Cy Heterocyclization) + PF00881 (Nitroreductase FMN Ox)",
        "cofactors": "FMN, O2, L-Serine / L-Threonine",
    },
    26: {
        "candidate_enzymes": "CurM-ST (Curacin A PAPS sulfotransferase), StaL (Glycopeptide sulfotransferase), SfaH",
        "uniprot_ids": "F4Y4B5 (CurM), Q8KND8 (StaL)",
        "pfam_domains": "PF00685 (Sulfotransfer_1)",
        "cofactors": "3'-Phosphoadenosine-5'-phosphosulfate (PAPS)",
    },
    27: {
        "candidate_enzymes": "KirAVI / KirB (Kirromycin beta-alanine pyridone Dieckmann cyclase), TenS / DesS (2-Pyridone synthase)",
        "uniprot_ids": "C7C460 (KirB), A0A089QRC1",
        "pfam_domains": "PF00668 (Condensation) + PF07993 (R0 Dieckmann cyclase)",
        "cofactors": "beta-Alanine / L-Amino Acid, ATP",
    },
    28: {
        "candidate_enzymes": "AviO1 / AviB1 (Avilamycin Fe(II)/2-OG cyclic acetal/orthoester synthase), HygX (Hygromycin dioxolane synthase)",
        "uniprot_ids": "Q93KW8 (AviO1), Q54390 (HygX)",
        "pfam_domains": "PF03171 (2OG-FeII_Oxy)",
        "cofactors": "Fe(II), 2-oxoglutarate, O2",
    },
    29: {
        "candidate_enzymes": "BorI + BorJ + BorK (Borrelidin P450 + transaminase C12-nitrile synthase), SaqC (Saframycin)",
        "uniprot_ids": "Q65A03 (BorI), Q65A02 (BorJ)",
        "pfam_domains": "PF00067 (Cytochrome P450 BorI) + PF00155 (Aminotran_1_2 BorJ)",
        "cofactors": "Heme, O2, PLP, NAD(P)H",
    },
    30: {
        "candidate_enzymes": "PimG (Natamycin/Pimaricin C12-methyl carboxylating P450), NysN (Nystatin C16-methyl oxidase), AmphN (Amphotericin B)",
        "uniprot_ids": "Q9EW92 (PimG), Q9L4W3 (NysN), Q93NX5 (AmphN)",
        "pfam_domains": "PF00067 (Cytochrome P450 6-electron methyl-to-carboxyl oxidase)",
        "cofactors": "Heme, 3 O2, Ferredoxin / Ferredoxin Reductase, 3 NAD(P)H",
    },
    31: {
        "candidate_enzymes": "GdmM (Geldanamycin/Herbimycin/Macbecin AHBA C21-quinone flavin monooxygenase) + GdmF (Ansamycin amide synthase)",
        "uniprot_ids": "Q846X9 (GdmM), Q846Y4 (GdmF), O52552 (RifF)",
        "pfam_domains": "PF01494 (FAD_binding_3 quinone monooxygenase) + PF01884 (Arylamine amide synthase)",
        "cofactors": "FAD, O2, NADPH",
    },
    32: {
        "candidate_enzymes": "MonBI + MonBII (Monensin polyether THF/THP cyclase), Lsd19 (Lasalocid A), SalBIII (Salinomycin), NigBI",
        "uniprot_ids": "Q846W9 (MonBI), B6ZK66 (Lsd19), A0A0F7V0Y5 (SalBIII)",
        "pfam_domains": "PF12680 (SnoaL_2 polyether cyclase) / PF14765 (PS-DH)",
        "cofactors": "None (Acid-base oxa-Michael / epoxide-opening polyether cyclization)",
    },
}


# Complete 68-entry RetroMol rxn.yml -> Characterized Enzyme Catalog
RETROMOL_68_RXN_ENZYME_CATALOG: List[Dict[str, str]] = [
    {"rxn_name": "break ester bond (intermolecular)", "enzymes": "MdmB / LovD / PikAIV TE (Intermolecular esterase / O-acyltransferase)", "uniprot": "Q00719, Q9Y7D1", "pfam": "PF00975 / PF02458", "ec": "EC 2.3.1.- / 3.1.1.-"},
    {"rxn_name": "break thio-ester bond (intermolecular)", "enzymes": "PKS/NRPS Thioesterase (TEI / TEII) or Transthioesterification AT", "uniprot": "Q03133 (DEBS3 TE)", "pfam": "PF00975 (Thioesterase)", "ec": "EC 3.1.2.-"},
    {"rxn_name": "break N-O glycosidic bond (intermolecular)", "enzymes": "CalG1 / CalG4 (Calicheamicin hydroxylamine N-O glycosyltransferase, PDB: 3IAA)", "uniprot": "Q8KND5 (CalG1)", "pfam": "PF00201 (UDPGT)", "ec": "EC 2.4.1.-"},
    {"rxn_name": "break glycosidic bond", "enzymes": "DesVII/DesVIII (Pikromycin), EryBV/EryCIII (Erythromycin), OleG1/G2", "uniprot": "O87603, O33939", "pfam": "PF00201 (UDPGT / GT1)", "ec": "EC 2.4.1.-"},
    {"rxn_name": "break furanosidic bond", "enzymes": "ChlC6 (Chlorothricin), MtmGIII/GIV (Mithramycin furanosyltransferase)", "uniprot": "Q0R4N9 (ChlC6)", "pfam": "PF00201 / PF00535", "ec": "EC 2.4.1.-"},
    {"rxn_name": "break glycosidic ester bond", "enzymes": "NysD1 (Nystatin), AmphDI (Amphotericin carboxyl glycosyltransferase)", "uniprot": "Q9L4W5", "pfam": "PF00201 (UDPGT)", "ec": "EC 2.4.1.-"},
    {"rxn_name": "break N-glycosidic linkage", "enzymes": "StaG (Staurosporine N-glycosyltransferase), RebG (Rebeccamycin)", "uniprot": "Q83WG3 (StaG), Q8KHE4 (RebG)", "pfam": "PF00201 (UDPGT)", "ec": "EC 2.4.1.-"},
    {"rxn_name": "reverse epoxidation", "enzymes": "AbyE (Abyssomicin P450), EpoK (Epothilone P450), MonCI (Monensin FAD epoxidase)", "uniprot": "F4F7G8, Q9KIZ4, Q846W8", "pfam": "PF00067 (P450) / PF01494 (FAD_binding_3)", "ec": "EC 1.14.14.- / 1.14.13.-"},
    {"rxn_name": "reverse O-methylation", "enzymes": "AviG2 (MITE0000001), EryG (Erythromycin O-MT), MonE (Monensin O-MT)", "uniprot": "Q93KW1, P33996, Q846W4", "pfam": "PF08241 (Methyltransf_11)", "ec": "EC 2.1.1.-"},
    {"rxn_name": "reverse N-methylation", "enzymes": "Inline NRPS nMT domain (SimA, Cyclosporin), DesVI (Desosamine N-MT)", "uniprot": "Q09164, O87602", "pfam": "PF08241 (Methyltransf_11)", "ec": "EC 2.1.1.-"},
    {"rxn_name": "reverse S-methylation", "enzymes": "GliN (Gliotoxin bis-thiomethyltransferase), TioN", "uniprot": "Q4WMJ7 (GliN)", "pfam": "PF08241 (Methyltransf_11)", "ec": "EC 2.1.1.-"},
    {"rxn_name": "reverse C-methylation amino acid", "enzymes": "MrsA / TsrM (Radical SAM cobalamin-dependent amino acid C-MT)", "uniprot": "C0HJR5 (TsrM)", "pfam": "PF04055 (Radical_SAM) + PF02310 (B12-binding)", "ec": "EC 2.1.1.-"},
    {"rxn_name": "reverse halogenation", "enzymes": "RadH / ChlB4 (FADH2 halogenase) & SyrB2 / CytC3 (Fe/2-OG aliphatic halogenase)", "uniprot": "C5H885, Q9FB02", "pfam": "PF04820 (Trp_halogenase) / PF05721 (PhyH)", "ec": "EC 1.14.19.- / 1.14.11.-"},
    {"rxn_name": "reverse carbamic acid tailoring (type 1)", "enzymes": "Asm21 (Ansamitocin O-carbamoyltransferase), NovN (Novobiocin), CmcH", "uniprot": "Q846Y7, Q9L9F5", "pfam": "PF02543 (CmcH_NodU)", "ec": "EC 2.1.3.-"},
    {"rxn_name": "reverse carbamic acid tailoring (type 2)", "enzymes": "BlmD (Bleomycin N-carbamoyltransferase), TobZ", "uniprot": "Q9FB15 (BlmD)", "pfam": "PF02543 (CmcH_NodU)", "ec": "EC 2.1.3.-"},
    {"rxn_name": "reverse carbinolamide tailoring", "enzymes": "GdmN / Asm21 (Maytansinoid/Ansamitocin cyclic carbinolamide synthase, PMID:12060743)", "uniprot": "Q846Y7 (Asm21)", "pfam": "PF02543 (CmcH_NodU)", "ec": "EC 2.1.3.-"},
    {"rxn_name": "reverse acetic acid tailoring", "enzymes": "MdmB (Midecamycin 3-O-acyltransferase), AcyA, LovD", "uniprot": "Q00719 (MdmB)", "pfam": "PF02458 (BAHD Transferase)", "ec": "EC 2.3.1.-"},
    {"rxn_name": "reverse carboxylic acid tailoring", "enzymes": "NRPS/PKS N-Acyltransferase", "uniprot": "Q00719", "pfam": "PF02458", "ec": "EC 2.3.1.-"},
    {"rxn_name": "reverse acylation (type 1)", "enzymes": "MdmB / LovD (O/N-acetyltransferase)", "uniprot": "Q00719, Q9Y7D1", "pfam": "PF02458", "ec": "EC 2.3.1.-"},
    {"rxn_name": "reverse acylation (type 2)", "enzymes": "AcyA / CarE (Isobutyryl/Isovaleryl O-acyltransferase)", "uniprot": "Q54297 (CarE)", "pfam": "PF02458", "ec": "EC 2.3.1.-"},
    {"rxn_name": "reverse acylation (type 3)", "enzymes": "N-Alkyltransferase / Reductive Aminase", "uniprot": "Q9L9F5", "pfam": "PF08241", "ec": "EC 2.5.1.-"},
    {"rxn_name": "reverse cyanide tailoring", "enzymes": "BorI + BorJ + BorK (Borrelidin P450 + transaminase nitrile synthase), SaqC", "uniprot": "Q65A03 (BorI)", "pfam": "PF00067 (P450) + PF00155 (Aminotran_1_2)", "ec": "EC 1.14.99.-"},
    {"rxn_name": "reverse hydroxyl sulfonation", "enzymes": "CurM-ST (Curacin A sulfotransferase), StaL (A47934 sulfotransferase)", "uniprot": "F4Y4B5, Q8KND8", "pfam": "PF00685 (Sulfotransfer_1)", "ec": "EC 2.8.2.-"},
    {"rxn_name": "reverse hydroxyl phosphonylation", "enzymes": "MphA / MphB (Macrolide 2'-phosphotransferase), Ser/Thr kinase", "uniprot": "Q02579 (MphA)", "pfam": "PF01636 (APH)", "ec": "EC 2.7.1.-"},
    {"rxn_name": "reverse 1,3-dioxane formation (cyclic acetal)", "enzymes": "AviO1 (Avilamycin Fe/2-OG cyclic acetal/orthoester synthase), BryA", "uniprot": "Q93KW8 (AviO1)", "pfam": "PF03171 (2OG-FeII_Oxy)", "ec": "EC 1.14.11.-"},
    {"rxn_name": "reverse 1,3-dioxolane formation (spiro-bis(pyranose) system)", "enzymes": "AviB1 / AviO2 (Avilamycin/Everninomicin orthoester synthase), HygX", "uniprot": "Q93KW8", "pfam": "PF03171 (2OG-FeII_Oxy)", "ec": "EC 1.14.11.-"},
    {"rxn_name": "reverse 2,3-diaminopropionate amidation", "enzymes": "BlmIII/IV (Bleomycin Dap amidase/condensation), SbnA/SbnB", "uniprot": "Q9FB18", "pfam": "PF00668 / PF00733 (Asn_synthase)", "ec": "EC 6.3.5.-"},
    {"rxn_name": "reverse hydrogenation tailoring", "enzymes": "MxaA-R / Lys2-R (Terminal 4-electron thioester reductase TD/R domain)", "uniprot": "Q93NW7 (MxaA)", "pfam": "PF07993 (NAD_binding_4 / Terminal Reductase)", "ec": "EC 1.2.1.-"},
    {"rxn_name": "reverse etherification", "enzymes": "MonBI + MonBII (Monensin polyether epoxide-opening cyclase), Lsd19 (Lasalocid)", "uniprot": "Q846W9, B6ZK66", "pfam": "PF12680 (SnoaL_2)", "ec": "EC 3.3.2.-"},
    {"rxn_name": "reverse terminal etherification", "enzymes": "SalBIII (Salinomycin pyran cyclase), RevJ (Reveromycin)", "uniprot": "A0A0F7V0Y5, G4V4H2", "pfam": "PF12680 (SnoaL_2)", "ec": "EC 4.2.1.-"},
    {"rxn_name": "reverse etherification (6-ring; no hydroxyl group)", "enzymes": "AmbDH3 (Ambruticin THP-forming pyran synthase PS-DH), PedI PS", "uniprot": "Q108P4 (AmbDH3)", "pfam": "PF14765 (PS-DH)", "ec": "EC 5.5.1.-"},
    {"rxn_name": "reverse etherification (5-ring; no hydroxyl group)", "enzymes": "PolG /NysN / MonBI (5-exo-tet THF oxa-Michael/epoxide cyclase)", "uniprot": "Q846W9", "pfam": "PF12680 / PF14765", "ec": "EC 5.5.1.-"},
    {"rxn_name": "reverse macrolactonization", "enzymes": "DEBS3 TE / PikAIV TE (Type I PKS Macrolactone Thioesterase)", "uniprot": "Q03133, Q9ZGI2", "pfam": "PF00975 (Thioesterase)", "ec": "EC 3.1.2.-"},
    {"rxn_name": "break aromatic ester bond (intramolecular)", "enzymes": "Isocoumarin / Resorcylic Acid Lactone (RAL) TE (Radicicol Rdc1 TE, Zearalenone)", "uniprot": "B3FWT6 (Rdc1)", "pfam": "PF00975 (Thioesterase)", "ec": "EC 3.1.2.-"},
    {"rxn_name": "reverse macrolactonethionization", "enzymes": "Thiocoraline / Echinomycin Thiolactone TE Domain (TioS TE)", "uniprot": "Q2A784 (TioS)", "pfam": "PF00975 (Thioesterase)", "ec": "EC 3.1.2.-"},
    {"rxn_name": "reverse carbocyclization (type 1)", "enzymes": "SpnF / SpnL (Spinosad [4+2] / Rauhut-Currier carbocyclases), LovB", "uniprot": "Q9AL95 (SpnF), Q9AL90 (SpnL)", "pfam": "PF08241 (SpnF-like)", "ec": "EC 5.5.1.26"},
    {"rxn_name": "reverse carbocyclization (type 2)", "enzymes": "AbyU / ChlJ / PyrI4 (Conjugated polyene carbocyclase / Diels-Alderase)", "uniprot": "F4F7H5, Q0R4M6", "pfam": "PF13577 (SnoaL_4)", "ec": "EC 5.5.1.-"},
    {"rxn_name": "reverse carbocycle oxidation", "enzymes": "LovA / MtmOIV (Cytochrome P450 / BVMO allylic carbocycle oxidase)", "uniprot": "Q9Y7C8 (LovA)", "pfam": "PF00067 (P450)", "ec": "EC 1.14.14.-"},
    {"rxn_name": "reverse heterocyclization", "enzymes": "TmcA / IkaA (Pyrrolidine-2,4-dione Dieckmann cyclase)", "uniprot": "D6B6R8 (TmcA)", "pfam": "PF07993 / PF00975", "ec": "EC 2.3.1.-"},
    {"rxn_name": "break dilsufide bridge (intramolecular)", "enzymes": "GliT (Gliotoxin disulfide oxidase), DepH (Romidepsin/FK228 FAD disulfide oxidase)", "uniprot": "Q4WMJ5 (GliT), B1B533 (DepH)", "pfam": "PF07992 (Pyr_redox_2)", "ec": "EC 1.8.3.-"},
    {"rxn_name": "break disulfide bridge (intermolecular)", "enzymes": "GliT / HolH (Epidithiodiketopiperazine / Pyrrothine disulfide oxidase)", "uniprot": "Q4WMJ5", "pfam": "PF07992", "ec": "EC 1.8.3.-"},
    {"rxn_name": "break threonine-cysteine bridge (intramolecular)", "enzymes": "NisC / LanM (Methyl-lanthionine cyclase)", "uniprot": "P29559 (NisC)", "pfam": "PF05147 (LANC_like)", "ec": "EC 4.4.1.-"},
    {"rxn_name": "break threonine-cysteine bridge (intermolecular)", "enzymes": "LanC-like Thioether Lyase", "uniprot": "P29559", "pfam": "PF05147", "ec": "EC 4.4.1.-"},
    {"rxn_name": "break serine-cysteine bridge (intramolecular)", "enzymes": "NisC / SpaC (Lanthionine dehydratase-cyclase)", "uniprot": "P29559", "pfam": "PF05147 (LANC_like)", "ec": "EC 4.4.1.-"},
    {"rxn_name": "break serine-cysteine bridge (intermolecular)", "enzymes": "LanC-like Thioether Synthase", "uniprot": "P29559", "pfam": "PF05147", "ec": "EC 4.4.1.-"},
    {"rxn_name": "break cysteine bond (intramolecular)", "enzymes": "AlbA (Subtilosin radical SAM sactionine C-S bond synthase)", "uniprot": "P71011 (AlbA)", "pfam": "PF04055 (Radical_SAM)", "ec": "EC 1.97.1.-"},
    {"rxn_name": "reverse reduction (type 1)", "enzymes": "SyrP / VioC (Non-heme Fe(II)/2-OG amino acid beta/alpha-hydroxylase)", "uniprot": "Q9FB00 (SyrP), Q6WZB0 (VioC)", "pfam": "PF05721 (PhyH) / PF02668 (TauD)", "ec": "EC 1.14.11.-"},
    {"rxn_name": "reverse reduction (type 2)", "enzymes": "AsnO / CdaPS (Asparagine/Glutamate Fe(II)/2-OG hydroxylase)", "uniprot": "O31428 (AsnO)", "pfam": "PF02668 (TauD)", "ec": "EC 1.14.11.-"},
    {"rxn_name": "reverse oxazole formation", "enzymes": "OzmP / PksJ / DisD (NRPS Heterocyclization Cy + FMN Oxidase Ox domain)", "uniprot": "Q1EMV2 (OzmP), Q9R9J1 (PksJ)", "pfam": "PF00668 (Cy) + PF00881 (Ox)", "ec": "EC 4.2.1.- / 1.3.99.-"},
    {"rxn_name": "reverse oxazoline formation", "enzymes": "VibF / EntF Cy (NRPS Heterocyclization Cy domain without Ox)", "uniprot": "O52820 (VibF)", "pfam": "PF00668 (Cy Heterocyclization)", "ec": "EC 4.2.1.-"},
    {"rxn_name": "reverse thiazole formation", "enzymes": "EpoB (Epothilone Cy-A-Ox-PCP), CtaB (Cystothiazole), BlmIII (Bleomycin)", "uniprot": "Q9KIZ9 (EpoB), Q9F635 (CtaB)", "pfam": "PF00668 (Cy) + PF00881 (Ox)", "ec": "EC 4.2.1.- / 1.3.99.-"},
    {"rxn_name": "reverse thiazoline tautomerization", "enzymes": "YbtE / HMWP2 (Yersiniabactin thiazoline cyclase), PchE (Pyochelin)", "uniprot": "P48633 (HMWP2)", "pfam": "PF00668 (Cy)", "ec": "EC 4.2.1.-"},
    {"rxn_name": "reverse thiazoline formation (type 1)", "enzymes": "HMWP2 / BacA (NRPS Cysteine Heterocyclization Cy domain)", "uniprot": "P48633, O68006", "pfam": "PF00668 (Cy)", "ec": "EC 4.2.1.-"},
    {"rxn_name": "reverse thiazoline formation (type 2)", "enzymes": "CurA / AprA (Hybrid PKS-NRPS Cysteine Cy domain)", "uniprot": "F4Y4A4 (CurA)", "pfam": "PF00668 (Cy)", "ec": "EC 4.2.1.-"},
    {"rxn_name": "reverse tetramate formation (type 1)", "enzymes": "TmcA (Tirandamycin Dieckmann cyclase), IkaA (Ikarugamycin), LipNrps", "uniprot": "D6B6R8 (TmcA), G9M9B6 (IkaA)", "pfam": "PF07993 (R0 Dieckmann cyclase) / PF00975 (TE)", "ec": "EC 2.3.1.-"},
    {"rxn_name": "reverse tetramate formation (type 2)", "enzymes": "HSAF PKS-NRPS / CpaS (Cyclopiazonic acid Dieckmann R0 domain)", "uniprot": "B8N0E8 (CpaS)", "pfam": "PF07993 (R0 Dieckmann cyclase)", "ec": "EC 2.3.1.-"},
    {"rxn_name": "reverse tetramate formation (type 3; aromatic)", "enzymes": "FtdB /EqxS (Aromatized tetramate PKS-NRPS Dieckmann cyclase)", "uniprot": "S0EET5 (EqxS)", "pfam": "PF07993", "ec": "EC 2.3.1.-"},
    {"rxn_name": "reverse tetronate formation (type 1)", "enzymes": "RkD + RkE (RK-682 FabH-like glyceryl-ACP tetronate synthase), Agg4/Agg5", "uniprot": "D3Y1I2 (RkD)", "pfam": "PF08545 (ACP_syn_III)", "ec": "EC 2.3.1.-"},
    {"rxn_name": "reverse tetronate formation (type 2)", "enzymes": "AbyA1–A3 (Abyssomicin), ChlM/ChlD1–D4 (Chlorothricin), KijA–E (Kijanimicin)", "uniprot": "F4F7G2 (AbyA1)", "pfam": "PF08545 (ACP_syn_III) + PF00561", "ec": "EC 2.3.1.-"},
    {"rxn_name": "reverse tetronate formation (type 3; spirotetronate)", "enzymes": "AbyA1–A3 + AbyU (Spirotetronate synthase + Diels-Alderase)", "uniprot": "F4F7G2, F4F7H5", "pfam": "PF08545 + PF13577", "ec": "EC 2.3.1.- / 5.5.1.-"},
    {"rxn_name": "reverse acetylation", "enzymes": "AbyA2 + AbyA3 / RkF + RkG (Tetronate O-acetylation + E1cb eliminase to exomethylene)", "uniprot": "F4F7G3, F4F7G4", "pfam": "PF02458 (Acyltransferase) + PF00561 (Eliminase)", "ec": "EC 2.3.1.- / 4.2.1.-"},
    {"rxn_name": "reverse kirromycin-like substructure formation", "enzymes": "KirAVI / KirB (Kirromycin beta-alanine pyridone Dieckmann cyclase)", "uniprot": "C7C460 (KirB)", "pfam": "PF00668 + PF07993", "ec": "EC 2.3.1.-"},
    {"rxn_name": "reverse beta-lactam formation", "enzymes": "PcbC (Isopenicillin N Synthase IPNS, PDB: 1BK0)", "uniprot": "P05326 (PcbC)", "pfam": "PF03171 (2OG-FeII_Oxy)", "ec": "EC 1.21.3.1"},
    {"rxn_name": "reverse DAOC-synthetase", "enzymes": "CefE (Deacetoxycephalosporin C Synthase / Expandase DAOCS)", "uniprot": "P18548 (CefE)", "pfam": "PF03171 (2OG-FeII_Oxy)", "ec": "EC 1.14.20.1"},
    {"rxn_name": "reverse salinosporamide-like substructure formation", "enzymes": "SalC + SalA (Salinosporamide bicyclic gamma-lactam-beta-lactone KS/TE cascade)", "uniprot": "A4X5Q5 (SalC), A4X5Q3 (SalA)", "pfam": "PF00109 (ketoacyl-synt) + PF00975 (TE)", "ec": "EC 2.3.1.- / 3.1.2.-"},
    {"rxn_name": "reverse cyclization on lysine-like substructure", "enzymes": "MbtF / GobE (Mycobactin N6-hydroxylysine cyclic lactam TE/C domain)", "uniprot": "P9WQ61 (MbtF)", "pfam": "PF00975 / PF00668", "ec": "EC 3.1.2.-"},
    {"rxn_name": "open terminal aromatic ring", "enzymes": "Iterative Type I / Type II Aromatase/Cyclase (PT / ARO-CYC, e.g. CalE8, ActVII)", "uniprot": "Q8KND0 (CalE8), P16560 (ActVII)", "pfam": "PF03364 (Polyketide_cyc)", "ec": "EC 4.2.1.-"},
    {"rxn_name": "reverse spirocycle formation (type 1)", "enzymes": "RevJ (Reveromycin 6,6-spiroketalase), AveC (Avermectin spiroketalase)", "uniprot": "G4V4H2 (RevJ), Q9S0R6 (AveC)", "pfam": "PF12680 (SnoaL_2)", "ec": "EC 4.2.1.-"},
    {"rxn_name": "reverse spirocycle formation (type 2)", "enzymes": "SalBIII /SpnJ (6,5-Spiroketal synthase)", "uniprot": "A0A0F7V0Y5 (SalBIII)", "pfam": "PF12680 (SnoaL_2)", "ec": "EC 4.2.1.-"},
    {"rxn_name": "reversed cyclized hydroxy ornithine", "enzymes": "SidD / FusC (N5-hydroxyornithine cyclic hydroxamate TE domain)", "uniprot": "Q4W945 (SidD)", "pfam": "PF00975 / PF00668", "ec": "EC 3.1.2.-"},
    {"rxn_name": "reverse atrop-abyssomycin C like formation", "enzymes": "AbyZ / AbyA4 (Abyssomicin C intramolecular epoxide-opening enol-ether cyclase)", "uniprot": "F4F7I0 (AbyZ), F4F7G5 (AbyA4)", "pfam": "PF12680 / PF00561", "ec": "EC 3.3.2.-"},
    {"rxn_name": "reverse dies-alder (intramolecular)", "enzymes": "SpnF (Spinosad), LovB (Lovastatin), AbyU (Abyssomicin), PyrI4", "uniprot": "Q9AL95, Q9Y8A5, F4F7H5", "pfam": "PF08241 / PF13577", "ec": "EC 5.5.1.26"},
    {"rxn_name": "reverse dies-alder (intermolecular)", "enzymes": "MaDA / EupF (IntermolecularFAD/beta-barrel [4+2] Diels-Alderase)", "uniprot": "A0A3G1DJH3 (MaDA)", "pfam": "PF01565 (FAD_binding_4)", "ec": "EC 5.5.1.-"},
    {"rxn_name": "reverse aryl amicoumacin-like", "enzymes": "AmiI / XcnA (Amicoumacin dihydroisocoumarin aromatase/cyclase)", "uniprot": "A0A0K2JL70 (AmiI)", "pfam": "PF03364", "ec": "EC 4.2.1.-"},
    {"rxn_name": "reverse spiroborate", "enzymes": "Boromycin / Tartrolon cis-1,2/1,3-diol Borate Chelating Cavity (Bor/Trt PKS)", "uniprot": "Q9F6E1", "pfam": "PF08659 (KR cis-diol pair)", "ec": "Spontaneous on Bor/Trt diol"},
    {"rxn_name": "reverse cremimycin-like substructure", "enzymes": "CmiS1 / CmiM6 (Cremimycin FAD monooxygenase + Michael cyclase, DOI:10.1002/cbic.201300370)", "uniprot": "M9V457 (CmiS1)", "pfam": "PF01494 (FAD_binding_3)", "ec": "EC 1.14.13.-"},
    {"rxn_name": "reversed NRP biosynthesis (alpha amino acid; intermolecular)", "enzymes": "Canonical NRPS C-A-PCP Module (Condensation + Adenylation + PCP)", "uniprot": "P0C062 (GrsB)", "pfam": "PF00668 (Condensation) + PF00501 (AMP-binding)", "ec": "EC 2.3.2.- / 6.3.2.-"},
    {"rxn_name": "reversed NRP biosynthesis (alpha amino acid; intramolecular)", "enzymes": "NRPS Macrocyclizing Thioesterase (TE) or Terminal C_T Domain", "uniprot": "P0C064 (TycC TE)", "pfam": "PF00975 (Thioesterase) / PF00668 (C_T)", "ec": "EC 2.3.2.-"},
    {"rxn_name": "reverse NRP biosynthesis (aromatic nitrogen; intermolecular)", "enzymes": "Arylamine-Accepting NRPS Condensation / Amide Synthase (Asm9 / RifF / NovL)", "uniprot": "O52552 (RifF), Q9L9F7 (NovL)", "pfam": "PF00501 / PF01884 (Arylamine N-AT)", "ec": "EC 2.3.1.-"},
    {"rxn_name": "reverse NRP biosynthesis (aromatic nitrogen; intramolecular)", "enzymes": "RifF / GdmF / AnsF (Ansamycin AHBA macrodilactam amide synthase)", "uniprot": "O52552 (RifF)", "pfam": "PF01884 (NAT)", "ec": "EC 2.3.1.-"},
    {"rxn_name": "reverse urea bond formation (intermolecular)", "enzymes": "SylC / GlbC /AnpC (Syringolin/Anabaenopeptin ureido-forming NRPS C domain)", "uniprot": "A7K9I0 (SylC)", "pfam": "PF00668 (Ureido-C domain)", "ec": "EC 2.3.2.-"},
    {"rxn_name": "reverse urea bond formation (intramolecular)", "enzymes": "SylC (Syringolin A cyclic ureido-macrolactam synthase)", "uniprot": "A7K9I0", "pfam": "PF00668", "ec": "EC 2.3.2.-"},
    {"rxn_name": "reversed NRP biosynthesis (alpha amino acid; proline-like)", "enzymes": "RapP / FkbP (Rapamycin/FK506 Pipecolate/Proline C-A-PCP module)", "uniprot": "P60208 (RapP), Q9KID9 (FkbP)", "pfam": "PF00668 + PF00501", "ec": "EC 2.3.2.-"},
    {"rxn_name": "reversed NRP biosynthesis (beta amino acid)", "enzymes": "CndF /McyA / BlmIV (Beta-amino-acid incorporating NRPS C-A-PCP module)", "uniprot": "B9W4V2 (CndF)", "pfam": "PF00668 + PF00501 + PF04551 (TAM aminomutase)", "ec": "EC 2.3.2.-"},
    {"rxn_name": "break amide bond (hydroxamic)", "enzymes": "DesD / AlcC / IucA (NIS Siderophore Hydroxamate Synthase / NRPS C domain)", "uniprot": "Q9Z3Q9 (DesD)", "pfam": "PF04183 (IucA_IucC) / PF00668", "ec": "EC 6.3.2.-"},
    {"rxn_name": "reversed polyketide synthesis (saturated)", "enzymes": "Full-Reduction PKS Module (KS-AT-DH-ER-KR-ACP -> deterministic_stage)", "uniprot": "Q03131 (DEBS1)", "pfam": "PF00109+PF00698+PF14765+PF08240+PF08659", "ec": "EC 2.3.1.94"},
    {"rxn_name": "reversed polyketide synthesis (saturated anhydride)", "enzymes": "Terminal Decarboxylative PKS Offloading (KSQ/TE)", "uniprot": "Q03133", "pfam": "PF00109+PF00975", "ec": "EC 2.3.1.-"},
    {"rxn_name": "reversed polyketide synthesis (unsaturated)", "enzymes": "Dehydrating PKS Module (KS-AT-DH-KR-ACP -> deterministic_stage)", "uniprot": "Q03132 (DEBS2)", "pfam": "PF00109+PF00698+PF14765+PF08659", "ec": "EC 2.3.1.94"},
    {"rxn_name": "reversed polyketide synthesis (unsaturated and shifted type 1)", "enzymes": "trans-AT Shifted Double-Bond Module (EI / DH* isomerizing dehydratase, PksJ/RhiE)", "uniprot": "A1X8C6 (RhiE)", "pfam": "PF14765 (Enoyl isomerase DH*)", "ec": "EC 5.3.3.-"},
    {"rxn_name": "reversed polyketide synthesis (unsaturated and shifted type 2)", "enzymes": "trans-AT Beta,Gamma-Alkenyl Shift Module (CurF / BaeJ)", "uniprot": "Q9R9J1 (BaeJ)", "pfam": "PF14765", "ec": "EC 5.3.3.-"},
    {"rxn_name": "reversed polyketide synthesis (shifted and late stage oxidation)", "enzymes": "trans-AT Shifted Module + Inline Oxygenase (Oxy / P450)", "uniprot": "Q9R9J1", "pfam": "PF14765 + PF00067", "ec": "EC 1.14.14.-"},
    {"rxn_name": "reverse polyketide unit carbonylation", "enzymes": "PksF+PksG+PksH+PksI / CurB–F (PKS beta-branching HMG-CoA synthase + ECH1/ECH2 cassette)", "uniprot": "P40830 (PksG HMGS), P40805 (PksH ECH1)", "pfam": "PF00195 (HMGS) + PF00378 (ECH_1)", "ec": "EC 2.3.3.10 / 4.2.1.17"},
]


def build_enzymatic_rule_bank() -> List[EnzymaticTailoringRule]:
    """
    Construct the 4-phase tiered enzymatic rule bank combining MITE, Rhea/BKMS,
    and intramolecular pericyclases (Diels-Alder / oxa-Michael / spiroketal),
    and validate each rule's Imaginary Transition State (ITS) graph with SynKit.
    """
    raw_rules = [
        # ---------------------------------------------------------------------
        # PHASE 1: Scaffold Cyclization & Pericyclases (Diels-Alder, Oxa-Michael, Spiroketal)
        # ---------------------------------------------------------------------
        EnzymaticTailoringRule(
            rule_id=0,
            name="intramolecular_diels_alder_4plus2",
            phase=1,
            enzyme_family="Pericyclase",
            ec_class="EC 5.5.1.-",
            source_db="MITE+Rhea (SpnF/LovB)",
            # Conjugated diene + alkene -> cyclohexene bridge
            forward_smarts="([C:1]=[C:2]-[C:3]=[C:4].[C:5]=[C:6])>>[C:1]1-[C:2]=[C:3]-[C:4]-[C:6]-[C:5]-1",
            reverse_smarts="[C:1]1-[C:2]=[C:3]-[C:4]-[C:6]-[C:5]-1>>([C:1]=[C:2]-[C:3]=[C:4].[C:5]=[C:6])",
            balanced_its_rsmi="[CH2:1]=[CH:2][CH:3]=[CH2:4].[CH2:5]=[CH2:6]>>[CH2:1]1[CH:2]=[CH:3][CH2:4][CH2:6][CH2:5]1",
            is_bridge_forming=True,
            caps_parent_oxygen=False,
        ),
        EnzymaticTailoringRule(
            rule_id=1,
            name="intramolecular_oxa_michael_pyran",
            phase=1,
            enzyme_family="Pericyclase",
            ec_class="EC 5.5.1.-",
            source_db="MITE+BKMS (PKS THP cyclase)",
            # Delta-hydroxyl + alpha,beta-unsaturated carbonyl -> cyclic ether ring
            forward_smarts="[O;H1;D1:1]-[C:2]-[C:3]-[C:4]-[C:5]=[C:6]-[C:7](=[O:8])>>[O:1]1-[C:2]-[C:3]-[C:4]-[C:5]-1-[C:6]-[C:7](=[O:8])",
            reverse_smarts="[O;R:1]1-[C;R:2]-[C;R:3]-[C;R:4]-[C;R:5]-1-[C:6]-[C:7](=[O:8])>>[O;H1:1]-[C:2]-[C:3]-[C:4]-[C:5]=[C:6]-[C:7](=[O:8])",
            balanced_its_rsmi="[CH3:2][OH:1].[CH2:5]=[CH:6][CH:7]=[O:8]>>[CH3:2][O:1][CH2:5][CH2:6][CH:7]=[O:8]",
            is_bridge_forming=True,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=2,
            name="hemiketal_spiroketal_cyclization",
            phase=1,
            enzyme_family="Pericyclase",
            ec_class="EC 4.2.1.-",
            source_db="MITE+Rhea (Spiroketalase)",
            # Intramolecular ketone + hydroxyl -> cyclic hemiketal
            forward_smarts="([C;X3:1](=[O:2])([#6:3])[#6:4].[O;H1;D1:5]-[C;X4:6])>>[C:1]([O;H1:2])([#6:3])([#6:4])-[O:5]-[C:6]",
            reverse_smarts="[C;R:1]([O;H1;D1:2])([#6:3])([#6:4])-[O;R:5]-[C;R:6]>>([C:1](=[O:2])([#6:3])[#6:4].[O;H1:5]-[C:6])",
            balanced_its_rsmi="[CH3:3][C:1](=[O:2])[CH3:4].[CH3:6][OH:5]>>[CH3:3][C:1]([OH:2])([CH3:4])[O:5][CH3:6]",
            is_bridge_forming=True,
            caps_parent_oxygen=True,
        ),

        # ---------------------------------------------------------------------
        # PHASE 2: Primary C-H Functionalization & Backbone Editing (P450, BVMO, Halogenase)
        # ---------------------------------------------------------------------
        EnzymaticTailoringRule(
            rule_id=3,
            name="p450_aliphatic_ch_hydroxylation",
            phase=2,
            enzyme_family="P450",
            ec_class="EC 1.14.14.-",
            source_db="MITE+Rhea (EryF/PikC)",
            # Unactivated or allylic CH3/CH2/CH -> CH2(OH)/CH(OH)/C(OH)
            forward_smarts="[C;X4;H1,H2;!$(C([O,N,S,F,Cl,Br])):1]([#6:2])[#6:3]>>[C:1]([O;H1])([#6:2])[#6:3]",
            reverse_smarts="[C;X4:1]([O;H1;D1:4])-[#6:2]>>[C:1]-[#6:2]",
            balanced_its_rsmi="[CH3:2][CH2:1][CH3:3].[O:4]=[O:5].[H:6][H:7]>>[CH3:2][CH:1]([OH:4])[CH3:3].[H:6][O:5][H:7]",
            is_bridge_forming=False,
            caps_parent_oxygen=False,
        ),
        EnzymaticTailoringRule(
            rule_id=4,
            name="p450_alkene_epoxidation",
            phase=2,
            enzyme_family="P450",
            ec_class="EC 1.14.14.-",
            source_db="MITE+Rhea (EpoK/OleP/AbyE)",
            # Non-aromatic C=C -> epoxide (oxirane)
            forward_smarts="[C;X3;!a:1]([#6:3])=[C;X3;!a:2]([#6:4])>>[C:1]1([#6:3])-[O]-[C:2]1([#6:4])",
            reverse_smarts="[C;r3:1]1-[O;r3:5]-[C;r3:2]-1>>[C:1]=[C:2]",
            balanced_its_rsmi="[CH3:3][CH:1]=[CH:2][CH3:4].[OH2:5]>>[CH3:3][CH:1]1[O:5][CH:2]1[CH3:4].[H][H]",
            is_bridge_forming=True,
            caps_parent_oxygen=False,
        ),
        EnzymaticTailoringRule(
            rule_id=5,
            name="bvmo_baeyer_villiger_insertion",
            phase=2,
            enzyme_family="BVMO",
            ec_class="EC 1.14.13.-",
            source_db="MITE+Rhea/BKMS (BVMO)",
            # Ketone -> Ester/Lactone oxygen insertion into C-C backbone
            forward_smarts="[#6:1]-[C;X3:2](=[O:3])-[C;X4;!$(C=O):4]>>[#6:1]-[C:2](=[O:3])-[O]-[C:4]",
            reverse_smarts="[#6:1]-[C;X3:2](=[O:3])-[O;X2:5]-[C;X4:4]>>[#6:1]-[C:2](=[O:3])-[C:4]",
            balanced_its_rsmi="[CH3:1][C:2](=[O:3])[CH3:4].[OH2:5]>>[CH3:1][C:2](=[O:3])[O:5][CH3:4].[H][H]",
            is_bridge_forming=True,
            caps_parent_oxygen=False,
        ),
        EnzymaticTailoringRule(
            rule_id=6,
            name="fadh2_aliphatic_chlorination",
            phase=2,
            enzyme_family="Halogenase",
            ec_class="EC 1.14.19.-",
            source_db="MITE+Rhea (RadH/CmlS)",
            # C-H -> C-Cl
            forward_smarts="[C;X4;H1,H2,H3;!$(C([Cl,Br,F])):1]([#6:2])>>[C:1](Cl)([#6:2])",
            reverse_smarts="[C;X4:1]([Cl:3])([#6:2])>>[C:1]([#6:2])",
            balanced_its_rsmi="[CH3:2][CH3:1].[ClH:3]>>[CH3:2][CH2:1][Cl:3].[H][H]",
            is_bridge_forming=False,
            caps_parent_oxygen=False,
        ),
        EnzymaticTailoringRule(
            rule_id=7,
            name="fadh2_aromatic_or_alkene_bromination",
            phase=2,
            enzyme_family="Halogenase",
            ec_class="EC 1.14.19.-",
            source_db="MITE+Rhea",
            # sp2 C-H -> C-Br
            forward_smarts="[c,C;X3;H1:1]([#6:2])>>[c,C:1](Br)([#6:2])",
            reverse_smarts="[c,C:1]([Br:3])([#6:2])>>[c,C:1]([#6:2])",
            balanced_its_rsmi="[CH2:2]=[CH2:1].[BrH:3]>>[CH2:2]=[CH:1][Br:3].[H][H]",
            is_bridge_forming=False,
            caps_parent_oxygen=False,
        ),

        # ---------------------------------------------------------------------
        # PHASE 3: Secondary Redox & Group Transfer (SAM-MT, Alcohol DH, Acyltransferase, Epoxide Hydrolase)
        # ---------------------------------------------------------------------
        EnzymaticTailoringRule(
            rule_id=8,
            name="sam_o_methylation",
            phase=3,
            enzyme_family="Methyltransferase",
            ec_class="EC 2.1.1.-",
            source_db="MITE0000001+Rhea (O-MT)",
            # Hydroxyl (not carboxylic acid OH) -> Methoxy (-OCH3)
            forward_smarts="[#6;!$(C=O):1]-[O;H1;D1:2]>>[#6:1]-[O:2]-[CH3]",
            reverse_smarts="[#6;!$(C=O):1]-[O;D2:2]-[CH3:3]>>[#6:1]-[O;H1:2]",
            balanced_its_rsmi="[CH3:1][OH:2].[CH3:3][SH:4]>>[CH3:1][O:2][CH3:3].[SH2:4]",
            is_bridge_forming=False,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=9,
            name="sam_c_methylation_alpha_or_aromatic",
            phase=3,
            enzyme_family="Methyltransferase",
            ec_class="EC 2.1.1.-",
            source_db="MITE+Rhea (C-MT)",
            # Post-PKS C-methylation on aromatic or active methine carbon
            forward_smarts="[c,C;H1,H2:1]([#6:2])([#6:3])>>[c,C:1]([CH3])([#6:2])([#6:3])",
            reverse_smarts="[c,C:1]([CH3;D1:4])([#6:2])([#6:3])>>[c,C:1]([#6:2])([#6:3])",
            balanced_its_rsmi="[CH3:2][CH2:1][CH3:3].[CH3:4][SH:5]>>[CH3:2][CH:1]([CH3:4])[CH3:3].[SH2:5]",
            is_bridge_forming=False,
            caps_parent_oxygen=False,
        ),
        EnzymaticTailoringRule(
            rule_id=10,
            name="secondary_alcohol_oxidation",
            phase=3,
            enzyme_family="Oxidoreductase",
            ec_class="EC 1.1.1.-",
            source_db="MITE+Rhea/BKMS (ADH)",
            # Secondary alcohol -> Ketone
            forward_smarts="[#6:2]-[C;X4;H1:1](-[O;H1;D1:3])-[#6:4]>>[#6:2]-[C:1](=[O:3])-[#6:4]",
            reverse_smarts="[#6:2]-[C;X3:1](=[O;D1:3])-[#6:4]>>[#6:2]-[C;X4:1](-[O;H1:3])-[#6:4]",
            balanced_its_rsmi="[CH3:2][CH:1]([OH:3])[CH3:4]>>[CH3:2][C:1](=[O:3])[CH3:4].[H][H]",
            is_bridge_forming=False,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=11,
            name="acyltransferase_o_acetylation",
            phase=3,
            enzyme_family="Acyltransferase",
            ec_class="EC 2.3.1.-",
            source_db="MITE+Rhea+RetroMol (O-AT / Intermolecular Ester)",
            # Alcohol -> O-Acetyl / O-Acyl ester (reverse cleaves non-ring ester side chains like acetyl, propionyl, isovaleryl, Dichloroisoeverninic, etc.)
            forward_smarts="[#6;!$(C=O):1]-[O;H1;D1:2]>>[#6:1]-[O:2]-C(=O)C",
            reverse_smarts="[#6;!$(C=O):1]-[O;!R;D2:2]-[C;!R;X3:3](=[O:4])-[#6:5]>>[#6:1]-[O;H1:2]",
            balanced_its_rsmi="[CH3:1][OH:2].[CH3:5][C:3](=[O:4])[SH:6]>>[CH3:1][O:2][C:3](=[O:4])[CH3:5].[SH2:6]",
            is_bridge_forming=False,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=12,
            name="epoxide_hydrolase_diol_opening",
            phase=3,
            enzyme_family="Hydrolase",
            ec_class="EC 3.3.2.-",
            source_db="MITE+Rhea (Epoxide hydrolase)",
            # Epoxide + H2O -> vicinal 1,2-diol
            forward_smarts="[C;R3:1]1([#6:3])-[O;R3:2]-[C;R3:4]1([#6:5])>>[C:1]([O;H1])([#6:3])-[C:4]([O;H1:2])([#6:5])",
            reverse_smarts="[C;X4:1]([O;H1;D1:6])([#6:3])-[C;X4:4]([O;H1;D1:2])([#6:5])>>[C:1]1([#6:3])-[O:2]-[C:4]1([#6:5])",
            balanced_its_rsmi="[CH3:3][CH:1]1[O:2][CH:4]1[CH3:5].[OH2:6]>>[CH3:3][CH:1]([OH:6])[CH:4]([OH:2])[CH3:5]",
            is_bridge_forming=False,
            caps_parent_oxygen=False,
        ),

        # ---------------------------------------------------------------------
        # PHASE 4: Bulky Terminal Decorations (Glycosylation, Prenylation, Carbamoylation)
        # ---------------------------------------------------------------------
        EnzymaticTailoringRule(
            rule_id=13,
            name="deoxysugar_o_glycosylation",
            phase=4,
            enzyme_family="Glycosyltransferase",
            ec_class="EC 2.4.1.-",
            source_db="MITE+Rhea (DesVII/EryBV/KijC1-C4)",
            # Attach 6-deoxypyranose ring (L-rhamnose / D-desosamine core) to alcohol oxygen;
            # In reverse, cleaves a single terminal monosaccharide (<=12 heavy atoms) per enzymatic GT step!
            forward_smarts="[#6;!$(C=O):1]-[O;H1;D1:2]>>[#6:1]-[O:2]-[C@H]1O[C@H](C)[C@@H](O)[C@H](O)[C@H]1O",
            reverse_smarts="[#6;!$(C=O):1]-[O;D2:2]-[C;r6:3]1-[O;r6:4]-[C;r6:5]-[C;r6:7]-[C;r6:9]-[C;r6:11]-1>>[#6:1]-[O;H1:2]",
            balanced_its_rsmi="[CH3:1][OH:2].[CH3:6][CH:5]1[O:4][CH:3]([OH:13])[CH:11]([OH:12])[CH:9]([OH:10])[CH:7]1[OH:8]>>[CH3:1][O:2][CH:3]1[O:4][CH:5]([CH3:6])[CH:7]([OH:8])[CH:9]([OH:10])[CH:11]1[OH:12].[OH2:13]",
            is_bridge_forming=False,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=14,
            name="dmapp_prenylation",
            phase=4,
            enzyme_family="Prenyltransferase",
            ec_class="EC 2.5.1.-",
            source_db="MITE+Rhea (Prenyltransferase)",
            # Attach prenyl (3-methylbut-2-enyl) group to aromatic/alkenyl CH or phenolic OH
            forward_smarts="[#6;H1,H2:1]([#6:2])=[#6:3]>>[#6:1](CC=C(C)C)([#6:2])=[#6:3]",
            reverse_smarts="[#6:1](-[CH2:4]-[CH1:5]=[C:6](-[CH3:7])-[CH3:8])([#6:2])=[#6:3]>>[#6:1]([#6:2])=[#6:3]",
            balanced_its_rsmi="[CH2:3]=[CH:1][CH3:2].[CH3:7][C:6]([CH3:8])=[CH:5][CH2:4][OH:9]>>[CH2:3]=[C:1]([CH3:2])[CH2:4][CH:5]=[C:6]([CH3:7])[CH3:8].[OH2:9]",
            is_bridge_forming=False,
            caps_parent_oxygen=False,
        ),
        EnzymaticTailoringRule(
            rule_id=15,
            name="carbamoyltransferase_o_carbamoylation",
            phase=4,
            enzyme_family="Acyltransferase",
            ec_class="EC 2.1.3.-",
            source_db="MITE+Rhea (Asm21/NovN)",
            # Alcohol -> O-Carbamoyl (-OC(=O)NH2)
            forward_smarts="[#6;!$(C=O):1]-[O;H1;D1:2]>>[#6:1]-[O:2]-C(=O)N",
            reverse_smarts="[#6;!$(C=O):1]-[O;D2:2]-[C;X3:3](=[O:4])-[NH2;D1:5]>>[#6:1]-[O;H1:2]",
            balanced_its_rsmi="[CH3:1][OH:2].[NH2:5][C:3](=[O:4])[OH:6]>>[CH3:1][O:2][C:3](=[O:4])[NH2:5].[OH2:6]",
            is_bridge_forming=False,
            caps_parent_oxygen=True,
        ),

        # ---------------------------------------------------------------------
        # TETRONATE / SPIROTETRONATE & BRIDGED EPOXIDE CASCADE RULES (Abyssomicin / Chlorothricin / Quartromicin / Kijanimicin)
        # ---------------------------------------------------------------------
        EnzymaticTailoringRule(
            rule_id=16,
            name="tetronate_glyceryl_offloading_cyclization",
            phase=1,
            enzyme_family="Acyltransferase",
            ec_class="EC 2.3.1.-",
            source_db="MITE+Rhea (AbyA1/AbyA2/AbyA3/KijA-E)",
            # Linear beta-keto acid + glyceryl-ACP -> exomethylene tetronate (handles both keto and enol tetronate tautomers in reverse)
            forward_smarts="[#6:23](=[O:24])-[CH2:7]-[C;X3:8](=[O:9])-[O;H1;D1]>>[CH2]=[C]1-[O]-[C](=O)-[CH:7]([#6:23](=[O:24]))-[C:8]1=[O:9]",
            reverse_smarts="[CH2:2]=[C:3]1-[O:4]-[C:5]~[C:7]([#6:23](=[O:24]))-[C:8]1=[O:9]>>[CH2:7]([#6:23](=[O:24]))-[C:8](=[O:9])O",
            balanced_its_rsmi="[CH3:23][C:24](=O)[CH2:7][C:8](=[O:9])[OH:10].[CH2:2]=[C:3]([OH:4])[C:5](=[O:6])[OH:11]>>[CH2:2]=[C:3]1[O:4][C:5](=[O:6])[CH:7]([C:24]([CH3:23])=O)[C:8]1=[O:9].[OH2:10].[OH2:11]",
            is_bridge_forming=True,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=17,
            name="spirotetronate_diels_alder_4plus2",
            phase=1,
            enzyme_family="Pericyclase",
            ec_class="EC 5.5.1.-",
            source_db="MITE+Rhea (AbyU/ChlJ/QmnH/Kij)",
            # Conjugated diene + exomethylene tetronate dienophile -> spirotetronate cyclohexene (both keto & enol tetronate tautomers)
            forward_smarts="([C:1]=[C:10]-[C:11]=[C:13].[C;D1:2]=[C:3]1-[O:4]-[C:5](=[O:6])-[C:7]-[C:8]1=[O:9])>>[C:1]1-[C:2]-[C:3]2([O:4]-[C:5](=[O:6])-[C:7]-[C:8]2=[O:9])-[C:13]-[C:11]=[C:10]-1",
            reverse_smarts="[C;R:1]1-[C;R:2]-[C;R:3]2([O;R:4]-[C;R:5]~[C;R:7]-[C;R:8]2=[O:9])-[C;R:13]-[C;R:11]=[C;R:10]-1>>([C:1]=[C:10]-[C:11]=[C:13].[C:2]=[C:3]1-[O:4]-[C:5]~[C:7]-[C:8]1=[O:9])",
            balanced_its_rsmi="[CH2:1]=[CH:10][CH:11]=[CH2:13].[CH2:2]=[C:3]1[O:4][C:5](=[O:6])[CH2:7][C:8]1=[O:9]>>[CH2:1]1[CH2:2][C:3]2([O:4][C:5](=[O:6])[CH2:7][C:8]2=[O:9])[CH2:13][CH:11]=[CH:10]1",
            is_bridge_forming=True,
            caps_parent_oxygen=False,
        ),
        EnzymaticTailoringRule(
            rule_id=18,
            name="intramolecular_epoxide_enol_ether_opening",
            phase=3,
            enzyme_family="Hydrolase",
            ec_class="EC 3.3.2.-",
            source_db="MITE+Rhea (AbyZ/AbyA4)",
            # Intramolecular attack of tetronate keto/enol oxygen onto epoxide to form bridged hydroxy-ether
            forward_smarts="([C;R:7]-[C;R:8](=[O:9])-[C;R:3].[C;r3:10]1-[O;r3:12]-[C;r3:11]-1-[C;R:13])>>[C:7]=[C:8](-[O:9]-[C:10]-[C:11](-[O;H1:12])-[C:13])-[C:3]",
            reverse_smarts="([C;R:7]=[C;R:8](-[O;R:9]-[C;R:10]-[C;R:11](-[O;H1;D1:12])-[C;R:13])-[C;R:3])>>([C:7]-[C:8](=[O:9])-[C:3].[C:10]1-[O:12]-[C:11]-1-[C:13])",
            balanced_its_rsmi="[CH3:3][C:8](=[O:9])[CH3:7].[CH2:10]1[O:12][CH:11]1[CH3:13]>>[CH3:3][C:8](=[CH2:7])[O:9][CH2:10][CH:11]([OH:12])[CH3:13]",
            is_bridge_forming=True,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=19,
            name="polyether_epoxide_zipper_cyclization",
            phase=3,
            enzyme_family="Hydrolase",
            ec_class="EC 3.3.2.- / 5.5.1.-",
            source_db="MITE+RetroMol (MonBI/MonBII/Lsd19)",
            forward_smarts="[O;H1;D1:1]-[C:2]-[C:3]-[C:4]-[C;r3:5]1-[O;r3:6]-[C;r3:7]-1>>[O:1]1-[C:2]-[C:3]-[C:4]-[C:5]-1-[C:7]-[O;H1:6]",
            reverse_smarts="[O;R:1]1-[C;R:2]-[C;R:3]-[C;R:4]-[C;R:5]-1-[C:7]-[O;H1;D1:6]>>[O;H1:1]-[C:2]-[C:3]-[C:4]-[C:5]1-[O:6]-[C:7]-1",
            balanced_its_rsmi="[CH3:2][OH:1].[CH2:5]1[O:6][CH2:7]1>>[CH3:2][O:1][CH2:5][CH2:7][OH:6]",
            is_bridge_forming=True,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=20,
            name="nrps_terminal_amino_acid_condensation",
            phase=1,
            enzyme_family="Acyltransferase",
            ec_class="EC 2.3.2.- / 6.3.2.-",
            source_db="RetroMol rxn.yml (NRPS C-A-PCP module)",
            forward_smarts="[#6:1]-[C;X3:2](=[O:3])-[O;H1;D1:4]>>[#6:1]-[C:2](=[O:3])-[NH]-[CH2]-C(=O)O",
            reverse_smarts="[#6:1]-[C;X3:2](=[O:3])-[NH1:5]-[C;X4;!R:6]-[C;X3:7](=[O:8])-[O;H1:9]>>[#6:1]-[C:2](=[O:3])-[O;H1]",
            balanced_its_rsmi="[CH3:1][C:2](=[O:3])[OH:4].[NH2:5][CH2:6][C:7](=[O:8])[OH:9]>>[CH3:1][C:2](=[O:3])[NH:5][CH2:6][C:7](=[O:8])[OH:9].[OH2:4]",
            is_bridge_forming=False,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=21,
            name="nrps_thiazole_heterocyclization_oxidation",
            phase=2,
            enzyme_family="Pericyclase",
            ec_class="EC 1.3.99.- / 4.2.1.-",
            source_db="RetroMol rxn.yml (EpoB/CtaB Cy+Ox domain)",
            forward_smarts="[#6:1]-[C;X3:2](=[O:3])-[NH1:4]-[CH2:5]-[C;X3:6](=[O:7])-[O;H1:8]>>[#6:1]-c1nc([C:6](=[O:7])[O:8])cs1",
            reverse_smarts="[#6:1]-[c:2]1[n:4][c:6]([C:7](=[O:8])[O;H1:9])[c:5][s:3]1>>[#6:1]-[C:2](=O)-[NH:4]-[CH2:5]-[C:7](=[O:8])[O:9]",
            balanced_its_rsmi="[CH3:1][C:2](=[O:3])[NH:4][CH:5]([CH2:6][SH:7])[C:8](=[O:9])[OH:10]>>[CH3:1][c:2]1[n:4][c:5]([C:8](=[O:9])[OH:10])[cH:6][s:7]1.[OH2:3].[H][H]",
            is_bridge_forming=True,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=22,
            name="tetramate_dieckmann_offloading_cyclization",
            phase=2,
            enzyme_family="Acyltransferase",
            ec_class="EC 2.3.1.-",
            source_db="RetroMol rxn.yml (TmcA/IkaA Dieckmann R0/TE)",
            forward_smarts="[#6:1]-[C;X3:2](=[O:3])-[CH2:4]-[C;X3:5](=[O:6])-[NH1:7]-[CH2:8]-[C;X3:9](=[O:10])-[O;H1]>>[#6:1]-[C:2](-[O;H1:3])=[C:4]1-[C:5](=[O:6])-[NH:7]-[CH2:8]-[C:9]1=[O:10]",
            reverse_smarts="[#6:1]-[C:2](-[O;H1:3])=[C:4]1-[C:5](=[O:6])-[NH1:7]-[C:8]-[C:9]1=[O:10]>>[#6:1]-[C:2](=[O:3])-[CH2:4]-[C:5](=[O:6])-[NH:7]-[C:8]-[C:9](=[O:10])O",
            balanced_its_rsmi="[CH3:1][C:2](=[O:3])[CH2:4][C:5](=[O:6])[NH:7][CH2:8][C:9](=[O:10])[OH:11]>>[CH3:1][C:2]([OH:3])=[C:4]1[C:5](=[O:6])[NH:7][CH2:8][C:9]1=[O:10].[OH2:11]",
            is_bridge_forming=True,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=23,
            name="sam_n_methylation",
            phase=3,
            enzyme_family="Methyltransferase",
            ec_class="EC 2.1.1.-",
            source_db="RetroMol rxn.yml (Inline NRPS nMT / DesVI)",
            forward_smarts="[#6:1]-[NH1,NH2:2]>>[#6:1]-[N:2]-[CH3]",
            reverse_smarts="[#6:1]-[N;X3:2]-[CH3:3]>>[#6:1]-[N:2]",
            balanced_its_rsmi="[CH3:1][NH2:2].[CH3:3][SH:4]>>[CH3:1][NH:2][CH3:3].[SH2:4]",
            is_bridge_forming=False,
            caps_parent_oxygen=False,
        ),
        EnzymaticTailoringRule(
            rule_id=24,
            name="nrps_proline_pipecolate_condensation",
            phase=1,
            enzyme_family="Acyltransferase",
            ec_class="EC 2.3.2.- / 6.3.2.-",
            source_db="RetroMol rxn.yml (RapP/FkbP pipecolate/proline module)",
            forward_smarts="[#6:1]-[C;X3:2](=[O:3])-[O;H1;D1:4]>>[#6:1]-[C:2](=[O:3])-N1CCCC[C@H]1C(=O)O",
            reverse_smarts="[#6:1]-[C;X3:2](=[O:3])-[N;R1:4]1-[C;R1:5]-[C;R1:6]-[C;R1:7]-[C;R1:8](-[C:9](=[O:10])[O;H1:11])1>>[#6:1]-[C:2](=[O:3])-[O;H1]",
            balanced_its_rsmi="[CH3:1][C:2](=[O:3])[OH:4].[NH:5]1[CH2:6][CH2:7][CH2:8][CH:9]1[C:10](=[O:11])[OH:12]>>[CH3:1][C:2](=[O:3])[N:5]1[CH2:6][CH2:7][CH2:8][CH:9]1[C:10](=[O:11])[OH:12].[OH2:4]",
            is_bridge_forming=False,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=25,
            name="nrps_oxazole_heterocyclization_oxidation",
            phase=2,
            enzyme_family="Pericyclase",
            ec_class="EC 4.2.1.- / 1.3.99.-",
            source_db="RetroMol rxn.yml (OzmP/PksJ/DisD Cy+Ox domain)",
            forward_smarts="[#6:1]-[C;X3:2](=[O:3])-[NH1:4]-[CH2:5]-[C;X3:6](=[O:7])-[O;H1:8]>>[#6:1]-c1nc([C:6](=[O:7])[O:8])co1",
            reverse_smarts="[#6:1]-[c:2]1[n:4][c:6]([C:7](=[O:8])[O;H1:9])[c:5][o:3]1>>[#6:1]-[C:2](=O)-[NH:4]-[CH2:5]-[C:7](=[O:8])[O:9]",
            balanced_its_rsmi="[CH3:1][C:2](=[O:3])[NH:4][CH:5]([CH2:6][OH:7])[C:8](=[O:9])[OH:10]>>[CH3:1][c:2]1[n:4][c:5]([C:8](=[O:9])[OH:10])[cH:6][o:7]1.[OH2:3].[H][H]",
            is_bridge_forming=True,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=26,
            name="paps_hydroxyl_sulfonation",
            phase=4,
            enzyme_family="Acyltransferase",
            ec_class="EC 2.8.2.-",
            source_db="RetroMol rxn.yml (CurM-ST / StaL sulfotransferase)",
            forward_smarts="[#6;!$(C=O):1]-[O;H1;D1:2]>>[#6:1]-[O:2]-S(=O)(=O)O",
            reverse_smarts="[#6;!$(C=O):1]-[O;D2:2]-[S;X4:3](=[O:4])(=[O:5])-[O;H1:6]>>[#6:1]-[O;H1:2]",
            balanced_its_rsmi="[CH3:1][OH:2].[OH:6][S:3](=[O:4])(=[O:5])[OH:7]>>[CH3:1][O:2][S:3](=[O:4])(=[O:5])[OH:6].[OH2:7]",
            is_bridge_forming=False,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=27,
            name="kirromycin_pyridone_cyclization",
            phase=2,
            enzyme_family="Acyltransferase",
            ec_class="EC 2.3.1.-",
            source_db="RetroMol rxn.yml (KirB / TenS pyridone cyclase)",
            forward_smarts="[#6:1]-[C;X3:2](=[O:3])-[CH2:4]-[C;X3:5](=[O:6])-[NH1:7]-[CH2:8]-[C;X3:9](=[O:10])-[O;H1]>>[#6:1]-c1c(O)cc[nH]c1=O",
            reverse_smarts="[#6:1]-[c:2]1[c:3]([O;H1:4])[c:5][c:6][nH:7][c:8]1=[O:9]>>[#6:1]-[C:2](=[O:4])-[CH2:3]-[C:8](=[O:9])-[NH:7]-[CH2:6]-[C:5](=O)O",
            balanced_its_rsmi="[CH3:1][C:2](=[O:4])[CH2:3][C:8](=[O:9])[NH:7][CH2:6][CH2:5][C:10](=[O:11])[OH:12]>>[CH3:1][c:2]1[c:3]([OH:4])[cH:5][cH:6][nH:7][c:8]1=[O:9].[OH2:12].[H][H]",
            is_bridge_forming=True,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=28,
            name="cyclic_acetal_dioxolane_formation",
            phase=3,
            enzyme_family="Oxidoreductase",
            ec_class="EC 1.14.11.-",
            source_db="RetroMol rxn.yml (AviO1 / HygX Fe/2-OG dioxolane synthase)",
            forward_smarts="[C;X4:1]([O;H1;D1:2])-[C;X4:3]([O;H1;D1:4])>>[C:1]1-[O:2]-[CH2]-[O:4]-[C:3]-1",
            reverse_smarts="[C;R:1]1-[O;R:2]-[CH2;R:5]-[O;R:4]-[C;R:3]-1>>[C:1]([O;H1:2])-[C:3]([O;H1:4])",
            balanced_its_rsmi="[CH2:1]([OH:2])[CH2:3][OH:4].[CH2:5]=[O:6]>>[CH2:1]1[O:2][CH2:5][O:4][CH2:3]1.[OH2:6]",
            is_bridge_forming=True,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=29,
            name="oxidative_nitrile_installation",
            phase=3,
            enzyme_family="P450",
            ec_class="EC 1.14.99.-",
            source_db="RetroMol rxn.yml (BorI/BorJ/BorK nitrile synthase)",
            forward_smarts="[C;X4;H1,H2:1]([#6:2])([#6:3])>>[C:1](C#N)([#6:2])([#6:3])",
            reverse_smarts="[C:1](-[C;D2:4]#[N;D1:5])([#6:2])([#6:3])>>[C:1]([#6:2])([#6:3])",
            balanced_its_rsmi="[CH3:2][CH2:1][CH3:3].[CH:4]#[N:5]>>[CH3:2][CH:1]([C:4]#[N:5])[CH3:3].[H][H]",
            is_bridge_forming=False,
            caps_parent_oxygen=False,
        ),
        EnzymaticTailoringRule(
            rule_id=30,
            name="p450_methyl_to_carboxyl_oxidation",
            phase=2,
            enzyme_family="P450",
            ec_class="EC 1.14.14.-",
            source_db="MITE+Rhea (PimG / NysN / AmphN polyene C-methyl oxidase)",
            # Exocyclic alpha-CH3 branch on polyene/macrolide -> -C(=O)OH (leaving C1 ester/acid intact!)
            forward_smarts="[C;X4;H1:1]([#6:5])([#6:6])-[CH3;D1:2]>>[C:1]([#6:5])([#6:6])-[C:2](=O)O",
            reverse_smarts="[C;X4:1]([#6:5])([#6:6])-[C;X3:2](=[O:3])-[O;H1;D1:4]>>[C:1]([#6:5])([#6:6])-[CH3:2]",
            balanced_its_rsmi="[CH3:5][CH:1]([CH3:6])[CH3:2].[O:3]=[O:4]>>[CH3:5][CH:1]([CH3:6])[C:2](=[O:3])[OH:4].[H][H]",
            is_bridge_forming=False,
            caps_parent_oxygen=False,
        ),
        EnzymaticTailoringRule(
            rule_id=31,
            name="flavin_ahba_quinone_lactamo_oxidation",
            phase=2,
            enzyme_family="Oxidoreductase",
            ec_class="EC 1.14.13.- / 2.3.1.-",
            source_db="MITE+RetroMol (GdmM / GdmF AHBA benzoquinone ansamycin synthase)",
            # AHBA aminophenol macrolactone/carboxylic acid -> 1,4-benzoquinone macrolactam (and reverse!)
            forward_smarts="([#6:1]-[c:2]1[cH:3][c:4]([O;H1:5])[cH:6][c:7]([NH2:8])[cH:12]1.[#6:11]-[C;R:9](=[O:10])-[O;R:14]-[C;R:15])>>([#6:1]-[C:2]1=[CH:3]-[C:4](=[O:5])-[CH:6]=[C:7](-[NH:8]-[C:9](=[O:10])-[#6:11])-[C:12]1=O.[O;H1:14]-[C:15])",
            reverse_smarts="[#6:1]-[C;R:2]1=[C;R:3]-[C;R:4](=[O:5])-[C;R:6]=[C;R:7](-[NH1;R:8]-[C;R:9](=[O:10])-[#6:11])-[C;R:12]1=[O:13]>>([#6:1]-[c:2]1[c:3][c:4]([O;H1:5])[c:6][c:7]([NH2:8])[cH:12]1.[#6:11]-[C:9](=[O:10])O)",
            balanced_its_rsmi="[CH3:1][c:2]1[cH:3][c:4]([OH:5])[cH:6][c:7]([NH2:8])[cH:12]1.[CH3:11][C:9](=[O:10])[OH:14].[O:13]=[O:15]>>[CH3:1][C:2]1=[CH:3][C:4](=[O:5])[CH:6]=[C:7]([NH:8][C:9](=[O:10])[CH3:11])[C:12]1=[O:13].[OH2:14].[OH2:15]",
            is_bridge_forming=True,
            caps_parent_oxygen=True,
        ),
        EnzymaticTailoringRule(
            rule_id=32,
            name="polyether_cyclic_ether_unzipping",
            phase=1,
            enzyme_family="Pericyclase",
            ec_class="EC 5.5.1.- / 3.3.2.-",
            source_db="RetroMol rxn.yml (MonBI / Lsd19 / SalBIII 5/6-ring polyether cyclase)",
            # Intramolecular hydroxyl + alkene -> 5- or 6-membered THF/THP polyether ring
            forward_smarts="[O;H1;D1:1]-[C:2]-[C:3]-[C:4]-[C:5]=[C:6]>>[O:1]1-[C:2]-[C:3]-[C:4]-[C:5]-1-[CH1:6]",
            reverse_smarts="[O;r5,r6:1]1-[C;r5,r6:2]-[C;r5,r6:3]-[C;r5,r6:4]-[C;r5,r6:5]-1-[C;X4;H1,H2:6]>>[O;H1:1]-[C:2]-[C:3]-[C:4]-[C:5]=[C:6]",
            balanced_its_rsmi="[CH3:2][OH:1].[CH2:5]=[CH2:6]>>[CH3:2][O:1][CH2:5][CH3:6]",
            is_bridge_forming=True,
            caps_parent_oxygen=True,
        ),
    ]

    # Validate every rule's ITS graph and extract reaction center (RC) size with SynKit
    validated_rules: List[EnzymaticTailoringRule] = []
    for r in raw_rules:
        meta = ENZYME_CATALOG_BY_RULE_ID.get(r.rule_id, {})
        r.candidate_enzymes = meta.get("candidate_enzymes", r.source_db)
        r.uniprot_ids = meta.get("uniprot_ids", "")
        r.pfam_domains = meta.get("pfam_domains", "")
        r.cofactors = meta.get("cofactors", "")
        try:
            its_graph = rsmi_to_its(r.balanced_its_rsmi, core=True)
            syn_rule = SynRule(its_graph, name=r.name)
            r.synkit_rc_nodes = len(syn_rule.rc.raw.nodes())
            r.synkit_rc_edges = len(syn_rule.rc.raw.edges())
        except Exception as exc:
            print(f"Warning: SynKit ITS construction fallback for {r.name}: {exc}")
            r.synkit_rc_nodes = 2
            r.synkit_rc_edges = 1
        validated_rules.append(r)

    return validated_rules


# =============================================================================
# 2. ATOM-TRACKED FORWARD APPLICATION & CAUSAL EXPOSED-LEAF TRACKING
# =============================================================================

@dataclass
class AppliedModification:
    """Record of one forward tailoring step applied during a rollout."""
    step_idx: int
    rule_id: int
    rule_name: str
    phase: int
    enzyme_family: str
    anchor_atom_map_nums: List[int]  # Atom map numbers on the core/parent where modification occurred
    installed_atom_map_nums: List[int]  # Newly installed decoration atom map numbers
    depends_on_step: Optional[int]  # If this step capped/modified atoms installed in an earlier step


def prepare_tracked_core_mol(core_rec: Dict[str, Any]) -> Optional[Chem.Mol]:
    """
    Create an RDKit Mol for M_0 where every core atom i has AtomMapNum = i + 1 (1..N_core)
    and stores its register role and module origin.
    """
    mol = Chem.MolFromSmiles(core_rec["smiles"])
    if mol is None:
        return None
    for idx, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(idx + 1)
    return mol


def apply_forward_rule_tracked(
    mol_in: Chem.Mol,
    rule: EnzymaticTailoringRule,
    next_map_base: int,
    core_map_max: int,
    history: List[AppliedModification],
    rng: random.Random,
    prefer_tailored_sites_in_phase3_4: bool = True,
) -> Optional[Tuple[Chem.Mol, AppliedModification, int]]:
    """
    Apply `rule.forward_smarts` to `mol_in`, propagating atom map numbers from
    reactant to product so we know exactly which atoms are nascent PKS core atoms
    (`1..core_map_max`) vs. newly added tailoring atoms (`>= next_map_base`).
    """
    rxn_list = [AllChem.ReactionFromSmarts(rule.forward_smarts)]
    if rule.rule_id == 31:
        # Also support linear carboxylic-acid AHBA cores (thiolysis release) -> benzoquinone macrolactam!
        rxn_list.append(
            AllChem.ReactionFromSmarts(
                "([#6:1]-[c:2]1[cH:3][c:4]([O;H1:5])[cH:6][c:7]([NH2:8])[cH:12]1.[#6:11]-[C;!R:9](=[O:10])-[O;H1:14])>>[#6:1]-[C:2]1=[CH:3]-[C:4](=[O:5])-[CH:6]=[C:7](-[NH:8]-[C:9](=[O:10])-[#6:11])-[C:12]1=O"
            )
        )
    # Copy input molecule and store map numbers in Isotope so RDKit reaction preserves them
    reactant = Chem.Mol(mol_in)
    map_by_idx = {}
    for atom in reactant.GetAtoms():
        amap = atom.GetAtomMapNum()
        map_by_idx[atom.GetIdx()] = amap
        atom.SetIsotope(amap)
        atom.SetAtomMapNum(0)

    raw_products = []
    for rxn in rxn_list:
        try:
            raw_products.extend(rxn.RunReactants((reactant,)))
        except Exception:
            pass

    if not raw_products:
        return None

    # Collect valid sanitized single-fragment products
    valid_candidates = []
    for prod_tuple in raw_products:
        if not prod_tuple:
            continue
        prod = Chem.Mol(prod_tuple[0])
        try:
            Chem.SanitizeMol(prod)
            Chem.AssignStereochemistry(prod, cleanIt=True, force=True)
        except Exception:
            continue
        if len(Chem.GetMolFrags(prod)) != 1:
            continue

        # Identify which existing atoms were modified and which atoms are newly added (Isotope == 0)
        new_atoms_count = sum(1 for a in prod.GetAtoms() if a.GetIsotope() == 0)
        # Determine anchor atoms (existing atoms whose degree, H-count, or bond order changed, or neighbors of new atoms)
        anchor_maps: Set[int] = set()
        for a in prod.GetAtoms():
            iso = a.GetIsotope()
            if iso == 0:
                for nb in a.GetNeighbors():
                    if nb.GetIsotope() > 0:
                        anchor_maps.add(int(nb.GetIsotope()))
        if not anchor_maps:
            # For rearrangements / oxidations / cyclizations without new atoms, compare atom environments
            old_env = {
                a.GetIsotope(): (a.GetDegree(), a.GetTotalNumHs(), int(a.IsInRing()))
                for a in reactant.GetAtoms()
            }
            for a in prod.GetAtoms():
                iso = a.GetIsotope()
                if iso > 0 and old_env.get(iso) != (a.GetDegree(), a.GetTotalNumHs(), int(a.IsInRing())):
                    anchor_maps.add(int(iso))

        if not anchor_maps:
            continue

        # Biologically realistic targeting:
        # In Phases 3 & 4 (capping/glycosylation/methylation/oxidation), strongly prefer
        # capping a newly installed tailoring hydroxyl (from Phase 2 P450/epoxide) OR a peripheral hydroxyl
        # so we generate realistic non-commuting cascades (e.g. P450 -> O-MT or P450 -> O-GT)!
        touches_previous_tailoring = any(m > core_map_max for m in anchor_maps)
        valid_candidates.append((touches_previous_tailoring, prod, sorted(anchor_maps), new_atoms_count))

    if not valid_candidates:
        return None

    if prefer_tailored_sites_in_phase3_4 and rule.phase in (3, 4):
        tailored_hits = [c for c in valid_candidates if c[0]]
        if tailored_hits:
            valid_candidates = tailored_hits

    _, chosen_prod, anchor_maps, _ = rng.choice(valid_candidates)

    # Restore AtomMapNum from Isotope and assign fresh map numbers to newly added atoms
    installed_maps: List[int] = []
    cur_map = next_map_base
    for atom in chosen_prod.GetAtoms():
        iso = atom.GetIsotope()
        atom.SetIsotope(0)
        if iso > 0:
            atom.SetAtomMapNum(int(iso))
        else:
            atom.SetAtomMapNum(cur_map)
            installed_maps.append(cur_map)
            cur_map += 1

    # Check if this step depends on (caps) an earlier step in `history`
    depends_on: Optional[int] = None
    for prev_mod in reversed(history):
        if set(anchor_maps) & set(prev_mod.installed_atom_map_nums):
            depends_on = prev_mod.step_idx
            break
        if rule.caps_parent_oxygen and (set(anchor_maps) & set(prev_mod.anchor_atom_map_nums)):
            depends_on = prev_mod.step_idx
            break

    mod = AppliedModification(
        step_idx=len(history) + 1,
        rule_id=rule.rule_id,
        rule_name=rule.name,
        phase=rule.phase,
        enzyme_family=rule.enzyme_family,
        anchor_atom_map_nums=anchor_maps,
        installed_atom_map_nums=installed_maps,
        depends_on_step=depends_on,
    )
    return chosen_prod, mod, cur_map


def extract_state_supervision_labels(
    mol: Chem.Mol,
    core_rec: Dict[str, Any],
    history: List[AppliedModification],
    rule_bank: Dict[int, EnzymaticTailoringRule],
) -> Dict[str, Any]:
    """
    Given a tailored intermediate M_t (with atom map numbers preserved), compute the exact
    ground-truth supervision targets for the 3-Head Reverse-Peeling GNN:
      - Canonical unmapped isomeric SMILES (`smiles`)
      - Node masks in canonical SMILES atom order:
        * `is_pks_core_atom`: List[bool]
        * `atom_register_role`: List[str]
        * `is_exposed_peel_atom`: List[bool] (atoms belonging to exposed leaf modifications C_exposed(M_t))
      - Edge masks:
        * `cleavable_bridge_bonds`: List[Tuple[int, int]] (bonds between core and exposed decoration, or Diels-Alder/BVMO bridges)
      - Policy targets:
        * `exposed_rule_ids`: List[int] (reverse rules currently valid & uncapped on M_t)
        * `primary_reverse_rule_id`: int (the LIFO last-applied rule_id)
      - Value target:
        * `distance_to_pks`: int (t = len(history))
        * `value_target`: float = exp(-t / 2.5)
    """
    core_map_max = len(core_rec["is_pks_core_atom"])
    orig_roles = core_rec["atom_register_role"]

    # Determine which steps in `history` are currently EXPOSED leaves (not capped by a dependent step)
    capped_steps: Set[int] = {
        m.depends_on_step for m in history if m.depends_on_step is not None
    }
    exposed_mods = [m for m in history if m.step_idx not in capped_steps]

    exposed_map_nums: Set[int] = set()
    for m in exposed_mods:
        exposed_map_nums.update(m.anchor_atom_map_nums)
        exposed_map_nums.update(m.installed_atom_map_nums)

    # Canonicalize molecule while tracking atom order via RDKit `_smilesAtomOutputOrder`
    unmapped = Chem.Mol(mol)
    map_at_old_idx = [a.GetAtomMapNum() for a in unmapped.GetAtoms()]
    for a in unmapped.GetAtoms():
        a.SetAtomMapNum(0)
        a.SetIsotope(0)

    canon_smiles = Chem.MolToSmiles(unmapped, isomericSmiles=True)
    canon_mol = Chem.MolFromSmiles(canon_smiles)
    if canon_mol is None:
        raise ValueError(f"Failed to parse canonical SMILES: {canon_smiles}")

    # Map canonical atom indices back to our tracked AtomMapNums
    # By embedding AtomMapNum into Isotope on a copy and matching to canon_mol:
    iso_copy = Chem.Mol(mol)
    for a in iso_copy.GetAtoms():
        a.SetIsotope(a.GetAtomMapNum())
        a.SetAtomMapNum(0)
    iso_smiles = Chem.MolToSmiles(iso_copy, isomericSmiles=True)
    iso_mol = Chem.MolFromSmiles(iso_smiles)

    n_atoms = canon_mol.GetNumAtoms()
    is_core_atom: List[bool] = []
    register_roles: List[str] = []
    is_exposed_atom: List[bool] = []
    map_num_by_canon_idx: List[int] = []

    for idx in range(n_atoms):
        amap = iso_mol.GetAtomWithIdx(idx).GetIsotope() if iso_mol is not None else 0
        map_num_by_canon_idx.append(amap)
        if 1 <= amap <= core_map_max:
            is_core_atom.append(True)
            register_roles.append(orig_roles[amap - 1])
        else:
            is_core_atom.append(False)
            register_roles.append("tailoring_decoration")
        is_exposed_atom.append(amap in exposed_map_nums)

    # Identify cleavable/bridge bonds in canonical atom indices:
    # 1) Bonds connecting a core atom and a tailoring decoration atom
    # 2) Bonds inside exposed bridge-forming modifications (Diels-Alder, oxa-Michael, spiroketal, BVMO, epoxide)
    cleavable_bonds: List[List[int]] = []
    for bond in canon_mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if is_core_atom[u] != is_core_atom[v]:
            cleavable_bonds.append([u, v])
        elif is_exposed_atom[u] and is_exposed_atom[v]:
            # Check if an exposed bridge-forming rule touched both u and v
            mu, mv = map_num_by_canon_idx[u], map_num_by_canon_idx[v]
            for m in exposed_mods:
                if rule_bank[m.rule_id].is_bridge_forming:
                    if mu in m.anchor_atom_map_nums and mv in m.anchor_atom_map_nums:
                        cleavable_bonds.append([u, v])
                        break

    exposed_rule_ids = sorted({m.rule_id for m in exposed_mods})
    primary_rule_id = history[-1].rule_id if history else -1
    dist = len(history)
    val_target = float(np.exp(-dist / 2.5)) if dist > 0 else 1.0

    return {
        "smiles": canon_smiles,
        "is_pks_core_atom": is_core_atom,
        "atom_register_role": register_roles,
        "is_exposed_peel_atom": is_exposed_atom,
        "cleavable_bridge_bonds": cleavable_bonds,
        "exposed_rule_ids": exposed_rule_ids,
        "primary_reverse_rule_id": primary_rule_id,
        "distance_to_pks": dist,
        "value_target": val_target,
    }


# =============================================================================
# 3. CORE-CORRUPTED DEAD-END NEGATIVE GENERATOR (FOR HEAD 3 VALUE NETWORK)
# =============================================================================

DEAD_END_CORRUPTION_SMARTS = [
    # Strip an inline PKS beta-hydroxyl into an alkane (destroys KR register)
    ("strip_inline_kr_hydroxyl", "[C;X4:1]([O;H1;D1])([#6:2])[C;X4:3][C:4](=[O:5])>>[C:1]([#6:2])[C:3][C:4](=[O:5])"),
    # Strip an inline alpha-methyl from a methylmalonyl unit (corrupts AT extender identity)
    ("strip_inline_alpha_methyl", "[C;X4:1]([CH3;D1])([#6:2])[C:3](=[O:4])[O,N:5]>>[C:1]([#6:2])[C:3](=[O:4])[O,N:5]"),
    # Break a C-C bond inside the PKS carbon backbone (corrupts chain length)
    ("cleave_backbone_cc_bond", "[CH2:1]-[CH2:2]-[C:3](=[O:4])[O;H1:5]>>[CH3:2]-[C:3](=[O:4])[O;H1:5]"),
    # Decarboxylate the C1 carboxylic acid (destroys the C1 anchor required by deterministic_stage)
    ("decarboxylate_c1_acid", "[#6:1]-[C;X3:2](=[O:3])[O;H1;D1:4]>>[#6;H1:1]"),
]


def generate_core_corrupted_dead_end(
    core_rec: Dict[str, Any],
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    """
    Generate a negative/dead-end state M_dead by corrupting an inline PKS functional group
    or C1 anchor on M_0, and verifying that `decompose_pks_terminal` fails or changes module count.
    Labeled with `distance_to_pks = -1` (infinity) and `value_target = 0.0`.
    """
    mol = Chem.MolFromSmiles(core_rec["smiles"])
    if mol is None:
        return None

    rules = list(DEAD_END_CORRUPTION_SMARTS)
    rng.shuffle(rules)

    for corr_name, smarts in rules:
        rxn = AllChem.ReactionFromSmarts(smarts)
        try:
            prods = rxn.RunReactants((mol,))
        except Exception:
            continue
        for prod_tuple in prods:
            if not prod_tuple:
                continue
            cand = Chem.Mol(prod_tuple[0])
            try:
                Chem.SanitizeMol(cand)
                Chem.AssignStereochemistry(cand, cleanIt=True, force=True)
            except Exception:
                continue
            if len(Chem.GetMolFrags(cand)) != 1 or cand.GetNumAtoms() < 4:
                continue
            cand_smi = Chem.MolToSmiles(cand, isomericSmiles=True)
            if cand_smi == core_rec["smiles"]:
                continue

            n = cand.GetNumAtoms()
            return {
                "smiles": cand_smi,
                "corruption_type": corr_name,
                "is_pks_core_atom": [False] * n,
                "atom_register_role": ["corrupted_core"] * n,
                "is_exposed_peel_atom": [False] * n,
                "cleavable_bridge_bonds": [],
                "exposed_rule_ids": [],
                "primary_reverse_rule_id": -1,
                "distance_to_pks": -1,
                "value_target": 0.0,
            }
    return None


# =============================================================================
# 4. MULTI-STEP TRAJECTORY GENERATOR PER NASCENT PKS CORE
# =============================================================================

def generate_trajectory_for_core(
    core_rec: Dict[str, Any],
    rule_bank: List[EnzymaticTailoringRule],
    max_steps: int = 3,
    rng: Optional[random.Random] = None,
) -> Optional[Dict[str, Any]]:
    """
    Given a nascent PKS core M_0, generate a biologically ordered multi-step
    forward tailoring trajectory M_0 -> M_1 -> ... -> M_k (with monotonically
    non-decreasing phase numbers Phase 1 <= Phase 2 <= Phase 3 <= Phase 4)
    plus a core-corrupted dead-end negative M_dead.
    """
    if rng is None:
        rng = random.Random(42)

    mol_0 = prepare_tracked_core_mol(core_rec)
    if mol_0 is None:
        return None

    rule_by_id = {r.rule_id: r for r in rule_bank}
    core_map_max = len(core_rec["is_pks_core_atom"])
    next_map_base = core_map_max + 1

    # Record M_0 (step 0: nascent PKS core)
    try:
        state_0 = extract_state_supervision_labels(mol_0, core_rec, [], rule_by_id)
    except Exception:
        return None

    steps_data: List[Dict[str, Any]] = [
        {
            "step": 0,
            "parent_smiles": core_rec["smiles"],
            "applied_rule_id": -1,
            "applied_rule_name": "nascent_pks_core",
            "phase": 0,
            "enzyme_family": "PKS_Core",
            **state_0,
        }
    ]

    cur_mol = mol_0
    cur_phase = 1
    history: List[AppliedModification] = []
    seen_smiles: Set[str] = {core_rec["smiles"]}

    for step_idx in range(1, max_steps + 1):
        # Enforce Tiered Biological Grammar across rollout steps:
        # Step 1: Phase 1 (Pericyclases / Diels-Alder / Spiroketals) -> Phase 2 (P450 / BVMO / Halogenases)
        # Step 2: Phase 2 or Phase 3 (Methyltransferases / Oxidoreductases / Acyltransferases / Epoxide Hydrolases)
        # Step 3+: Phase 3 or Phase 4 (Deoxysugar Glycosyltransferases / Prenyltransferases / Carbamoyltransferases)
        phase1_rules = [r for r in rule_bank if r.phase == 1]
        phase2_rules = [r for r in rule_bank if r.phase == 2]
        phase3_rules = [r for r in rule_bank if r.phase == 3]
        phase4_rules = [r for r in rule_bank if r.phase == 4]
        for grp in (phase1_rules, phase2_rules, phase3_rules, phase4_rules):
            rng.shuffle(grp)

        if step_idx == 1:
            # Always try specialized Phase 1 cascades + specialized Phase 2 (AHBA quinone Rule 31 / P450-COOH Rule 30) first
            spec_p2 = [r for r in phase2_rules if r.rule_id in (31, 30, 4, 3, 5)]
            other_p2 = [r for r in phase2_rules if r.rule_id not in (31, 30, 4, 3, 5)]
            if "ahba" in str(core_rec.get("starter_code", "")).lower():
                candidate_rules = [rule_by_id[31]] + phase1_rules + spec_p2 + other_p2 + phase3_rules
            else:
                candidate_rules = phase1_rules + spec_p2 + other_p2 + phase3_rules
        elif step_idx == 2:
            # Put Phase 4 (glycosylation/carbamoylation/sulfonation) & Phase 2 before generic Phase 3 methylations!
            candidate_rules = (
                (phase2_rules + phase4_rules + phase3_rules + phase1_rules)
                if cur_phase == 1
                else (phase4_rules + phase2_rules + phase3_rules)
            )
        else:
            # In Steps 3 & 4, prioritize Phase 4 (terminal sugars, carbamates, sulfates, prenyls) first so the GNN
            # sees hundreds of glycosylated, carbamoylated, and multi-sugar states!
            candidate_rules = phase4_rules + phase3_rules + [r for r in phase2_rules if r.phase >= cur_phase]

        # Prioritize natural biosynthetic cascades when a precursor step just fired:
        if history:
            last_rule_id = history[-1].rule_id
            cascade_next = {
                16: [17],
                17: [4],
                4: [18, 19, 12, 13, 30],
                12: [28, 13],
                20: [22, 21, 25, 27, 23],
                24: [3, 8, 13],
                30: [13, 4, 2],
                31: [15, 8],
                32: [4, 8, 3],
                13: [13, 15, 11],
                3: [13, 15, 26, 8, 11, 10],
            }.get(last_rule_id, [])
            if cascade_next:
                priority = [rule_by_id[rid] for rid in cascade_next if rid in rule_by_id]
                rest = [r for r in candidate_rules if r.rule_id not in cascade_next]
                candidate_rules = priority + rest

        applied = None
        parent_smiles = steps_data[-1]["smiles"]

        for rule in candidate_rules:
            res = apply_forward_rule_tracked(
                cur_mol, rule, next_map_base, core_map_max, history, rng
            )
            if res is None:
                continue
            next_mol, mod, new_map_base = res
            try:
                state_t = extract_state_supervision_labels(
                    next_mol, core_rec, history + [mod], rule_by_id
                )
            except Exception:
                continue
            if state_t["smiles"] in seen_smiles:
                continue

            applied = (next_mol, mod, new_map_base, state_t, rule)
            break

        if applied is None:
            break

        next_mol, mod, next_map_base, state_t, rule = applied
        cur_mol = next_mol
        cur_phase = max(cur_phase, rule.phase)
        history.append(mod)
        seen_smiles.add(state_t["smiles"])

        steps_data.append(
            {
                "step": step_idx,
                "parent_smiles": parent_smiles,
                "applied_rule_id": rule.rule_id,
                "applied_rule_name": rule.name,
                "phase": rule.phase,
                "enzyme_family": rule.enzyme_family,
                **state_t,
            }
        )

    if len(steps_data) < 2:
        # Must have at least 1 valid forward tailoring step
        return None

    dead_end = generate_core_corrupted_dead_end(core_rec, rng)

    return {
        "root_pks_smiles": core_rec["smiles"],
        "release_mechanism": core_rec["release_mechanism"],
        "num_extension_modules": core_rec["num_extension_modules"],
        "starter_code": core_rec["starter_code"],
        "extender_codes": core_rec["extender_codes"],
        "num_tailoring_steps": len(steps_data) - 1,
        "steps": steps_data,
        "dead_end": dead_end,
    }


# =============================================================================
# 5. VISUALIZATION OF MULTI-STEP TAILORING TRAJECTORIES
# =============================================================================

def draw_sample_trajectories(trajectories: List[Dict[str, Any]], output_png: str, num_trajectories: int = 3):
    """
    Render a publication-quality grid showing multi-step forward/reverse trajectories:
    M_0 (Nascent PKS Core) -> M_1 (Phase 1/2 Tailored) -> M_2 (Phase 3 Capped) -> M_3 (Phase 4 Decorated)
    with PKS Core atoms highlighted in green and Exposed Leaf Tailoring atoms highlighted in magenta/orange.
    """
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    rich_trajs = [t for t in trajectories if len(t["steps"]) >= 4]
    if not rich_trajs:
        rich_trajs = [t for t in trajectories if len(t["steps"]) >= 3]
    if not rich_trajs:
        rich_trajs = trajectories[:num_trajectories]

    selected = rich_trajs[:num_trajectories]
    mols = []
    legends = []
    highlight_atoms = []
    highlight_colors = []

    COLOR_CORE = (0.60, 0.88, 0.65)       # Soft green for nascent PKS core atoms
    COLOR_EXPOSED = (1.00, 0.45, 0.45)    # Coral red for currently exposed leaf modification
    COLOR_CAPPED = (0.95, 0.80, 0.35)     # Amber for internal/capped historical modification

    for t_idx, traj in enumerate(selected, start=1):
        steps = traj["steps"][:4]
        for s in steps:
            mol = Chem.MolFromSmiles(s["smiles"])
            if mol is None:
                continue
            AllChem.Compute2DCoords(mol)

            h_atoms = []
            h_col = {}
            for a_idx in range(mol.GetNumAtoms()):
                is_core = s["is_pks_core_atom"][a_idx]
                is_exp = s["is_exposed_peel_atom"][a_idx]
                if is_exp and not is_core:
                    h_atoms.append(a_idx)
                    h_col[a_idx] = COLOR_EXPOSED
                elif not is_core:
                    h_atoms.append(a_idx)
                    h_col[a_idx] = COLOR_CAPPED
                else:
                    h_atoms.append(a_idx)
                    h_col[a_idx] = COLOR_CORE

            mols.append(mol)
            highlight_atoms.append(h_atoms)
            highlight_colors.append(h_col)

            step_num = s["step"]
            if step_num == 0:
                legends.append(
                    f"Traj {t_idx} | M_0: Nascent PKS Core (d=0, V=1.00)\n"
                    f"LM:{traj['starter_code']} + {'+'.join(traj['extender_codes'])}"
                )
            else:
                legends.append(
                    f"Traj {t_idx} | M_{step_num}: {s['applied_rule_name']}\n"
                    f"Phase {s['phase']} ({s['enzyme_family']}) | d={s['distance_to_pks']} | V={s['value_target']:.2f}"
                )

    if mols:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=4,
            subImgSize=(450, 340),
            legends=legends,
            highlightAtomLists=highlight_atoms,
            highlightAtomColors=highlight_colors,
            useSVG=False,
        )
        img.save(output_png)
        print(f"Saved multi-step trajectory visualization to: {output_png}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate tiered SynKit/MITE/Rhea multi-step trajectories")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help="Input parquet of stereo-preserved PKS cores with provenance")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output parquet file with trajectory states and backward-compatible triplets")
    parser.add_argument("--stats", type=str, default=DEFAULT_STATS,
                        help="Output JSON file for trajectory generation statistics")
    parser.add_argument("--max_cores", type=int, default=1200,
                        help="Maximum number of PKS cores to expand into multi-step trajectories")
    parser.add_argument("--rollouts_per_core", type=int, default=2,
                        help="Number of distinct branched trajectories per PKS core")
    parser.add_argument("--max_steps", type=int, default=3,
                        help="Maximum tailoring steps K per trajectory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    print("=" * 72)
    print("SYNKIT + MITE + RHEA/BKMS TIERED RETROBIOSYNTHESIS TRAJECTORY GENERATOR")
    print("=" * 72)

    # 1. Build and validate the enzymatic rule bank with SynKit
    rule_bank = build_enzymatic_rule_bank()
    print(f"\nValidated {len(rule_bank)} enzymatic reaction rules via SynKit ITS/SynRule:")
    for r in rule_bank:
        print(
            f"  [Rule {r.rule_id:2d} | Phase {r.phase}] {r.name:<38s} "
            f"| {r.enzyme_family:<20s} | RC=({r.synkit_rc_nodes}N, {r.synkit_rc_edges}E)"
        )

    os.makedirs(os.path.dirname(DEFAULT_RULEBANK_JSON), exist_ok=True)
    with open(DEFAULT_RULEBANK_JSON, "w") as f:
        json.dump([asdict(r) for r in rule_bank], f, indent=2)
    print(f"Saved rule bank metadata to {DEFAULT_RULEBANK_JSON}")

    # 2. Load stereo-preserved nascent PKS cores
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Core provenance file not found: {input_path}. Run scripts/01_generate_PKS_products.py first."
        )

    df_cores = pd.read_parquet(input_path)
    if args.max_cores and len(df_cores) > args.max_cores:
        # Always include all long-chain family cores (3-14 modules from generate_by_family.py) first,
        # then sample evenly from 1-2 module cores so the GNN trains across 1 to 14 extension modules!
        long_cores = df_cores[df_cores["num_extension_modules"] >= 3]
        short_cores = df_cores[df_cores["num_extension_modules"] < 3]
        rem = max(0, args.max_cores - len(long_cores))
        if rem > 0 and len(short_cores) > rem:
            stride = max(1, len(short_cores) // rem)
            short_cores = short_cores.iloc[::stride].head(rem)
        df_cores = pd.concat([long_cores, short_cores], ignore_index=True)
    print(f"\nLoaded {len(df_cores)} stereo-preserved nascent PKS cores (modules range: "
          f"{int(df_cores['num_extension_modules'].min())}..{int(df_cores['num_extension_modules'].max())}) from {input_path}")

    # 3. Generate multi-step forward trajectories + dead-end negatives
    trajectories: List[Dict[str, Any]] = []
    flat_state_rows: List[Dict[str, Any]] = []
    triplet_rows: List[Dict[str, Any]] = []

    core_records = df_cores.to_dict("records")
    for core_idx, core_rec in enumerate(tqdm(core_records, desc="Generating trajectories", unit="core")):
        # Convert numpy arrays inside core_rec to Python lists
        core_rec_clean = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in core_rec.items()
        }

        is_ahba = "ahba" in str(core_rec_clean.get("starter_code", "")).lower()
        n_rollouts = max(args.rollouts_per_core, 20 if is_ahba else args.rollouts_per_core)
        for branch_idx in range(n_rollouts):
            traj = generate_trajectory_for_core(
                core_rec_clean, rule_bank, max_steps=args.max_steps, rng=rng
            )
            if traj is None:
                continue

            traj_id = len(trajectories)
            trajectories.append(traj)

            # Record all trajectory states M_0, M_1, ..., M_k
            # Following PATRO (`extract_single_step_reactions.py` & `mcts_controller.py`),
            # for a route M_0 -> M_1 -> ... -> M_K with forward rules [r_1, r_2, ..., r_K],
            # when reverse-peeling from M_K down to M_0, the reaction context at state M_t
            # is the sequence of reverse rules already peeled along the path M_K -> ... -> M_t:
            #     rxn_context(M_t) = [r_K, r_{K-1}, ..., r_{t+1}]  (oldest peeled first, most recent peeled last)
            all_forward_rules = [s["applied_rule_id"] for s in traj["steps"] if s["step"] >= 1]
            for step_dict in traj["steps"]:
                # Include M_0 once per root core (on branch 0) to avoid duplicating M_0
                if step_dict["step"] == 0 and branch_idx > 0:
                    continue
                t_step = step_dict["step"]
                # Reverse rules peeled from M_K down to M_t:
                peeled_so_far = list(reversed(all_forward_rules[t_step:])) if t_step >= 0 else []
                flat_state_rows.append(
                    {
                        "trajectory_id": traj_id,
                        "root_pks_smiles": traj["root_pks_smiles"],
                        "smiles": step_dict["smiles"],
                        "parent_smiles": step_dict["parent_smiles"],
                        "step": step_dict["step"],
                        "phase": step_dict["phase"],
                        "enzyme_family": step_dict["enzyme_family"],
                        "applied_rule_id": step_dict["applied_rule_id"],
                        "applied_rule_name": step_dict["applied_rule_name"],
                        "primary_reverse_rule_id": step_dict["primary_reverse_rule_id"],
                        "exposed_rule_ids_json": json.dumps(step_dict["exposed_rule_ids"]),
                        "rxn_context_rule_ids_json": json.dumps(peeled_so_far),
                        "is_pks_core_atom_json": json.dumps(step_dict["is_pks_core_atom"]),
                        "atom_register_role_json": json.dumps(step_dict["atom_register_role"]),
                        "is_exposed_peel_atom_json": json.dumps(step_dict["is_exposed_peel_atom"]),
                        "cleavable_bridge_bonds_json": json.dumps(step_dict["cleavable_bridge_bonds"]),
                        "distance_to_pks": step_dict["distance_to_pks"],
                        "value_target": step_dict["value_target"],
                        "state_type": "nascent_pks_core" if step_dict["step"] == 0 else "tailored_intermediate",
                        "label": 1 if step_dict["step"] == 0 else 0,
                    }
                )

            # Record core-corrupted dead-end negative M_dead (once per core on branch 0)
            if branch_idx == 0 and traj["dead_end"] is not None:
                de = traj["dead_end"]
                flat_state_rows.append(
                    {
                        "trajectory_id": traj_id,
                        "root_pks_smiles": traj["root_pks_smiles"],
                        "smiles": de["smiles"],
                        "parent_smiles": traj["root_pks_smiles"],
                        "step": -1,
                        "phase": -1,
                        "enzyme_family": "Corrupted_Core_DeadEnd",
                        "applied_rule_id": -1,
                        "applied_rule_name": de["corruption_type"],
                        "primary_reverse_rule_id": -1,
                        "exposed_rule_ids_json": json.dumps([]),
                        "rxn_context_rule_ids_json": json.dumps([]),
                        "is_pks_core_atom_json": json.dumps(de["is_pks_core_atom"]),
                        "atom_register_role_json": json.dumps(de["atom_register_role"]),
                        "is_exposed_peel_atom_json": json.dumps(de["is_exposed_peel_atom"]),
                        "cleavable_bridge_bonds_json": json.dumps([]),
                        "distance_to_pks": -1,
                        "value_target": 0.0,
                        "state_type": "corrupted_dead_end",
                        "label": 0,
                    }
                )

            # Also build backward-compatible triplet row (root PKS, 1-step tailored, K-step tailored / dead-end)
            m0_smi = traj["root_pks_smiles"]
            m1_smi = traj["steps"][1]["smiles"]
            mk_smi = (
                traj["steps"][-1]["smiles"]
                if len(traj["steps"]) > 2
                else (traj["dead_end"]["smiles"] if traj["dead_end"] else m1_smi)
            )
            triplet_rows.append(
                {
                    "trajectory_id": traj_id,
                    "pks_smiles": m0_smi,
                    "enzymatic_aug_smiles": m1_smi,
                    "synthetic_aug_smiles": mk_smi,
                    "num_tailoring_steps": traj["num_tailoring_steps"],
                }
            )

    # Save trajectory states parquet & triplet parquet
    df_states = pd.DataFrame(flat_state_rows)
    states_output_path = Path(args.output).with_name("pks_peeling_trajectory_states.parquet")
    df_states.to_parquet(states_output_path, index=False)

    df_triplets = pd.DataFrame(triplet_rows)
    df_triplets.to_parquet(args.output, index=False)

    # Render trajectory visualization
    traj_plot_path = os.path.join(PLOTS_DIR, "multi_step_tailoring_trajectories.png")
    draw_sample_trajectories(trajectories, traj_plot_path, num_trajectories=3)

    # Compute summary statistics
    phase_counts = df_states[df_states["phase"] > 0]["phase"].value_counts().to_dict()
    family_counts = df_states[df_states["phase"] > 0]["enzyme_family"].value_counts().to_dict()
    step_counts = df_states["step"].value_counts().to_dict()

    stats = {
        "num_input_cores": len(df_cores),
        "num_trajectories": len(trajectories),
        "total_trajectory_states": len(df_states),
        "num_rules_in_bank": len(rule_bank),
        "step_distribution": {str(k): int(v) for k, v in step_counts.items()},
        "phase_distribution": {f"Phase_{k}": int(v) for k, v in sorted(phase_counts.items())},
        "enzyme_family_distribution": {str(k): int(v) for k, v in family_counts.items()},
    }
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 72)
    print("TRAJECTORY GENERATION SUMMARY")
    print("=" * 72)
    print(f"Total trajectories generated:      {len(trajectories):,}")
    print(f"Total states saved:                {len(df_states):,} -> {states_output_path}")
    print(f"Triplet table saved:               {len(df_triplets):,} -> {args.output}")
    print(f"Step distribution:                 {stats['step_distribution']}")
    print(f"Phase distribution:                {stats['phase_distribution']}")
    print(f"Enzyme family distribution:        {stats['enzyme_family_distribution']}")
    print(f"Visualization saved to:            {traj_plot_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
