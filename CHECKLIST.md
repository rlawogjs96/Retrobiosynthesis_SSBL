1. Make sure that we have correctly abided logic about KR domains in Keasling Lab.
2. Make sure we are getting the best performance on 183 ClusterCAD BGCs without using any form of bonus logic (ad-hoc situational score bonuses) in our model when performing retrobiosynthesis.
3. Make sure our GNN model is properly coded and trained.
4. Make sure that we are properly using SynCat, PATRO to our terms

---

## ClusterCAD 183-BGC Benchmark Tracker (Incremental Cache — Passed BGCs Locked)

- **Latest Summary (`183` Total BGCs = `102` Pure PKS + `24` Hybrid PKS-NRPS + `57` Pure NRPS):**
  - **Pure NRPS Rejected as Non-PKS:** `57 / 57 (100.0%)`
  - **Solved PKS + Hybrid BGCs:** **`37 / 126 (29.4%)`** (`35 / 102 = 34.3%` Pure PKS, `2 / 24 = 8.3%` Hybrid PKS-NRPS)
  - **Exact Stereo & Constitution:** **`36 / 37 (97.3% of solved)`**

### Certified Solved BGCs (`37` Total — Do Not Re-Run)
| # | BGC ID | Natural Product | Class | Peeling Depth | Reconstructed Assembly Line | Verification Status |
|---:|---|---|---|---:|---|---|
| 1 | `BGC0000001` | Abyssomicin | Pure_PKS | 4 | `LM:mal` + 6 PKS modules | `exact_stereo_and_constitution` |
| 2 | `BGC0000002` | Aculeximycin | Pure_PKS | 5 | `LM:mal` + 19 PKS modules | `exact_stereo_and_constitution` |
| 3 | `BGC0000034` | Candicidin | Pure_PKS | 2 | `LM:paba` + 21 PKS modules | `exact_stereo_and_constitution` |
| 4 | `BGC0000035` | Chalcomycin | Pure_PKS | 5 | `LM:mal` + 7 PKS modules | `exact_stereo_and_constitution` |
| 5 | `BGC0000040` | Concanamycin A | Pure_PKS | 1 | `LM:mal` + 13 PKS modules | `exact_stereo_and_constitution` |
| 6 | `BGC0000047` | Dihydrochalcomycin | Pure_PKS | 2 | `LM:mal` + 7 PKS modules | `exact_stereo_and_constitution` |
| 7 | `BGC0000054` | Erythromycin A | Pure_PKS | 4 | `LM:mmal` + 6 PKS modules | `exact_stereo_and_constitution` |
| 8 | `BGC0000055` | Erythromycin | Pure_PKS | 4 | `LM:mmal` + 6 PKS modules | `exact_stereo_and_constitution` |
| 9 | `BGC0000058` | FD-891 | Pure_PKS | 2 | `LM:mal` + 12 PKS modules | `exact_stereo_and_constitution` |
| 10 | `BGC0000061` | Candicidin A | Pure_PKS | 2 | `LM:paba` + 21 PKS modules | `exact_stereo_and_constitution` |
| 11 | `BGC0000074` | Herbimycin A | Pure_PKS | 4 | `LM:ahba` + 7 PKS modules | `exact_stereo_and_constitution` |
| 12 | `BGC0000082` | Kijanimicin | Pure_PKS | 9 | `LM:mal` + 11 PKS modules | `exact_stereo_and_constitution` |
| 13 | `BGC0000090` | Macbecin I | Pure_PKS | 4 | `LM:ahba` + 7 PKS modules | `exact_stereo_and_constitution` |
| 14 | `BGC0000092` | Megalomicin | Pure_PKS | 5 | `LM:mmal` + 6 PKS modules | `exact_stereo_and_constitution` |
| 15 | `BGC0000094` | Methymycin / Pikromycin | Pure_PKS | 2 | `LM:mmal` + 5 PKS modules | `exact_stereo_and_constitution` |
| 16 | `BGC0000102` | Mycinamicin | Pure_PKS | 4 | `LM:mmal` + 7 PKS modules | `exact_stereo_and_constitution` |
| 17 | `BGC0000103` | Mycolactone | Pure_PKS | 4 | `LM:mal` + 9 PKS modules | `exact_stereo_and_constitution` |
| 18 | `BGC0000105` | Nanchangmycin *(Polyether)* | Pure_PKS | 8 | `LM:hmal` + 14 PKS modules | `exact_stereo_and_constitution` |
| 19 | `BGC0000108` | Natamycin / Pimaricin | Pure_PKS | 3 | `LM:mal` + 12 PKS modules | `exact_stereo_and_constitution` |
| 20 | `BGC0000114` | Nigericin *(Polyether)* | Pure_PKS | 7 | `LM:hmal` + 14 PKS modules | `exact_stereo_and_constitution` |
| 21 | `BGC0000115` | Nystatin A1 | Pure_PKS | 2 | `LM:mal` + 18 PKS modules | `exact_stereo_and_constitution` |
| 22 | `BGC0000116` | Nystatin-like Polyene | Pure_PKS | 2 | `LM:mal` + 18 PKS modules | `exact_stereo_and_constitution` |
| 23 | `BGC0000117` | Oligomycin | Pure_PKS | 3 | `LM:mal` + 12 PKS modules | `exact_stereo_and_constitution` |
| 24 | `BGC0000122` | Phenylnannolone A | Pure_PKS | 0 | `LM:benz` + 5 PKS modules | `exact_stereo_and_constitution` |
| 25 | `BGC0000123` | Phoslactomycin B | Pure_PKS | 0 | `LM:chc` + 7 PKS modules | `exact_stereo_and_constitution` |
| 26 | `BGC0000125` | Natamycin / Pimaricin | Pure_PKS | 3 | `LM:mal` + 12 PKS modules | `exact_stereo_and_constitution` |
| 27 | `BGC0000144` | Salinomycin *(Polyether)* | Pure_PKS | 4 | `LM:mal` + 14 PKS modules | `constitution_rebuild_match` |
| 28 | `BGC0000166` | Tylactone | Pure_PKS | 0 | `LM:mmal` + 7 PKS modules | `exact_stereo_and_constitution` |
| 29 | `BGC0001012` | Meridamycin | Hybrid_PKS_NRPS | 2 | `LM:mal` + 14 PKS (+1 NRPS) | `exact_stereo_and_constitution` |
| 30 | `BGC0001065` | Herboxidiene *(Polyether/THP)* | Pure_PKS | 5 | `LM:mmal` + 8 PKS modules | `exact_stereo_and_constitution` |
| 31 | `BGC0001394` | Phenalamide C7-OH | Hybrid_PKS_NRPS | 1 | `LM:plac` + 8 PKS (+1 NRPS) | `exact_stereo_and_constitution` |
| 32 | `BGC0001492` | Abyssomicin C | Pure_PKS | 4 | `LM:mal` + 6 PKS modules | `exact_stereo_and_constitution` |
| 33 | `BGC0001495` | Actinoallolide | Pure_PKS | 2 | `LM:mmal` + 10 PKS modules | `exact_stereo_and_constitution` |
| 34 | `BGC0001580` | Ebelactone | Pure_PKS | 0 | `LM:mal` + 6 PKS modules | `exact_stereo_and_constitution` |
| 35 | `BGC0001690` | Natamycin | Pure_PKS | 3 | `LM:mal` + 12 PKS modules | `exact_stereo_and_constitution` |
| 36 | `BGC0001709` | Nystatin | Pure_PKS | 2 | `LM:mal` + 18 PKS modules | `exact_stereo_and_constitution` |
| 37 | `BGC0001828` | Griseochelin / Zincophorin *(Polyether)* | Pure_PKS | 2 | `LM:mmal` + 11 PKS modules | `exact_stereo_and_constitution` |

---

## External Antibiotic Discovery Screen (`NPAtlas` + `COCONUT` + `LOTUS` — Out-of-ClusterCAD Targets)

Screened `7,273` unique Type I PKS-like natural products across `NPAtlas`, `COCONUT`, and `LOTUS` with the Multitask ESKAPE + Cytotoxicity GNN (`BioactivityPredictor/models/mtl_bioactivity.pt`), selecting top high-ESKAPE, low-cytotoxicity candidates for 8-GPU Retrobiosynthesis Discovery.

- **Key Result:** Solved **`38` out-of-ClusterCAD antibiotic/bioactive natural products**, including **`28` tailored natural products (`Depth = 1–4`) that BioPKS / RetroTide alone (`Depth = 0`) cannot solve (`BioPKS_D0 = False`)**.

| # | Database : ID | Natural Product | BioPKS Alone (`D=0`) | Our Depth | ESKAPE Score (`Cyto`) | BGC Status | Reverse-Tailoring Trajectory (`Stage 1`) | Reconstructed Assembly Line (`Stage 2`) | Verification |
|---:|---|---|:---:|---:|---|---|---|---|---|
| 1 | `LOTUS:Q77506551` | **Bu-4514N** *(BMS Microtetraspora antibiotic)* | `False` | **4** | `0.999` (`0.358`) | **Orphan BGC** *(De Novo)* | `R0 (Diels-Alder)` $\to$ `R22 (Tetramate Dieckmann)` $\to$ `R13 (Deoxysugar)` $\to$ `R20 (NRPS AA)` | `LM:mal` + 7 PKS + 1 NRPS | `exact_stereo_and_constitution` |
| 2 | `NPAtlas:NPA022322` | **8-OH-22-dehydroxymethylkijanolide** *(Spirotetronate)* | `False` | **4** | `1.000` (`0.449`) | **Orphan Congener** | `R3 (P450 OH)` $\to$ `R0 (IMDA)` $\to$ `R0 (IMDA)` $\to$ `R16 (Tetronate)` | `LM:mal` + 11 PKS | `exact_stereo_and_constitution` |
| 3 | `LOTUS:Q105383017` | **Abyssomicin V** *(Anti-MRSA/TB spirotetronate)* | `False` | **4** | `0.994` (`0.355`) | **Orphan Congener** | `R18 (Epoxide-enol opening)` $\to$ `R4 (Epox)` $\to$ `R17 (Spirotetronate DA)` $\to$ `R16 (Tetronate)` | `LM:hmal` + 6 PKS | `exact_stereo_and_constitution` |
| 4 | `LOTUS:Q105162516` | **Abyssomicin 3** *(Anti-MRSA/TB spirotetronate)* | `False` | **4** | `0.990` (`0.363`) | **Orphan Congener** | `R18 (Epoxide-enol opening)` $\to$ `R4 (Epox)` $\to$ `R17 (Spirotetronate DA)` $\to$ `R16 (Tetronate)` | `LM:mal` + 6 PKS | `exact_stereo_and_constitution` |
| 5 | `NPAtlas:NPA000197` | **Maltepolide B** *(Myxobacterial THF-macrolide)* | `False` | **4** | `0.912` (`0.290`) | **Orphan BGC** *(De Novo)* | `R8 (O-MT)` $\to$ `R19 (Polyether 5-exo)` $\to$ `R4 (Epox)` $\to$ `R4 (Epox)` | `LM:mal` + 11 PKS | `exact_stereo_and_constitution` |
| 6 | `NPAtlas:NPA00228` | **Tirandamycin G** *(Anti-VRE tetramic acid)* | `False` | **4** | `0.945` (`0.240`) | **Known Family** *(Alt Pathway)* | `R9 (C-MT)` $\to$ `R13 (Ketal/Sugar)` $\to$ `R22 (Tetramate)` $\to$ `R20 (NRPS Gly)` | `LM:mal` + 4 PKS + 1 NRPS | `exact_stereo_and_constitution` |
| 7 | `NPAtlas:NPA002911` | **Antifungal Macrolide** *(Bis-THP lactone)* | `False` | **4** | `1.000` (`0.389`) | **Orphan BGC** *(De Novo)* | `R5 (BVMO)` $\to$ `R1 (Oxa-Michael THP)` $\to$ `R9 (C-MT)` $\to$ `R1 (Oxa-Michael THP)` | `LM:mal` + 8 PKS | `exact_stereo_and_constitution` |
| 8 | `NPAtlas:NPA015510` / `COCONUT:CNP0160240.1` | **Oblongolide N** *( plus 7 congeners B, D, E, O, P, Q, S)* | `False` | **1–4** | `1.000` (`0.023–0.302`) | **Orphan Fungal** *(Chimeric Bacterial PKS)* | `R3 (P450 OH)` $\to$ `R9 (C-MT)` $\to$ `R3 (P450 OH)` $\to$ `R0 (IMDA)` | `LM:hmal` + 5 PKS *(all 8 congeners converge!)* | `exact_stereo_and_constitution` |
| 9 | `NPAtlas:NPA030416` / `NPA030417` | **Flavofungin IV & V** *(THF-bridged polyene macrolides)* | `False` | **2** | `1.000` (`0.243`) | **Orphan BGC** *(De Novo)* | `R19 (Polyether 5-exo)` $\to$ `R4 (P450 Epoxidation)` | `LM:isobut` + 15 PKS / `LM:mal` + 16 PKS | `exact_stereo_and_constitution` |
| 10 | `NPAtlas:NPA028434` | **Reedsmycin F** *(THF-bridged polyene antibiotic)* | `False` | **2** | `1.000` (`0.226`) | **Orphan BGC** *(De Novo)* | `R19 (Polyether 5-exo)` $\to$ `R4 (P450 Epoxidation)` | `LM:mal` + 16 PKS | `exact_stereo_and_constitution` |
| 11 | `NPAtlas:NPA023205` | **Tricholide A** *(Cyanobacterial macrolide antibiotic)* | `False` | **2** | `1.000` (`0.423`) | **Orphan BGC** *(De Novo)* | `R8 (SAM O-MT)` $\to$ `R24 (NRPS Pro/Pipecolate)` | `LM:mal` + 7 PKS + 1 NRPS | `exact_stereo_and_constitution` |
| 12 | `NPAtlas:NPA014051` | **Antascomicins A, D, E** *(Pipecolate macrolactams)* | `False` | **2–3** | `1.000` (`0.175`) | **Orphan BGC** *(De Novo)* | `R3 (P450 OH)` $\to$ `R24 (NRPS Pipecolate)` $\to$ `R3 (P450 OH)` | `LM:dhch` + 10 PKS + 1 NRPS | `constitution_rebuild_match` |
| 13 | `NPAtlas:NPA031104` | **Mangrolide D** *(Gram-negative ribosome antibiotic)* | `False` | **1** | `1.000` (`0.379`) | **Orphan BGC** *(De Novo)* | `R13 (Deoxysugar O-glycosylation)` | `LM:isobut` + 8 PKS | `exact_stereo_and_constitution` |
| 14 | `NPAtlas:NPA032246–8` | **Strasseriolides A, B, C** *(Antimalarial/anti-infective macrolides)* | `False` | **1** | `1.000` (`0.388`) | **Orphan Fungal** *(Chimeric Bacterial PKS)* | `R30 (P450 methyl-to-carboxyl oxidation)` | `LM:isobut` + 8 PKS | `exact_stereo_and_constitution` |
| 15 | `NPAtlas:NPA011763` | **Multiplolide A** *(Antifungal epoxy-macrolide)* | `False` | **1** | `1.000` (`0.242`) | **Orphan Fungal** *(Chimeric Bacterial PKS)* | `R4 (P450 alkene epoxidation)` | `LM:mal` + 4 PKS | `exact_stereo_and_constitution` |
| 16 | `COCONUT:CNP0497220.1` | **Cryptofolione-class Dihydropyran-6-ones** *(5 hits)* | `True` | **0** | `0.998` (`0.060`) | **Orphan Plant/Fungal** *(BioPKS Baseline)* | Direct Nascent PKS (`Depth = 0`) | `LM:mal` + 6–9 PKS | `exact_stereo_and_constitution` |

---

## Finalized Paper Showcase Portfolio: In-House Gene-Verified Wet-Lab Targets (`gene_search.py` DP over `1,683` In-House T1PKS Genes)

Every target below is:
1. **Unsolvable by BioPKS / RetroTide alone (`BioPKS_D0 = False`, `Depth = 1–4`)** because of post-PKS tailoring;
2. **Solved with `exact_stereo_and_constitution` (`100%` 3D stereochemical + constitutional rebuild accuracy)** by our Two-Stage Engine; and
3. **100% Wet-Lab Assemblable (`allow_gene_splitting=False`, exact `KR` subtype match)** using intact in-house *Streptomyces* / actinomycete T1PKS genes from `/data/lukecarlate/MITE_augmentation/deterministic_stage/inhouse_antismash_t1pks_genes.tsv`.

| # | Target & Role | MW (Da) | Peeling Depth & Tailoring Enzymes (`Stage 1`) | Reconstructed Assembly Line (`Stage 2`) | Strictly In-House Genes (`<= 9.0 kb`, Max 2-Mod) | Strictly In-House Genes (`<= 6.0 kb`, Fits `3m15s` Q5 PCR + `pJET1.2`!) |
|---:|---|---:|---|---|---|---|
| **1** | **Protulactone B** (`NPA006758`, *Orphan Fungal Iterative PKS $\to$ Bacterial Modular, 4 tailoring steps*) | **230.2** | **Depth 4:** `R5 (BVMO)` $\to$ `R3 (P450 OH)` $\to$ `R32 (Cyclic ether)` $\to$ `R3 (P450 OH)` | `M0:mal` $\to$ `M1:emal(β-keto)` $\to$ `M2:mal(DH-ER-KR)` $\to$ `M3:mal(KR)-TE` | **4 Genes (`3.0–5.0 kb`):** `BR10-15::ctg4_248` (`3.0kb`) + `CW10-3::ctg3_292` (`3.0kb`) + `BR10-2::ctg1_6378` (`5.0kb`) + `CW21-3::ctg1_6510` (`4.3kb`) | **Same 4 Genes (`Max 5.0 kb`!)** — Every insert is `3.0–5.0 kb` (`<= 6 kb` Q5 PCR window) |
| **2** | **Xylariolide C** (`NPA005116`, *Orphan Fungal Iterative PKS $\to$ Bacterial Modular, 1-step cyclic ether*) | **232.3** | **Depth 1:** `R32` (`MonBI/Lsd19` cyclic ether unzipping) | `M0:mal` $\to$ `M1:mmal(KR)` $\to$ `M2:mmal(DH-KR)` $\to$ `M3:mmal(KR-A2)` | **3 Genes (`3.0–7.5 kb`):** `BR10-15::ctg4_248` (`3.0kb`) + `Bu2-5::ctg1_525` (`3.5kb`) + `Bu2-16::ctg4_6793` `[M2..M3]` (`7.5kb`) | **4 Genes (`Max 4.0 kb`!):** `BR10-15::ctg4_248` (`3.0kb`) + `Bu2-5::ctg1_525` (`3.5kb`) + `BR20-6::ctg1_1291` (`4.0kb`) + `Bu5-1::ctg1_168` (`3.5kb`) |
| **3** | **Botryolide B** (`NPA005290`, *Orphan Fungal Iterative PKS $\to$ Bacterial Modular, 1-step epoxide*) | **198.2** | **Depth 1:** `R4` (`AbyE/EpoK` P450 alkene epoxidation) | `M0:mal` $\to$ `M1:mmal(KR-B1)` $\to$ `M2..M3:mal(DH-KR)` $\to$ `M4:mal(KR-B1)-TE` | **4 Genes (`3.0–8.1 kb`):** `BR10-15::ctg4_248` (`3.0kb`) + `KA1-1::ctg1_300` (`3.5kb`) + `BR26-2::ctg1_1958` `[M2..M3]` (`8.1kb`) + `CW21-3::ctg1_6510` (`4.3kb`) | **5 Genes (`Max 4.3 kb`!):** `BR10-15::ctg4_248` (`3.0kb`) + `KA1-1::ctg1_300` (`3.5kb`) + `2× BR10-2::ctg1_6380` (`4.0kb`) + `CW21-3::ctg1_6510` (`4.3kb`) |
| **4** | **Phomolide A** (`NPA007033`, *Orphan Fungal Iterative PKS $\to$ Bacterial Modular, 10-ring epoxy-macrolide*) | **208.3** | **Depth 1:** `R4` (`AbyE/EpoK` P450 alkene epoxidation) | `M0:mal` $\to$ `M1:mal(DH-ER-KR)` $\to$ `M2:mal(KR-A1)` $\to$ `M3..M5:mal(DH-KR)-TE` | **3 Genes (`7.5–8.9 kb`):** `GN6-9::ctg4_4598` `[M0..M1]` (`8.0kb`) + `Bu8-19::ctg1_1391` `[M2..M3]` (`7.5kb`) + `CW14-6::ctg2_7195` `[M4..M5-TE]` (`8.9kb`) | **6 Genes (`Max 5.0 kb`!):** `BR10-15::ctg4_248` (`3.0kb`) + `BR10-2::ctg1_6378` (`5.0kb`) + `BR5-1::ctg1_3018` (`3.5kb`) + `2× BR10-2::ctg1_6380` (`4.0kb`) + `Bu5-1::ctg1_177` (`4.8kb`) |
| **5** | **Acremolide C** (`NPA007544`, *Orphan Hybrid PKS-NRPS lipopeptide-ketide, 4 steps*) | **468.6** | **Depth 4:** `R9 (C-MT)` $\to$ `R3 (P450 OH)` $\to$ `R24 (NRPS Pro)` $\to$ `R20 (NRPS AA)` | `M0:mal` $\to$ `M1..M5` (5 PKS) $\to$ `M6..M7` (2 NRPS) | **5 PKS Genes (`3.0–7.5 kb`):** `BR10-15::ctg4_248` `[M0]` + `Bu3-20::ctg1_3321` `[M1..M2]` + `KB3-5::ctg2_6069` `[M3]` + `UN44-4::ctg1_11498` `[M4]` + `UN11-4::ctg2_763` `[M5]` | **6 PKS Genes (`Max 5.0 kb`!)** |
| **6** | **Bu-4514N** (`LOTUS:Q77506551`, *Orphan Bacterial Hybrid PKS-NRPS antibiotic, 4 steps*) | **498.6** | **Depth 4:** `R0 (IMDA)` $\to$ `R22 (Tetramate Dieckmann)` $\to$ `R13 (Deoxysugar)` $\to$ `R20 (NRPS Gly)` | `M0:mal` $\to$ `M1..M7` (7 PKS) $\to$ `M8: C-A(Gly)-PCP-R0` (1 NRPS) | **5 PKS Genes (`3.0–9.0 kb`):** `BR10-15::ctg4_248` `[M0]` + `CP984_RS01405` `[M1..M2]` + `SAVERM_RS15025` `[M3..M4]` + `BR20-8::ctg1_9596` `[M5..M6]` + `KCCM40584::ctg1_428` `[M7]` | **8 PKS Genes (`Max 5.0 kb`!)** |
| **7** | **Pravastatin** (`C23H36O7`, *PI API Target: Fungal Iterative $\to$ Bacterial Modular PKS*) | **424.5** | **Depth 2:** `R11 (LovD Ester cleavage)` $\to$ `R0 (SpnF candidate retro-[4+2] carbocyclization)` | `M0:mal` $\to$ `M1..M2:mal(DH-KR)` $\to$ **`M3:hmal(DH-KR)`** $\to$ **`M4:hmal(DH-ER-KR)`** $\to$ `M5:mal(DH-KR)` $\to$ `M6:mal(DH-ER-KR)` $\to$ `M7:mal(KR-B1)` $\to$ `M8:mal(KR-B1)-TE` | **6 Genes (`3.5–9.0 kb`):** `BR6-3::ctg2_654` `[M0..M1]` (`7.0kb`) + `BR10-2::ctg1_6380` `[M2]` (`4.0kb`) + **`SNU6-10::ctg1_7337` `[M3..M4]` (`9.0kb`, in-house `FkbH` `hmal` BGC!)** + `SNU6-10::ctg1_7107` `[M5..M6]` (`9.0kb`) + `UN37-2::ctg1_1806` `[M7]` (`3.5kb`) + `In5-9::ctg1_1002` `[M8-TE]` (`4.3kb`) | **9 Genes (`Max 5.0 kb`!)** |




