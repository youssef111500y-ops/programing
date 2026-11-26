# Dataset Inventory
## Cerebral Aneurysm AI Research

Last reviewed: 2026-11-26

## 1. AneuX Morphology Database

Source:
https://zenodo.org/records/6678442

Repository:
https://github.com/hirsch-lab/aneuxdb

Reported contents:
- 750 aneurysm dome surface meshes
- 668 vessel tree surface meshes
- 5 clinical parameters
- 170 pre-computed morphometric indices

License:
CC BY-NC 4.0.

Planned use:
- Morphological analysis
- Rupture-status classification
- 3D geometry processing
- Point-cloud experiments

Important:
This dataset does not establish longitudinal patient-level
aneurysm growth trajectories.

## 2. AneuG-Flow

Paper:
https://proceedings.neurips.cc/paper_files/paper/2025/hash/e2b8ff0035bc9f572a7deefbcbea85bc-Abstract-Datasets_and_Benchmarks_Track.html

Reported contents:
- 14,000 steady-flow cases
- 200 pulsatile-flow cases
- Velocity, pressure and WSS
- Surface meshes and volume data

Planned use:
- Hemodynamic prediction
- Neural operator experiments
- Flow simulation validation

Access and license:
Must be verified before downloading or redistributing.

## 3. Aneumo

Paper:
https://arxiv.org/abs/2505.14717

Repository:
https://github.com/Xigui-Li/Aneumo

Reported contents:
- 427 real aneurysm geometries
- 10,660 synthetic 3D shapes
- 85,280 CFD simulations
- Segmentation masks

Planned use:
- 3D geometry experiments
- Hemodynamic learning
- Synthetic geometry evaluation

Important:
Synthetic geometric evolution must not be treated
as real longitudinal patient follow-up.

## 4. Recent Research Baseline (2026)

Paper:
https://doi.org/10.1371/journal.pone.0360923

Reported:
- 708 patient-specific aneurysms
- 55 clinical, morphological and hemodynamic predictors
- Independent test ROC-AUC around 0.78
  for the best reported classifier

Planned use:
- Reproducibility benchmark
- Feature comparison
- Experimental baseline
- Research gap identification

## 5. Research Decisions

1. Do not merge datasets before checking case overlap.
2. Keep raw datasets outside Git.
3. Record dataset versions and checksums.
4. Verify licenses before use.
5. Prevent patient-level data leakage.
6. Separate retrospective rupture classification
   from prospective future rupture prediction.
7. Do not treat pulsatile flow as longitudinal
   aneurysm progression.
