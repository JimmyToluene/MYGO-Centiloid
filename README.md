<p align="center">
  <img src="./figures/logos/mygo_centiloid_logo.svg" alt="MYGO-Centiloid Logo" width="200" style="margin-bottom: -20px;"/>
</p>

<h1 align="center">MYGO-Centiloid</h1>
<p align="center">
  <b>M</b>ultitracer-conditioned 3D ResNet18 for am<b>Y</b>loid β-PET centiloid re<b>G</b>ressi<b>O</b>n
</p>

<p align="center">
   <b>1st prize</b> at the
  <a href="https://medaihack.org/"><b>MedAI Spring 2026 Hackathon</b></a>
  organized by the
  <a href="https://github.com/vkola-lab/medaihack"><b>Kolachalama Lab, Boston University</b></a>.
</p>

<p align="center">
  <b>Team 25 — It's MYGO!!!!!!</b><br/>
  <a href="https://github.com/JimmyToluene">Jimmy Jia</a> ·
  <a href="https://github.com/Yujie-Jessie">Yujie Hu</a> ·
  <a href="https://github.com/ayiii-a">Zijiang Zhao</a> ·
  <a href="https://github.com/karthikayanidevaraj">Karthikayani Devaraj</a> ·
  <a href="https://github.com/">Shruthi Ashok</a> ·
  <a href="https://github.com/">Member 6</a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.8-ee4c2c.svg" alt="PyTorch"/></a>
  <a href="https://medaihack.org/"><img src="https://img.shields.io/badge/MedAI%20Spring%202026-1st%20place-gold.svg" alt="1st place"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"/></a>
</p>

---

We predict continuous Centiloid scores from preprocessed 3D amyloid β-PET
volumes (`(1, 128, 128, 128)`, four tracers: **FBP**, **FBB**, **NAV**, **PIB**),
trained on the MedAI Spring 2026 Hackathon data (2,000 train + 500 val,
NACC + A4 cohorts).

The pipeline is specifically designed to handle the
extreme right-skew and 64.8 % negative-class imbalance in the Centiloid
distribution.

> **Val set:** MAE **11.73 CL** Pearson r **0.936** — a **40.7 %** MAE reduction over the starter baseline (19.77 CL).

> **Note — public review version.** This release is intended for review of
> the hackathon submission's outputs and inference behavior. Model
> internals, training scripts, hyperparameters, and ongoing ablation
> studies are intentionally not included here; they are being prepared
> as a separate technical report. Please contact the authors for
> research collaboration.

---

## Contents

1. [Results](#results)
2. [Quick start](#quick-start)
3. [Repository Structure](#repository-structure)
4. [Data](#data)
5. [Outputs](#outputs)
6. [Disclaimer](#disclaimer)
7. [License](#license)
8. [References](#references)

---

## Results

We compared MYGO-Centiloid against the unmodified starter baseline on
the validation set (n = 500).

| | 3D CNN baseline | **MYGO (ours)** |
|---|-----------------|---|
| **Overall MAE** | 19.77 CL        | **11.73 CL** |
| **Overall Pearson r** | 0.790           | **0.936** |

**Per-tracer breakdown:**

| Tracer | N | Baseline 3D CNN MAE | **MYGO MAE** | Baseline 3D CNN r | **MYGO r** |
| ------ | --- |---------------------| ------------ |-------------------| ---------- |
| **ALL** | 500 | 19.77               | **11.73** | 0.790             | **0.936** |
| FBP | 236 | 19.28               | **11.49** | 0.797             | **0.930** |
| FBB | 114 | 20.04               | **12.37** | 0.804             | **0.933** |
| PIB | 133 | 21.17               | **11.94** | 0.790             | **0.939** |
| NAV | 17  | 13.86               | **9.28** | 0.946             | **0.981** |

**Improvement:** MAE 19.77 → 11.73 (−8.04 CL, **40.7 % reduction**); Pearson r 0.790 → 0.936.

---

## Quick start

This public-review build supports **inference only**. Reviewers can
reproduce the validation-set numbers above using the released checkpoint.

### Environment Setup

#### BU SCC

```bash
module load medaihack/spring-2026
module load python3/3.12.4

# Create venv (one-time)
virtualenv /projectnb/medaihack/team25/<your_venv_name>
source /projectnb/medaihack/team25/<your_venv_name>/bin/activate

cd /projectnb/medaihack/team25/MYGO-Centiloid
pip install -r requirements.txt
pip install -e .
```

For **OnDemand** (Jupyter / Code Server): load the two modules in the
module list and place the `source` command in the pre-launch dialog box.

#### Outside BU SCC

```bash
git clone https://github.com/<your-user>/mygo-centiloid.git
cd mygo-centiloid
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

### Running inference

```bash
# 1. Link data (BU SCC — one-liner, no copy)
ln -s /projectnb/medaihack/ABPET/data data

# 2. Predict → Evaluate
bash predict.sh data/val.csv                                   # writes predictions.csv
python dev/evaluate.py --pred predictions.csv --gt data/val.csv
```

Outside BU SCC, place the dataset at `data/` (or symlink):
```
data/
├── val.csv
└── npy_files/
```

---

## Repository Structure

```text
MYGO-Centiloid/
├── mygo_centiloid/                # Installable Python package (inference)
│   ├── model/                         model definition
│   ├── losses/                        loss utilities
│   └── data/                          PETDataset, transforms
│
├── dev/                          # Inference + evaluation entry points
│   ├── predict.py                     inference → predictions.csv
│   ├── evaluate.py                    MAE / RMSE / Pearson r report
│   └── config/
│       └── predict.yaml               inference paths + batch settings
│
├── eda/                          # Exploratory data analysis (see eda/README.md)
│   └── 01-04_*.py                     data + prediction analysis scripts
│
├── checkpoints/                  # released model weights (best_model.pt)
├── figures/                      # logo
│
├── predict.sh                    # reviewer / judge entry point
├── setup.py                      # pip install -e .
├── requirements.txt
└── README.md
```

After `pip install -e .`:

```python
from mygo_centiloid import PETDataset
```

---

## Data

| Split | Cohorts | N | Breakdown |
|-------|---------|---|-----------|
| Train | NACC + A4 | 2,000 | 1,195 NACC + 805 A4 |
| Val   | NACC + A4 | 500   | 305 NACC + 195 A4 |

Each sample is a preprocessed `.npy` volume with an associated Centiloid score and tracer label.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `ID` | str | Subject identifier |
| `npy_path` | str | Path to `(1, 128, 128, 128)` float32 `.npy`, range `[0, 1]` |
| `CENTILOIDS` | float | Regression target (typically −50 to 200+) |
| `TRACER.AMY` | str | Radiotracer: `FBP`, `FBB`, `NAV`, `PIB` |

### Tracers in the dataset

| Code | Full name | N (train+val) |
|------|-----------|---------------|
| `FBP` | Florbetapir | 1,182 |
| `FBB` | Florbetaben | 568 |
| `NAV` | Florbetanav | 85 |
| `PIB` | Pittsburgh Compound B | 665 |

Each tracer binds to amyloid with different affinity and produces different
uptake patterns. The Centiloid scale harmonizes across tracers.

### Preprocessing already applied

All images were preprocessed from raw NIfTI PET scans (we did **not** redo
any of these). The following steps were applied in order:

1. **Channel first** — `(C, H, W, D)` format.
2. **RAS orientation** — standard neuroimaging alignment.
3. **Isotropic resampling** — 2 mm × 2 mm × 2 mm, trilinear.
4. **Foreground cropping** — 10-voxel margin via MONAI `CropForeground`.
5. **Resize** — `128 × 128 × 128`, trilinear.
6. **Spatial padding** — to exactly 128³ if needed.
7. **Dynamic frame averaging** — multi-frame PET → single static volume.
8. **Shape enforcement** — final center-crop/pad to `(1, 128, 128, 128)`.
9. **Min-max normalization** — `img = (img - img.min()) / (img.max() - img.min())`.

---

## Outputs

```
predictions.csv                           from predict.sh / dev/predict.py

checkpoints/
└── best_model.pt                         released checkpoint (lowest val MAE)
```

---

## Disclaimer

This software and any trained model weights distributed with it are
provided **for academic and research purposes only**. They are **not a
medical device** and have **not** been validated or approved by the
U.S. Food and Drug Administration (FDA), the European Medicines Agency
(EMA), or any other regulatory body.

**The model and its inferences must not be used to inform clinical
diagnosis, treatment decisions, prognosis, or any patient-care
workflow.** The training data (2,000 hackathon samples across four
tracers) is too small and too narrow to support any clinical claim, and
the model has undergone no prospective or external validation.

Any use of this software or its outputs in a clinical setting is the
sole responsibility of the user.

---

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE) for the
full text, including the research-use-only notice.

All code in `mygo_centiloid/` and `dev/` is original to this project; no third-party
code carrying a copyleft or non-commercial license was incorporated.

---

## References

General references for the prior work and datasets that informed this project.

### Architecture and training (general)

1. **He K, Zhang X, Ren S, Sun J.** Deep Residual Learning for Image
   Recognition. *CVPR* 2016.
   [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
2. **Hara K, Kataoka H, Satoh Y.** Can Spatiotemporal 3D CNNs Retrace the
   History of 2D CNNs and ImageNet? *CVPR* 2018.
   [arXiv:1711.09577](https://arxiv.org/abs/1711.09577)

### Medical-imaging augmentation

3. **Pérez-García F, Sparks R, Ourselin S.** TorchIO: a Python library
   for efficient loading, preprocessing, augmentation and patch-based
   sampling of medical images in deep learning. *Computer Methods and
   Programs in Biomedicine* 2021;208:106236.
   [doi:10.1016/j.cmpb.2021.106236](https://doi.org/10.1016/j.cmpb.2021.106236)

### Domain — amyloid PET and the Centiloid scale

4. **Klunk WE, et al.** The Centiloid Project: standardizing
   quantitative amyloid plaque estimation by PET. *Alzheimer's &
   Dementia* 2015;11(1):1–15.
   [doi:10.1016/j.jalz.2014.07.003](https://doi.org/10.1016/j.jalz.2014.07.003)
5. **Jagust WJ, et al.** The Alzheimer's Disease Neuroimaging Initiative
   2 PET Core: 2015. *Alzheimer's & Dementia* 2015;11(7):757–771.
   [doi:10.1016/j.jalz.2015.05.001](https://doi.org/10.1016/j.jalz.2015.05.001)

### Related Kolachalama Lab work

6. **Qiu S, et al.** Multimodal deep learning for Alzheimer's disease
   dementia assessment. *Nature Communications* 2022;13:3404.
   [doi:10.1038/s41467-022-31037-5](https://doi.org/10.1038/s41467-022-31037-5)
   · [vkola-lab/ncomms2022](https://github.com/vkola-lab/ncomms2022)
7. **Kolachalama Lab.** AI-driven fusion of multimodal data for
   Alzheimer's disease biomarker assessment. *Nature Communications*
   2025.
   [doi:10.1038/s41467-025-62590-4](https://doi.org/10.1038/s41467-025-62590-4)
   · [vkola-lab/ncomms2025](https://github.com/vkola-lab/ncomms2025)
