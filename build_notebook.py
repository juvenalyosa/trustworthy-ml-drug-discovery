"""Construye el notebook educativo (Colab) sobre ML confiable en datos de cinasas.

Se genera con nbformat en vez de escribir JSON a mano para garantizar un .ipynb válido.
Cada celda se añade con md(...) o code(...), en orden.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ════════════════════════════════════════════════════════════════════ PART 0
md(r'''
# Trustworthy Machine Learning for Drug Discovery

### Predicting kinase inhibitor potency — and knowing when to believe your model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/trustworthy-ml-drug-discovery/blob/main/Trustworthy_ML_for_Drug_Discovery.ipynb)

---

A model that reports $R^2 = 0.95$ and a model that reports $R^2 = 0.60$ can describe the
same data. One of them is lying to you. By the end of this notebook you will know which,
and why — and you will have built the honest one.

We work with real measurements of inhibitors against three cancer drug targets:

| Target | Disease relevance | Famous drugs |
|---|---|---|
| **EGFR** | Non-small-cell lung cancer | gefitinib, erlotinib, osimertinib |
| **ALK** | ALK-positive lung cancer | crizotinib, alectinib, lorlatinib |
| **BRAF** | Melanoma | vemurafenib, dabrafenib |

Each target also appears as **mutated variants** — the versions that emerge in patients after
treatment and stop responding to the drug. Predicting potency against those variants is the
practical goal.

### What you will learn

1. **When machine learning is the right tool here, and when it is not**
2. **Why potency is modeled on a logarithmic scale** — and the thermodynamics behind it
3. **How to curate assay data** without silently corrupting it
4. **How to compute the ceiling on model performance** before training anything
5. **How molecules become numbers** (molecular fingerprints)
6. **How to split data so your error estimate is honest** — the single most consequential choice
7. **How gradient boosting works**, from the loss function up
8. **Which metrics to report**, and what each one hides
9. **How to prove a feature earns its place** using negative controls
10. **A reusable protocol** for trustworthy modeling

A workshop and an exam close the notebook.

> **Prerequisites:** basic Python, basic statistics. No chemistry background required —
> every chemical idea is built up from scratch.
''')

md(r'''
## Part 0 — Setup

Run the next two cells once. The first installs the chemistry toolkit; the second defines a
consistent visual style so every figure in the notebook reads the same way.
''')

code(r'''
# RDKit = the standard open-source cheminformatics library. ~20 s in Colab.
try:
    import rdkit
    print("RDKit already present:", rdkit.__version__)
except ImportError:
    !pip install -q rdkit
    import rdkit
    print("RDKit installed:", rdkit.__version__)
''')

code(r'''
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

# ── A deliberately small, validated palette ────────────────────────────────────
# Categorical hues are assigned in FIXED order and never cycled. Three slots only:
# beyond three, colours stop being reliably distinguishable for colour-blind readers.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, GRAY = "#e34948", "#9a9892"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8982"
SURFACE, GRID = "#fcfcfb", "#e6e5e0"
SEQ = ["#d6e4f7", "#a8c8ef", "#6ba3e3", "#2a78d6", "#1a4e8f"]  # one hue, light→dark

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "figure.dpi": 110,
    "font.size": 10.5, "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK3, "ytick.color": INK3,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "legend.fontsize": 9.5,
    "lines.linewidth": 2.0, "lines.markersize": 5,
})


def clean(ax, grid="y"):
    """Remove chart junk: top/right spines off, grid on one axis only."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    if grid == "y":
        ax.grid(axis="x", visible=False)
    else:
        ax.grid(axis="y", visible=False)
    return ax


def note(ax, text, xy=(0.98, 0.04), ha="right"):
    ax.text(*xy, text, transform=ax.transAxes, ha=ha, va="bottom",
            fontsize=9, color=INK2, linespacing=1.4)


np.random.seed(0)
print("Plot style ready.")
''')

md(r'''
### The dataset

The notebook runs in two modes:

- **Real mode** — if `kinase_ml_dataset.csv` is available (upload it, or mount Drive), you work
  with 18,145 curated measurements of 13,427 distinct molecules.
- **Teaching mode** — otherwise a *surrogate* dataset is simulated. It is built to reproduce the
  same statistical phenomena as the real data (scaffold structure, label noise, a size-biased
  docking score), so every lesson below behaves identically. Because we choose its ground truth,
  it is actually *better* for proving two of the concepts.

Every result in this notebook is reproduced in both modes.
''')

code(r'''
REAL_COLUMNS = ["mol_id", "smiles_std", "target", "mutation", "variant_id",
                "assay_category", "pIC50", "dock_best", "heavy_atoms", "scaffold"]


def make_surrogate(n_scaffolds=300, noise_sd=0.75, seed=0, motif_sd=0.55,
                   sub_sd=0.26, scaf_sd=0.25, n_rules=14, rule_sd=0.95, base=5.2):
    """Simulated potency data built to reproduce the real dataset's statistical structure.

    The key design choice is that potency is driven mostly by SHARED substructures, so
    knowledge transfers between chemical families. Had each scaffold carried its own
    unrelated effect, a scaffold-split model could learn nothing at all (R2 ~ 0) and the
    central lesson of Part 6 would be impossible to demonstrate.

      motifs       ring systems reused across scaffolds -> effects TRANSFER across families
      substituents decorations on the core             -> effects transfer too
      scaffold     a small family-specific residual    -> learnable ONLY within a family
      rules        "motif M together with substituent S" -> a logical AND, i.e. a genuine
                   interaction that a linear model cannot represent
      variant      per-mutation offset
      noise        measurement error, sd = noise_sd
      cliffs       4% of molecules jump 1.2-2.2 log units with almost no structural change
      docking      a score driven by SIZE, not potency -> a plausible but useless feature

    Calibrated so the random/scaffold gap (~0.10) and the scaffold-split R2 (~0.6) land
    close to what the real 18,145-measurement dataset gives.
    """
    rng = np.random.default_rng(seed)
    n_bits, n_motifs, n_sub = 256, 30, 70
    motif_bits = [rng.choice(n_bits, int(rng.integers(4, 11)), replace=False) for _ in range(n_motifs)]
    # bigger ring systems make slightly more contacts, hence a mild size-potency link
    motif_eff = rng.normal(0.10, motif_sd, n_motifs) + 0.035 * np.array([len(b) for b in motif_bits])
    sub_bits = rng.choice(n_bits, n_sub, replace=False)
    sub_eff = rng.normal(0.13, sub_sd, n_sub)
    # context-dependent SAR: a substituent that helps on one ring system does nothing on another
    rules = [(int(rng.integers(n_motifs)), int(rng.integers(n_sub)), float(rng.normal(0, rule_sd)))
             for _ in range(n_rules)]
    scaf_motifs = [rng.choice(n_motifs, int(rng.integers(2, 4)), replace=False) for _ in range(n_scaffolds)]
    scaf_resid = rng.normal(0, scaf_sd, n_scaffolds)
    VARIANTS = {"EGFR": [("Wild Type", 0.0), ("L858R", 0.30), ("T790M", -0.25), ("L858R, T790M", 0.45)],
                "BRAF": [("Wild Type", 0.0), ("V600E", 0.40)],
                "ALK":  [("Wild Type", 0.0), ("L1196M", -0.27)]}

    rows, FP = [], []
    for s in range(n_scaffolds):
        for _ in range(int(rng.integers(3, 13))):
            fp = np.zeros(n_bits, np.uint8)
            for m in scaf_motifs[s]:
                fp[motif_bits[m]] = 1
            picks = rng.choice(n_sub, int(rng.integers(2, 13)), replace=False)
            fp[sub_bits[picks]] = 1
            true = (base + motif_eff[scaf_motifs[s]].sum() + sub_eff[picks].sum() + scaf_resid[s]
                    + sum(e for mi, si, e in rules if mi in scaf_motifs[s] and si in picks))
            if rng.random() < 0.04:                               # activity cliff
                true += rng.choice([-1.0, 1.0]) * rng.uniform(1.2, 2.2)
            heavy = int(np.clip(14 + 0.8 * sum(len(motif_bits[m]) for m in scaf_motifs[s])
                                + 2.6 * len(picks) + rng.normal(0, 2.5), 12, 80))
            tgt = rng.choice(["EGFR", "BRAF", "ALK"], p=[0.6, 0.25, 0.15])
            mol = f"M{len(FP):05d}"
            # the same molecule is often measured against several variants, as in real data
            n_meas = int(rng.integers(1, min(3, len(VARIANTS[tgt])) + 1))
            for vi in rng.choice(len(VARIANTS[tgt]), n_meas, replace=False):
                vname, veff = VARIANTS[tgt][vi]
                rows.append(dict(
                    mol_id=mol, smiles_std="", target=tgt, mutation=vname,
                    variant_id=f"{tgt}|{vname.replace(', ', '+')}",
                    assay_category=rng.choice(["biochemical", "cellular", "other"], p=[.56, .27, .17]),
                    pIC50=float(np.clip(true + veff + rng.normal(0, noise_sd), 3, 12)),
                    pIC50_true=true + veff,                       # only the surrogate has this
                    dock_best=-(6.0 + 0.085 * heavy + rng.normal(0, 0.55)),
                    heavy_atoms=heavy, scaffold=f"S{s:04d}", _row=len(FP)))
            FP.append(fp)
    d = pd.DataFrame(rows)
    return d.drop(columns="_row"), np.array(FP, dtype=np.uint8)[d._row.to_numpy()]


RAW_DATA = ("https://raw.githubusercontent.com/juvenalyosa/"
            "trustworthy-ml-drug-discovery/main/kinase_ml_dataset.csv")


def load_dataset():
    """Real data if we can find it, surrogate otherwise."""
    import os
    import urllib.request
    paths = ["kinase_ml_dataset.csv", "/content/kinase_ml_dataset.csv",
             "/content/drive/MyDrive/kinase_ml_dataset.csv"]
    if not any(os.path.exists(p) for p in paths):
        # Opened straight from GitHub, Colab loads only the notebook: fetch the data too.
        try:
            print("downloading the curated dataset (6 MB) ...", flush=True)
            urllib.request.urlretrieve(RAW_DATA, "kinase_ml_dataset.csv")
        except Exception as err:
            print(f"download failed ({type(err).__name__}) — using the surrogate instead")
    for p in paths:
        if os.path.exists(p):
            d = pd.read_csv(p)
            from rdkit import Chem, RDLogger
            from rdkit.Chem import rdFingerprintGenerator
            RDLogger.DisableLog("rdApp.*")
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            uniq = d.drop_duplicates("mol_id").set_index("mol_id")["smiles_std"]
            fps = {m: gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for m, s in uniq.items()}
            FP = np.array([fps[m] for m in d["mol_id"]], dtype=np.uint8)
            print(f"REAL mode — {len(d):,} measurements, {d.mol_id.nunique():,} molecules")
            return d, FP, True
    d, FP = make_surrogate()
    print(f"TEACHING mode — simulated surrogate: {len(d):,} measurements, "
          f"{d.mol_id.nunique():,} molecules, {d.scaffold.nunique()} scaffolds")
    return d, FP, False


df, FP, IS_REAL = load_dataset()
df.head(3)
''')

# ════════════════════════════════════════════════════════════════════ PART 1
md(r'''
---
## Part 1 — Why machine learning for this problem?

### The size of the search space

The number of drug-like molecules that could exist is estimated at $10^{60}$. For scale, there
are about $10^{57}$ atoms in the solar system. A well-funded medicinal chemistry team might
synthesize and test a few hundred compounds per year.

So the real task is never "search the space". It is: **given the few thousand compounds someone
has already measured, decide which handful to make next.** That is a prediction problem with
labelled examples, which is exactly what supervised learning is for.

### What the model actually does

A chemist developing an inhibitor series works like someone learning a city by walking it. After
enough trips they can guess that a street two blocks over probably also leads downtown — but
only near the neighbourhoods they have walked. Drop them in a different city and the intuition
is worthless.

A trained model has exactly this property, and it matters enormously for how we evaluate it:
interpolation inside the chemistry you have sampled is reliable, extrapolation to new chemistry
is not. **Part 6 is entirely about measuring the difference honestly.**

### When ML is the wrong tool here

| Situation | Why it fails | What to do instead |
|---|---|---|
| Fewer than ~100 measurements | Nothing to learn from | Medicinal-chemistry intuition, free-energy calculations |
| Labels from incompatible assays | You model assay differences, not chemistry | Curate to one assay type first |
| A chemical class absent from training | Pure extrapolation | Measure a few members first |
| You need a mechanism, not a number | Models give correlation | Structural biology, MD simulation |
| The decision tolerates no error | Typical error is 5–10× in potency | Measure it |

Machine learning is a prioritisation tool here. It ranks what to test. It does not replace the
experiment.
''')

code(r'''
# How much of chemical space does a project actually touch?
fig, ax = plt.subplots(figsize=(7.2, 4.0))
rng = np.random.default_rng(3)
ax.scatter(*rng.normal(0, 1.0, (2, 2500)), s=6, c=GRID, alpha=.9, lw=0, label="unexplored chemistry")
for i, (cx, cy) in enumerate([(-.75, .45), (.85, -.3), (.1, 1.0)]):
    p = rng.normal([cx, cy], .17, (70, 2))
    ax.scatter(p[:, 0], p[:, 1], s=14, c=[BLUE, ORANGE, AQUA][i], lw=0,
               label=f"measured series {i+1}" if i == 0 else None)
    ax.add_patch(Circle((cx, cy), .40, fill=False, ec=[BLUE, ORANGE, AQUA][i], lw=1.4, alpha=.55))
ax.scatter([], [], s=14, c=ORANGE, lw=0)
ax.set(xticks=[], yticks=[], title="A project samples a few islands, not the ocean")
ax.set_xlabel("chemical space (schematic, 2 of ~1000 dimensions)")
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)
ax.grid(False)
ax.legend(loc="upper left", markerscale=1.8)
note(ax, "A model trained on the coloured islands\nis trustworthy near them — and guessing elsewhere.")
plt.tight_layout(); plt.show()
''')

# ════════════════════════════════════════════════════════════════════ PART 2
md(r'''
---
## Part 2 — The target variable: why $\mathrm{pIC_{50}}$ and not $\mathrm{IC_{50}}$

### What is measured

The **$\mathrm{IC_{50}}$** is the concentration of inhibitor that reduces enzyme activity by half.
*Lower is better*: 1 nM is a superb drug candidate, 10,000 nM is nearly inactive.

### The transformation

$$\mathrm{pIC_{50}} \;=\; -\log_{10}\!\big(\mathrm{IC_{50}}\,[\mathrm{M}]\big)
\;=\; 9 - \log_{10}\!\big(\mathrm{IC_{50}}\,[\mathrm{nM}]\big)$$

Now *higher is better*, and the scale is compact: $\mathrm{pIC_{50}} = 9$ means 1 nM,
$\mathrm{pIC_{50}} = 6$ means 1000 nM.

### Reason 1 — it is linear in binding free energy

Inhibition reflects an equilibrium between inhibitor $I$ and enzyme $E$:

$$E + I \rightleftharpoons EI, \qquad K_i = \frac{[E][I]}{[EI]}$$

and thermodynamics links that constant to free energy:

$$\Delta G^\circ = -RT\ln K \;=\; -2.303\,RT\log_{10} K$$

Since $\mathrm{IC_{50}}$ is proportional to $K_i$ under standard assay conditions:

$$\boxed{\;\Delta G^\circ \;\propto\; -\mathrm{pIC_{50}}\;}$$

At 300 K, $2.303\,RT \approx 5.74$ kJ/mol, so **one $\mathrm{pIC_{50}}$ unit $\approx$ 5.7 kJ/mol**
of binding energy — roughly one good hydrogen bond. The logarithm is not a statistical
convenience; it is the scale on which the physics is additive. Adding a methyl group that
contributes a fixed amount of binding energy shifts $\mathrm{pIC_{50}}$ by a fixed amount, at any
potency level. On the raw $\mathrm{IC_{50}}$ scale that same chemical change moves the number by
10 nM in one series and 10,000 nM in another.

### Reason 2 — it fixes the error structure

Assay error is *multiplicative*: a measurement is reported as "within 2-fold", never "within
20 nM". Under the logarithm, multiplicative error becomes additive and roughly symmetric, which
is what least-squares regression assumes.

### Reason 3 — it stops the loss function being hijacked

Squared error on raw $\mathrm{IC_{50}}$ makes one inactive compound at 100,000 nM contribute more
to the loss than a thousand potent compounds combined. The model would spend all its capacity on
the compounds you care least about.
''')

code(r'''
ic50 = 10 ** (9 - df.pIC50.values)   # back to nM for illustration
fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))

ax = axes[0]
ax.hist(np.clip(ic50, 0, 20000), bins=70, color=ORANGE, lw=0)
clean(ax); ax.set(title="Raw IC$_{50}$ — unusable", xlabel="IC$_{50}$ (nM)", ylabel="compounds")
note(ax, "Crushed against zero, long right tail.\nSquared error here is dominated\nby the weakest compounds.")

ax = axes[1]
ax.hist(df.pIC50, bins=70, color=BLUE, lw=0)
clean(ax); ax.set(title="pIC$_{50}$ — well behaved", xlabel="pIC$_{50}$", ylabel="compounds")
sec = ax.secondary_xaxis("top", functions=(lambda p: -5.74 * p, lambda g: -g / 5.74))
sec.set_xlabel("relative binding free energy ΔG° (kJ/mol)", fontsize=9.5, color=INK2)
sec.tick_params(colors=INK3, labelcolor=INK2)
note(ax, "Roughly symmetric, ~3 log units wide.\nEqual steps = equal energy.", xy=(0.02, 0.60), ha="left")

plt.tight_layout(); plt.show()
print(f"IC50  (nM) : median {np.median(ic50):8.1f}   min {ic50.min():.2g}   max {ic50.max():.3g}")
print(f"pIC50      : median {df.pIC50.median():8.2f}   min {df.pIC50.min():.2f}   max {df.pIC50.max():.2f}")
''')

md(r'''
> **The secondary axis above is a labelling device, not a second data series.** A chart must
> never carry two *independent* y-scales — that is the most common way to mislead with a figure.
> Here both axes measure the same quantity in different units, which is legitimate.
''')

# ════════════════════════════════════════════════════════════════════ PART 3
md(r'''
---
## Part 3 — Data curation: four ways to corrupt a dataset silently

Public bioactivity databases such as ChEMBL aggregate numbers from thousands of papers. The
numbers are real; their *comparability* is not. Four problems matter, and each has a correct
treatment.

### 1. Censored measurements

A paper reporting $\mathrm{IC_{50}} > 10{,}000$ nM means "we stopped looking". The true value is
unknown. Treating that as $= 10{,}000$ invents data: these are *left-censored observations*, the
same statistical object as a patient still alive at the end of a trial.

- **Correct:** drop them for regression, or use a censored likelihood (Tobit model).
- **Wrong:** substituting the threshold as if it were measured.

In the real dataset this is 18% of all rows.

### 2. Mixed units

Most values arrive in nM, a minority in µg/mL. Converting requires the molecular weight:

$$\mathrm{IC_{50}}[\mathrm{nM}] \;=\; \frac{\mathrm{IC_{50}}[\mu\mathrm{g/mL}]}{MW\,[\mathrm{g/mol}]}\times 10^{6}$$

Rows whose units cannot be converted must be dropped, not guessed.

### 3. Physically impossible values

Transcription errors produce $\mathrm{IC_{50}}$ values below 1 pM (tighter than biotin–streptavidin,
one of the strongest known non-covalent interactions) or above 1 M. A physical plausibility
window — we use $3 \le \mathrm{pIC_{50}} \le 12$ — removes them.

### 4. Replicate measurements

The same compound appears in many papers. Averaging *after* splitting the data leaks information
between train and test. Aggregate first, and keep the spread: it is the most valuable number in
the whole dataset, as Part 4 shows.

Use the **median** over replicates, not the mean — one transcription error shifts a mean of three
values by an order of magnitude, and leaves the median untouched.
''')

code(r'''
# The curation funnel actually applied to the real dataset.
steps = ["Raw records", "Exact values only\n(drop censored)", "Convertible units",
         "Physically plausible\n3 ≤ pIC50 ≤ 12", "Replicates aggregated\n(median)",
         "Consistent replicates\n(SD ≤ 1 log)"]
kept = [25197, 20694, 20688, 20643, 18380, 18145]

fig, ax = plt.subplots(figsize=(8.6, 4.2))
y = np.arange(len(kept))[::-1]
ax.barh(y, kept, color=[BLUE] * (len(kept) - 1) + [AQUA], height=.62, lw=0)
for yi, k, prev in zip(y, kept, [None] + kept[:-1]):
    lab = f"{k:,}" + (f"   −{prev-k:,}" if prev else "")
    ax.text(k + 260, yi, lab, va="center", fontsize=9.5, color=INK2)
ax.set(yticks=y, yticklabels=steps, xlim=(0, 29500), xlabel="measurements retained",
       title="Curation funnel — 28% of records removed before any modelling")
clean(ax, grid="x"); ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
plt.tight_layout(); plt.show()
''')

# ════════════════════════════════════════════════════════════════════ PART 4
md(r'''
---
## Part 4 — The ceiling: how good *could* any model be?

This is the most under-used idea in applied machine learning, and it costs teams months.

### The decomposition

Every observed label is a true value plus measurement error:

$$y_i \;=\; f(\mathbf{x}_i) \;+\; \varepsilon_i, \qquad
\varepsilon_i \sim \mathcal{N}(0,\sigma_{\text{exp}}^2)$$

Suppose you had a *perfect* model that returned $f(\mathbf{x}_i)$ exactly. Scored against the
noisy labels, it would still show

$$\mathrm{MSE}_{\min} = \sigma_{\text{exp}}^2
\qquad\Longrightarrow\qquad
\boxed{\;R^2_{\max} \;=\; 1 - \frac{\sigma_{\text{exp}}^{2}}{\sigma_{y}^{2}}\;}$$

No architecture, no amount of compute, and no quantity of extra features can cross that line.
Error below it is not skill — it is memorised noise.

### Estimating $\sigma_{\text{exp}}$ from your own data

You do not need a special experiment. Find pairs where the same compound was measured against
the same target in the same assay format, in different papers, and look at how much they disagree.
For $n_g$ replicates in group $g$ with sample SD $s_g$, the pooled estimate is

$$\sigma_{\text{exp}} \;=\;
\sqrt{\frac{\sum_g (n_g-1)\,s_g^{2}}{\sum_g (n_g-1)}}$$

**On the real dataset — 1,394 replicate groups, 3,597 measurements:**

| Comparison | $\sigma_{\text{exp}}$ |
|---|---|
| Within one publication | 0.72 log units |
| **Between different publications** | **0.92 log units** |
| Pooled | 0.83 log units |

34% of replicate groups disagree by more than 1 log unit — a **10-fold** discrepancy on the same
compound and target. A model achieving RMSE $\approx 0.8$ is therefore performing at the level of
the experiment itself.

### Read the caveat before you quote the number

The 0.92 figure is an **upper bound** on the noise in our curated labels, for three reasons:

1. We aggregated replicates by median, and the median of $n$ measurements is less noisy than one.
2. Compounds measured repeatedly are disproportionately the controversial ones.
3. Between-paper differences partly reflect genuine assay-condition differences (ATP concentration,
   incubation time) which the model partly sees through its `assay_category` feature.

So the honest statement is a *band*, not a point: the ceiling sits somewhere near
RMSE $\approx 0.7\text{–}0.9$. Quote it that way.
''')

code(r'''
# Prove the ceiling exists, using the surrogate — where we KNOW the truth.
sur, surFP = make_surrogate(n_scaffolds=260, noise_sd=0.75, seed=7)
sd_true = sur.pIC50_true.std()
sd_obs = sur.pIC50.std()
oracle_rmse = np.sqrt(np.mean((sur.pIC50_true - sur.pIC50) ** 2))   # perfect model vs noisy labels
r2_max = 1 - oracle_rmse ** 2 / sd_obs ** 2

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))

ax = axes[0]
ax.scatter(sur.pIC50_true, sur.pIC50, s=7, c=BLUE, alpha=.30, lw=0)
lim = (sur.pIC50.min() - .3, sur.pIC50.max() + .3)
ax.plot(lim, lim, color=INK2, lw=1.4)
ax.set(xlim=lim, ylim=lim, title="Even a perfect model cannot beat the noise",
       xlabel="true potency $f(x)$  (known only in simulation)", ylabel="measured potency $y$")
clean(ax, grid="both")
note(ax, f"An ORACLE predicting the true value\nstill scores RMSE = {oracle_rmse:.2f},\n$R^2$ = {r2_max:.2f}",
     xy=(0.03, 0.78), ha="left")

ax = axes[1]
parts = [("Irreducible\n(measurement noise)", oracle_rmse ** 2, GRAY),
         ("Reducible\n(what modelling can win)", sd_obs ** 2 - oracle_rmse ** 2, BLUE)]
left = 0
for lab, v, c in parts:
    ax.barh([0], [v], left=left, color=c, height=.5, lw=0)
    ax.text(left + v / 2, 0, f"{v / sd_obs**2:.0%}", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")
    left += v
ax.set(xlim=(0, sd_obs ** 2), ylim=(-.6, .9), yticks=[], xlabel="variance of the labels",
       title="Error budget — decide this before you train")
ax.legend(handles=[Rectangle((0, 0), 1, 1, color=c, lw=0) for _, _, c in parts],
          labels=[l for l, _, _ in parts], loc="upper center",
          bbox_to_anchor=(.5, 1.02), ncol=2)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.grid(False)
plt.tight_layout(); plt.show()

print(f"label SD                      : {sd_obs:.3f}")
print(f"injected noise SD             : 0.750")
print(f"oracle RMSE (perfect model)   : {oracle_rmse:.3f}")
print(f"=> maximum attainable R²      : {r2_max:.3f}   <-- the ceiling")
''')

md(r'''
> **Do this first, on every project.** If the ceiling is $R^2_{\max}=0.65$, a reviewer demanding
> 0.90 is demanding that you overfit, and a competitor reporting 0.95 on the same labels has a
> leakage bug. Knowing the ceiling turns an argument about taste into arithmetic.
''')

# ════════════════════════════════════════════════════════════════════ PART 5
md(r'''
---
## Part 5 — Turning molecules into numbers

A model consumes vectors. A molecule is a graph of atoms and bonds. The translation is called
*featurisation*, and the workhorse is the **extended-connectivity fingerprint** (ECFP, also
called a Morgan fingerprint).

### The algorithm

1. Give every atom an initial identifier from its element, charge, degree, and so on.
2. **Repeat $r$ times:** replace each atom's identifier with a hash of itself plus the
   identifiers of its immediate neighbours. After round $r$, an atom's identifier encodes
   everything within $r$ bonds of it.
3. Collect all identifiers generated at all rounds. Each is one *substructure*.
4. Hash each into one of $L$ buckets (we use $L = 2048$) and set that bit.

The result is a $\{0,1\}^{L}$ vector: "which substructures does this molecule contain?"
Radius $r=2$ (ECFP4) captures fragments up to about 5 atoms wide.

Each round is a message-passing step — the same operation a graph neural network performs, except
the aggregation function is a fixed hash instead of learned weights. ECFP is a GNN whose weights
were never trained, which is exactly why it is such a strong baseline on small datasets: it has no
parameters to overfit.

### Counted vs binary

A binary bit says a fragment is present. A **count** says how many times. Three carboxylic acids
behave differently from one, so counts carry real information — worth $\approx +0.01\ R^2$ in our
experiments.

### Measuring similarity

Molecular similarity is the **Tanimoto** (Jaccard) coefficient on the bit sets:

$$T(A,B) \;=\; \frac{|A \cap B|}{|A \cup B|} \;=\; \frac{c}{a + b - c}$$

where $a,b$ are the bit counts of each molecule and $c$ the shared bits. $T > 0.7$ is
conventionally "similar", $T < 0.3$ "unrelated". This single number drives the data-splitting
decision in Part 6.
''')

code(r'''
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdFingerprintGenerator, DataStructs
RDLogger.DisableLog("rdApp.*")

DRUGS = {
    "Gefitinib":    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Erlotinib":    "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "Osimertinib":  "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
    "Vemurafenib":  "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1F",
    "Crizotinib":   "C[C@H](Oc1cc(-c2cnn(C3CCNCC3)c2)cnc1N)c1c(Cl)ccc(F)c1Cl",
    "Imatinib":     "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
}
mols = {k: Chem.MolFromSmiles(v) for k, v in DRUGS.items()}
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = {k: gen.GetFingerprint(m) for k, m in mols.items()}

names = list(DRUGS)
T = np.array([[DataStructs.TanimotoSimilarity(fps[a], fps[b]) for b in names] for a in names])

fig, ax = plt.subplots(figsize=(6.4, 5.2))
im = ax.imshow(T, cmap=mpl.colors.LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1)
ax.set(xticks=range(len(names)), yticks=range(len(names)),
       title="Tanimoto similarity between approved kinase inhibitors")
ax.set_xticklabels(names, rotation=38, ha="right")
ax.set_yticklabels(names)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if T[i, j] > .55 else INK2)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=.8, label="Tanimoto $T$")
plt.tight_layout(); plt.show()

print("Gefitinib vs Erlotinib : %.2f  (same 4-anilinoquinazoline core — a chemical family)"
      % T[names.index("Gefitinib"), names.index("Erlotinib")])
print("Gefitinib vs Imatinib  : %.2f  (unrelated chemistry, both kinase drugs)"
      % T[names.index("Gefitinib"), names.index("Imatinib")])
Draw.MolsToGridImage(list(mols.values()), legends=names, molsPerRow=3, subImgSize=(260, 190))
''')

md(r'''
Note the structure in that matrix. Gefitinib and erlotinib score around 0.4 because they share the
**4-anilinoquinazoline** core; imatinib, from a different chemical family, scores below 0.2 against both.
That block structure is not a curiosity — it is the reason the next section exists.
''')

# ════════════════════════════════════════════════════════════════════ PART 6
md(r'''
---
## Part 6 — Splitting the data honestly

### The assumption that silently fails

Train/test splitting rests on the test set being an *independent* sample from the same
distribution. Molecular datasets violate this badly, because medicinal chemistry proceeds by
making dozens of close analogues of one promising compound.

Shuffle such a dataset at random and near-twins land on both sides of the split. The model then
gets credit for recognising a molecule it has effectively already seen. You have built an exam
where the students got the answer key, and the grade tells you nothing about what happens with a
new chemical series — which is the only question that matters.

### Scaffold splitting

Group molecules by their **Bemis–Murcko scaffold**: delete all side chains, keep the ring systems
and the linkers between them. Then assign whole scaffolds — never individual molecules — to train,
validation, or test.

```
          full molecule                     Murcko scaffold
   CC(=O)Nc1ccc(OCCN2CCCC2)cc1    ->    c1ccc(OCCN2CCCC2)cc1
   (side chains stripped, ring systems and linkers retained)
```

Now the test set contains chemical families the model has never encountered. The score drops —
and the lower number is the honest one.

### Variants that are stricter still

| Split | Groups by | Simulates |
|---|---|---|
| Random | nothing | *(not valid here)* |
| **Scaffold** | Murcko scaffold | a new chemical series |
| Time | publication year | prospective use |
| Leave-one-variant-out | protein variant | a new resistance mutation |

Scaffold splitting is the default; the others answer specific questions.
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.5))
rng = np.random.default_rng(1)
fam_colors = [BLUE, ORANGE, AQUA]

for ax, mode in zip(axes, ["Random split — leaks", "Scaffold split — honest"]):
    for f in range(3):
        for i in range(9):
            x, yv = 0.4 + i * 0.95, 2.1 - f * 0.95
            test = (rng.random() < .3) if "Random" in mode else (f == 2)
            ax.add_patch(Rectangle((x, yv), .72, .62, facecolor=fam_colors[f],
                                   alpha=1.0 if not test else .25, lw=0))
            if test:
                ax.text(x + .36, yv + .31, "T", ha="center", va="center",
                        fontsize=9, color=INK, fontweight="bold")
        ax.text(0.1, yv + .31, f"family {f+1}", ha="right", va="center", fontsize=9, color=INK2)
    ax.set(xlim=(-1.9, 9.3), ylim=(-1.25, 3.05), xticks=[], yticks=[], title=mode)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.grid(False)

note(axes[0], "Test molecules (T) sit beside\ntheir own close analogues.\nThe model recognises, not predicts.",
     xy=(0.56, 0.02), ha="left")
note(axes[1], "Family 3 is held out entirely.\nThe model must generalise\nto new chemistry.",
     xy=(0.56, 0.02), ha="left")
plt.tight_layout(); plt.show()
''')

code(r'''
# Measure the leakage. Same data, same model, same hyperparameters — only the split changes.
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error

X = np.hstack([FP, pd.get_dummies(df.variant_id, dtype=np.float32).to_numpy(),
               pd.get_dummies(df.assay_category, dtype=np.float32).to_numpy()]).astype(np.float32)
y = df.pIC50.to_numpy()
groups = df.scaffold.to_numpy()
MODEL = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
             min_samples_leaf=5, max_features=0.3, random_state=0)


def oof(splitter, **kw):
    """Out-of-fold predictions under a given splitting scheme."""
    p = np.zeros(len(y))
    for tr, te in splitter.split(X, y, **kw):
        p[te] = HistGradientBoostingRegressor(**MODEL).fit(X[tr], y[tr]).predict(X[te])
    return p


p_rand = oof(KFold(5, shuffle=True, random_state=0))
p_scaf = oof(GroupKFold(5), groups=groups)

res = pd.DataFrame({
    "split": ["Random (leaky)", "Scaffold (honest)"],
    "R2": [r2_score(y, p_rand), r2_score(y, p_scaf)],
    "RMSE": [mean_squared_error(y, p_rand) ** .5, mean_squared_error(y, p_scaf) ** .5],
})
print(res.round(3).to_string(index=False))
print(f"\nOptimism introduced by random splitting: ΔR² = {res.R2[0] - res.R2[1]:+.3f}")
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))

ax = axes[0]
ax.bar([0, 1], res.R2, color=[ORANGE, BLUE], width=.5, lw=0)
for i, v in enumerate(res.R2):
    ax.text(i, v + .012, f"{v:.3f}", ha="center", fontsize=11, color=INK, fontweight="bold")
ax.annotate("", xy=(1, res.R2[1]), xytext=(1, res.R2[0]),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.4))
ax.text(1.07, (res.R2[0] + res.R2[1]) / 2, f"illusion\nΔ = {res.R2[0]-res.R2[1]:.3f}",
        va="center", fontsize=9.5, color=INK2)
ax.set(xticks=[0, 1], xticklabels=["Random\n(leaky)", "Scaffold\n(honest)"],
       ylabel="$R^2$", ylim=(0, max(res.R2) * 1.25),
       title="The same model, scored two ways")
clean(ax)

ax = axes[1]
ax.scatter(y, p_scaf, s=6, c=BLUE, alpha=.22, lw=0)
lim = (y.min() - .3, y.max() + .3)
ax.plot(lim, lim, color=INK2, lw=1.4)
for d in (1, -1):
    ax.plot(lim, [lim[0] + d, lim[1] + d], color=GRAY, lw=1.0)
ax.set(xlim=lim, ylim=lim, xlabel="measured pIC$_{50}$", ylabel="predicted pIC$_{50}$",
       title="Honest predictions (scaffold split)")
clean(ax, grid="both")
within = (np.abs(p_scaf - y) < 1).mean() * 100
note(ax, f"Grey lines = ±1 log unit (10-fold).\n{within:.0f}% of predictions fall inside.",
     xy=(0.03, 0.80), ha="left")
plt.tight_layout(); plt.show()
''')

md(r'''
> **The number on the left is the one that gets published; the number on the right is the one
> that predicts your next experiment.** On the real dataset this gap is $\Delta R^2 = +0.12$.
> It is free to measure and it is the first thing to check in anyone else's model, including
> your own from six months ago.
''')

# ════════════════════════════════════════════════════════════════════ PART 7
md(r'''
---
## Part 7 — Baselines before models

A score has no meaning in isolation; it only has meaning relative to something simpler. Three
baselines take minutes and protect you from months of self-deception.

| Baseline | What it predicts | What beating it proves |
|---|---|---|
| **Global mean** | the training mean, always | your pipeline is wired correctly ($R^2 \approx 0$) |
| **Context only** | from target + variant + assay, no chemistry | the model uses the *molecule* |
| **Linear on fingerprints** | ridge regression | nonlinearity is worth its cost — *if* it wins |

The context-only baseline is the sharp one. If a complex model barely beats it, the model has
learned "EGFR assays average 6.9" and nothing about chemistry at all.
''')

code(r'''
from sklearn.linear_model import RidgeCV
from sklearn.dummy import DummyRegressor

ctx = np.hstack([pd.get_dummies(df.variant_id, dtype=np.float32).to_numpy(),
                 pd.get_dummies(df.assay_category, dtype=np.float32).to_numpy()]).astype(np.float32)
gkf = GroupKFold(5)


def eval_model(make, M):
    p = np.zeros(len(y))
    for tr, te in gkf.split(M, y, groups):
        p[te] = make().fit(M[tr], y[tr]).predict(M[te])
    return r2_score(y, p), mean_squared_error(y, p) ** .5


rows = [
    ("Global mean",              *eval_model(lambda: DummyRegressor(strategy="mean"), ctx)),
    ("Context only (no molecule)", *eval_model(lambda: HistGradientBoostingRegressor(**MODEL), ctx)),
    ("Ridge on fingerprints",    *eval_model(lambda: RidgeCV(alphas=np.logspace(-1, 3, 9)), X)),
    ("Gradient boosting (full)", r2_score(y, p_scaf), mean_squared_error(y, p_scaf) ** .5),
]
base = pd.DataFrame(rows, columns=["model", "R2", "RMSE"])
print(base.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(9.4, 3.6))
yy = np.arange(len(base))[::-1]
ax.barh(yy, base.R2, color=[GRAY, GRAY, BLUE, BLUE], height=.6, lw=0)
for yi, v in zip(yy, base.R2):
    ax.text(v + .008, yi, f"{v:.3f}", va="center", fontsize=10, color=INK2)
ax.set(yticks=yy, yticklabels=base.model, xlabel="$R^2$ (scaffold split)",
       xlim=(0, max(base.R2) * 1.22), title="Every model must earn its complexity")
clean(ax, grid="x"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
plt.tight_layout(); plt.show()

lin = base.loc[base.model == "Ridge on fingerprints", "R2"].iloc[0]
gbm = base.loc[base.model == "Gradient boosting (full)", "R2"].iloc[0]
print(f"\nRidge {lin:.3f}  vs  boosting {gbm:.3f}  ->  "
      + ("boosting earns its complexity." if gbm > lin + .01 else
         "the LINEAR model is as good or better. Boosting has not earned its place here."))
''')

md(r'''
### Read that last line carefully

Which model wins is a property of **the data**, not a fact about algorithms.

- On the **real** 18,145-measurement dataset, boosting wins clearly: $R^2$ 0.572 vs 0.452 for
  ridge. Real structure–activity relationships are full of context-dependence — a substituent
  that helps on one ring system does nothing on another — and capturing that needs interactions.
- On the **simulated** dataset, potency is built mostly as a *sum* of substructure
  contributions. For near-additive data, ridge regression is close to optimal and far more
  sample-efficient, so it can beat boosting.

Had we skipped the linear baseline and reported only the boosting result, we would never have
known which situation we were in. **The baseline is not a formality you clear on the way to the
real model; it is how you find out which model is the real one.**
''')

# ════════════════════════════════════════════════════════════════════ PART 8
md(r'''
---
## Part 8 — The model: gradient boosting, from the loss up

Gradient-boosted trees are the strongest *starting point* for fingerprint data of this size —
on datasets of $10^3$–$10^5$ molecules they are rarely beaten, though as Part 7 showed, you
verify that against a linear baseline rather than assuming it.

### The construction

Build an additive model one small correction at a time:

$$F_0(\mathbf{x}) = \bar{y}, \qquad
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu\, h_m(\mathbf{x})$$

where $h_m$ is a shallow regression tree and $\nu$ is the **learning rate**. Each tree is fitted
to the negative gradient of the loss at the current predictions. For squared loss
$L = \tfrac12\sum_i (y_i - F(\mathbf{x}_i))^2$:

$$-\frac{\partial L}{\partial F(\mathbf{x}_i)} \;=\; y_i - F_{m-1}(\mathbf{x}_i) \;=\; r_i$$

so each new tree simply fits the **residuals** — what the model has got wrong so far. The
procedure is relentlessly focused on current mistakes, which is both its power and its
overfitting risk; $\nu$ controls how much of each correction is accepted.

### The hyperparameters that matter

| Parameter | Effect | Our tuned value |
|---|---|---|
| `max_iter` | number of trees $M$ | 1500 |
| `learning_rate` $\nu$ | step size; small $\nu$ needs large $M$ | 0.06 |
| `max_leaf_nodes` | tree capacity, hence interaction order | 63 |
| `min_samples_leaf` | smoothing | 5 |
| `max_features` | fraction of columns offered per split | **0.3** |

`max_features = 0.3` was the single most important setting. With 2,751 highly correlated
fingerprint columns, offering every column at every split makes all trees pick the same few
features and agree with each other; sampling a third decorrelates them. An ensemble only helps
when its members make *different* mistakes.
''')

code(r'''
# Boosting in one dimension: watch the residuals get consumed.
rng = np.random.default_rng(0)
xs = np.linspace(0, 1, 160)
truth = np.sin(7 * xs) * .8 + xs * 1.6
obs = truth + rng.normal(0, .22, xs.size)

from sklearn.tree import DecisionTreeRegressor
Xd = xs.reshape(-1, 1)
F = np.full(xs.size, obs.mean())
snapshots, nu = {}, 0.35
for m in range(1, 61):
    r = obs - F
    F = F + nu * DecisionTreeRegressor(max_depth=3, random_state=0).fit(Xd, r).predict(Xd)
    if m in (1, 5, 20, 60):
        snapshots[m] = F.copy()

fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.3), sharey=True)
for ax, (m, Fm) in zip(axes, snapshots.items()):
    ax.scatter(xs, obs, s=9, c=GRAY, alpha=.55, lw=0)
    ax.plot(xs, truth, color=INK2, lw=1.3)
    ax.plot(xs, Fm, color=BLUE, lw=2.2)
    ax.set(title=f"M = {m} trees", xticks=[], xlabel="feature")
    clean(ax, grid="y")
    ax.text(.04, .93, f"RMSE {np.sqrt(np.mean((Fm-truth)**2)):.3f}", transform=ax.transAxes,
            fontsize=9, color=INK2, va="top")
axes[0].set_ylabel("target")
axes[0].legend(handles=[plt.Line2D([], [], color=INK2, lw=1.3),
                        plt.Line2D([], [], color=BLUE, lw=2.2)],
               labels=["truth", "model $F_m$"], loc="lower right")
fig.suptitle("Each tree fits what the previous ones got wrong", x=.007, ha="left",
             fontsize=12, fontweight="bold", color=INK)
plt.tight_layout(); plt.show()
''')

code(r'''
# Where does extra capacity stop helping? Validation curve under the honest split.
tr_idx, te_idx = next(GroupKFold(5).split(X, y, groups))
iters = [10, 25, 50, 100, 200, 400, 800]
tr_s, te_s = [], []
for it in iters:
    m = HistGradientBoostingRegressor(**{**MODEL, "max_iter": it}).fit(X[tr_idx], y[tr_idx])
    tr_s.append(r2_score(y[tr_idx], m.predict(X[tr_idx])))
    te_s.append(r2_score(y[te_idx], m.predict(X[te_idx])))

fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.plot(iters, tr_s, color=ORANGE, marker="o", label="training fold")
ax.plot(iters, te_s, color=BLUE, marker="o", label="held-out fold")
ax.set(xscale="log", xlabel="number of trees $M$  (log scale)", ylabel="$R^2$",
       title="The gap between the curves is overfitting")
clean(ax); ax.legend(loc="center right")
note(ax, "Training score keeps climbing.\nHeld-out score plateaus.\nOnly the lower curve is informative.",
     xy=(0.03, 0.70), ha="left")
plt.tight_layout(); plt.show()
''')

# ════════════════════════════════════════════════════════════════════ PART 9
md(r'''
---
## Part 9 — Metrics: what each one tells you, and what it hides

$$\mathrm{RMSE} = \sqrt{\tfrac1n\textstyle\sum_i (y_i-\hat y_i)^2}
\qquad
\mathrm{MAE} = \tfrac1n\textstyle\sum_i |y_i-\hat y_i|
\qquad
R^2 = 1-\frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2}$$

| Metric | Reads as | Blind spot |
|---|---|---|
| **RMSE** | typical error, in log units | one catastrophic miss dominates it |
| **MAE** | typical error, outlier-tolerant | treats a 100-fold miss as merely bad |
| $R^2$ | fraction of variance explained | **depends on the test set's spread** |
| **Spearman $\rho$** | is the ranking right? | ignores systematic bias entirely |
| **% within 1 log** | fraction predicted within 10-fold | hides how wrong the rest are |

### The $R^2$ trap

$R^2$ compares your model against predicting the mean. Evaluate on a narrow, homogeneous test
set and $\sigma_y^2$ shrinks, so $R^2$ collapses *even if RMSE is unchanged*. Two papers reporting
$R^2 = 0.6$ and $R^2 = 0.8$ may describe identically accurate models on different test sets.

**Always report RMSE alongside $R^2$.** RMSE is in physical units — log units of potency — and
is comparable across datasets.

### Interpreting our numbers

RMSE $= 0.80$ log units means the predicted $\mathrm{IC_{50}}$ is typically off by
$10^{0.80} \approx 6.3$-fold. For deciding which of twenty compounds to synthesise, that is
useful. For reporting a potency value in a paper, it is not.
''')

code(r'''
from scipy.stats import spearmanr

resid = p_scaf - y
m = dict(RMSE=mean_squared_error(y, p_scaf) ** .5, MAE=np.abs(resid).mean(),
         R2=r2_score(y, p_scaf), rho=spearmanr(y, p_scaf).statistic,
         within1=(np.abs(resid) < 1).mean() * 100)
print(f"RMSE {m['RMSE']:.3f} log  ->  typical IC50 error {10**m['RMSE']:.1f}-fold")
print(f"MAE  {m['MAE']:.3f} log      R² {m['R2']:.3f}      Spearman ρ {m['rho']:.3f}"
      f"      within 10-fold {m['within1']:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))

ax = axes[0]
ax.scatter(p_scaf, resid, s=6, c=BLUE, alpha=.22, lw=0)
ax.axhline(0, color=INK2, lw=1.4)
for d in (1, -1):
    ax.axhline(d, color=GRAY, lw=1.0)
ax.set(xlabel="predicted pIC$_{50}$", ylabel="residual (predicted − measured)",
       title="Residuals: look for structure, not just size")
clean(ax, grid="y")
note(ax, "Tilt = regression to the mean:\npotent compounds under-predicted,\nweak ones over-predicted.",
     xy=(0.03, 0.04), ha="left")

ax = axes[1]
err = np.sort(np.abs(resid))[::-1]
share = np.cumsum(err ** 2) / np.sum(err ** 2) * 100
frac = np.arange(1, err.size + 1) / err.size * 100
ax.plot(frac, share, color=BLUE)
k = np.searchsorted(frac, 10)
ax.plot([10, 10], [0, share[k]], color=GRAY, lw=1.0)
ax.plot([0, 10], [share[k], share[k]], color=GRAY, lw=1.0)
ax.scatter([10], [share[k]], s=45, c=BLUE, zorder=5)
ax.set(xlabel="worst predictions, cumulative % of dataset", ylabel="% of total squared error",
       title="A small tail owns most of the error", xlim=(0, 100), ylim=(0, 101))
clean(ax)
note(ax, f"The worst 10% of predictions\ncarry {share[k]:.0f}% of all squared error.",
     xy=(0.35, 0.18), ha="left")
plt.tight_layout(); plt.show()
''')

md(r'''
That concentration matters for diagnosis. On the real dataset the worst 6.5% of predictions carry
**40%** of the squared error. Two things produce such a tail, and they demand opposite responses:

- **Activity cliffs** — genuinely similar molecules with very different potency. A real
  limitation of 2-D structural features.
- **Bad measurements** — the 34% of replicate groups that disagree by >10-fold.

Separating them requires re-measurement, not more modelling. Reporting the tail honestly is part
of the job.
''')

# ════════════════════════════════════════════════════════════════════ PART 10
md(r'''
---
## Part 10 — Negative controls: making a feature earn its place

Adding features is cheap, and almost any feature *appears* to help if you look at it loosely.
Discipline requires the opposite stance: **a feature is presumed useless until it survives a test
designed to expose it.**

Two case studies from this project, both of which overturned the intuitive expectation.

### Case 1 — Molecular docking

Docking places a molecule into the protein's binding site and scores the fit, in kcal/mol. It is
physically motivated, universally used, and here it carries **no usable signal about potency**.
Three independent tests agreed:

| Test | Result |
|---|---|
| Rank correlation with potency | $\rho \le 0.19$ |
| Added to the model | $R^2$: 0.5719 → 0.5708 (no change) |
| Docking alone vs descriptors alone | 0.177 vs 0.340 — the credit belonged to size and lipophilicity |

The failure mode is instructive. Docking scores sum per-atom contact terms, so bigger molecules
accumulate better scores almost automatically. The score correlates with *size*
($\rho = 0.44$) far more than with potency. Normalising by atom count — ligand efficiency —

$$\mathrm{LE} \;=\; \frac{-\Delta G_{\text{dock}}}{N_{\text{heavy atoms}}}$$

does not rescue it either.

### Case 2 — Protein sequence embeddings (ESM2)

The intuition is strong: mutations change the protein, so the model should read the sequence. A
protein language model such as ESM2 provides a rich learned representation.

It did not help — and the reason generalises.

With only 30 protein variants, a one-hot encoding is **saturated**: it already distinguishes every
variant, and a tree model can assign each one any behaviour it likes. No alternative encoding of
those same 30 objects can add information *in-sample*. Sequence can only help where one-hot is
blind: a variant **absent from training**. So that is the test — hold out an entire variant,
train on the rest, predict.

Across 112 models (hand-built physicochemical features and ESM2 embeddings in four
configurations) nothing closed the gap. The diagnostic detail: giving the model a *stronger*
mutation signal made it **worse**. It was using the embedding to identify the variant, not to
understand it.

The deeper constraint is sample size in the right dimension. Learning "a bulky hydrophobic
substitution at this pocket position reduces potency" needs many *different mutations* to learn
from. With 27, no representation suffices. **The bottleneck was the number of variants, not how
they were encoded.**

### The habit to copy

**Write down your prediction before running the experiment.** We recorded "no ESM2 variant
improves weighted RMSE by more than 0.02" in the script before it ran. The best result was
+0.0136. A prediction made in advance cannot be rationalised afterwards.
''')

code(r'''
# Reproduce the docking diagnosis: potency vs size.
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))

ax = axes[0]
ax.scatter(-df.dock_best, df.pIC50, s=6, c=BLUE, alpha=.20, lw=0)
r_pot = spearmanr(-df.dock_best, df.pIC50).statistic
ax.set(xlabel="docking score (−kcal/mol, higher = better fit)", ylabel="measured pIC$_{50}$",
       title="Docking vs potency — the shot that misses")
clean(ax, grid="both")
note(ax, f"Spearman ρ = {r_pot:.3f}\nA featureless cloud.", xy=(0.03, 0.84), ha="left")

ax = axes[1]
ax.scatter(df.heavy_atoms, -df.dock_best, s=6, c=ORANGE, alpha=.20, lw=0)
r_size = spearmanr(df.heavy_atoms, -df.dock_best).statistic
b, a = np.polyfit(df.heavy_atoms, -df.dock_best, 1)
xr = np.array([df.heavy_atoms.min(), df.heavy_atoms.max()])
ax.plot(xr, b * xr + a, color=INK2, lw=1.6)
ax.set(xlabel="molecular size (heavy atoms)", ylabel="docking score (−kcal/mol)",
       title="Docking vs size — what it actually measures")
clean(ax, grid="both")
note(ax, f"Spearman ρ = {r_size:.3f}\nThe score mostly counts atoms.", xy=(0.03, 0.84), ha="left")
plt.tight_layout(); plt.show()

print(f"docking ~ potency : ρ = {r_pot:+.3f}")
print(f"docking ~ size    : ρ = {r_size:+.3f}   <- the feature is a size proxy")
''')

code(r'''
# Does docking add anything on top of the fingerprint? Same split, same model.
extra = df[["dock_best", "heavy_atoms"]].to_numpy(np.float32)
r2_no, rm_no = r2_score(y, p_scaf), mean_squared_error(y, p_scaf) ** .5
r2_yes, rm_yes = eval_model(lambda: HistGradientBoostingRegressor(**MODEL),
                            np.hstack([X, extra]).astype(np.float32))
print(f"fingerprint only          R² = {r2_no:.4f}   RMSE = {rm_no:.4f}")
print(f"fingerprint + docking     R² = {r2_yes:.4f}   RMSE = {rm_yes:.4f}")
print(f"\ngain from docking: ΔR² = {r2_yes - r2_no:+.4f}  "
      f"-> {'keep it' if r2_yes - r2_no > 0.01 else 'drop it: not worth 2 extra columns'}")
''')

# ════════════════════════════════════════════════════════════════════ PART 11
md(r'''
---
## Part 11 — Ask the question you actually need answered

"What is this compound's $\mathrm{IC_{50}}$?" is a harder question than "which of these compounds
should I make first?" — and usually it is not the question you have.

Reframing as **ranking** or **classification** has two advantages: activity cliffs hurt far less
(you only need the order roughly right), and the metrics map onto the real decision.

### Classification metrics

With a potency threshold (say $\mathrm{pIC_{50}} \ge 7$, i.e. $\mathrm{IC_{50}} \le 100$ nM):

$$\mathrm{ROC\text{-}AUC} = \Pr\big(\hat{s}(\text{active}) > \hat{s}(\text{inactive})\big)$$

the probability that a randomly chosen active outranks a randomly chosen inactive. 0.5 is
coin-flipping, 1.0 is perfect.

For screening, what matters is the top of the list. The **enrichment factor** at the top
fraction $f$:

$$\mathrm{EF}_{f} \;=\; \frac{\text{hit rate in the top } f}{\text{hit rate overall}}$$

### Read the enrichment factor with care

$\mathrm{EF}_f$ is bounded by $1/\text{(base rate)}$. Our dataset is 54% actives — these are
compounds from medicinal-chemistry papers, designed to be potent — so the ceiling is
$1/0.54 = 1.85$ and the metric is nearly uninformative. In a real screening library at a 0.1%
hit rate, the same model would show enrichment of 20–50×.

**A metric that looks poor may just be inapplicable to your dataset.** Here ROC-AUC and
precision in the top 5% are the meaningful ones.

### One model, not two

A dedicated classifier did **not** beat simply ranking by the regression model's output
(ROC-AUC 0.883 vs 0.894). Train the regressor, rank with it, and threshold wherever the project
needs — no second model to maintain.
''')

code(r'''
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

THR = 7.0
is_active = (y >= THR).astype(int)
fpr, tpr, _ = roc_curve(is_active, p_scaf)
auc = roc_auc_score(is_active, p_scaf)
ap = average_precision_score(is_active, p_scaf)
base_rate = is_active.mean()

order = np.argsort(-p_scaf)
fracs = np.linspace(.01, 1, 120)
ef = [is_active[order[:max(1, int(f * len(y)))]].mean() / base_rate for f in fracs]

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))

ax = axes[0]
ax.plot(fpr, tpr, color=BLUE)
ax.plot([0, 1], [0, 1], color=GRAY, lw=1.2)
ax.set(xlabel="false positive rate", ylabel="true positive rate", xlim=(0, 1), ylim=(0, 1),
       title=f"ROC curve — ranking actives (pIC$_{{50}}$ ≥ {THR:.0f})")
clean(ax, grid="both")
ax.legend(handles=[plt.Line2D([], [], color=BLUE, lw=2),
                   plt.Line2D([], [], color=GRAY, lw=1.2)],
          labels=[f"model (AUC = {auc:.3f})", "random (AUC = 0.5)"], loc="lower right")

ax = axes[1]
ax.plot(fracs * 100, ef, color=BLUE)
ax.axhline(1, color=GRAY, lw=1.2)
ax.axhline(1 / base_rate, color=ORANGE, lw=1.4)
ax.set(xlabel="top % of the ranked list", ylabel="enrichment factor",
       title="Enrichment is capped by the base rate")
clean(ax)
ax.legend(handles=[plt.Line2D([], [], color=BLUE, lw=2),
                   plt.Line2D([], [], color=ORANGE, lw=1.4),
                   plt.Line2D([], [], color=GRAY, lw=1.2)],
          labels=["model", f"theoretical max ({1/base_rate:.2f}×)", "random (1×)"],
          loc="upper right")
plt.tight_layout(); plt.show()

k = int(.05 * len(y))
print(f"actives in dataset        : {base_rate:.1%}   -> max possible enrichment {1/base_rate:.2f}×")
print(f"ROC-AUC                   : {auc:.3f}")
print(f"average precision         : {ap:.3f}")
print(f"precision in top 5%       : {is_active[order[:k]].mean():.1%}")
''')

# ════════════════════════════════════════════════════════════════════ PART 12
md(r'''
---
## Part 12 — The protocol

Everything above, as a sequence. The order is not arbitrary: each step can invalidate the ones
after it, so running them out of order wastes the work that follows.
''')

code(r'''
fig, ax = plt.subplots(figsize=(11.6, 5.0))
STEPS = [
    ("1. Frame the question", "regression? ranking?\nwhat decision does it serve?", BLUE),
    ("2. Curate", "censored values, units,\nimpossible values, replicates", BLUE),
    ("3. Transform the target", "pIC50 — linear in ΔG,\nadditive errors", BLUE),
    ("4. Estimate the ceiling", "σ_exp from replicates\nR²_max = 1 − σ²_exp/σ²_y", ORANGE),
    ("5. Split by scaffold", "group whole chemical families\nnever individual molecules", ORANGE),
    ("6. Baselines", "mean · context-only · linear", BLUE),
    ("7. Model + tune", "tune on a validation fold,\nkeep one fold untouched", BLUE),
    ("8. Negative controls", "every feature earns its place;\nregister predictions first", ORANGE),
    ("9. Report honestly", "RMSE with R², the error tail,\nand the known limits", AQUA),
]
ROW_H, ROW_GAP, BOX_W = 1.42, 0.70, 3.50
for i, (title, body, c) in enumerate(STEPS):
    r, cc = divmod(i, 3)
    x, yv = cc * 3.85, -r * (ROW_H + ROW_GAP)
    ax.add_patch(Rectangle((x, yv), BOX_W, ROW_H, facecolor=c, alpha=.11,
                           edgecolor=c, lw=1.5, zorder=1))
    ax.text(x + .17, yv + 1.08, title, fontsize=10.5, color=INK, fontweight="bold")
    ax.text(x + .17, yv + .28, body, fontsize=9.2, color=INK2, linespacing=1.45)
    if cc < 2:                       # arrow to the next box in the same row
        ax.add_patch(FancyArrowPatch((x + BOX_W + .06, yv + ROW_H / 2),
                                     (x + BOX_W + .30, yv + ROW_H / 2),
                                     arrowstyle="-|>", mutation_scale=13, color=INK3, lw=1.3))
    elif i < len(STEPS) - 1:          # row break: a short drop at the left of the next row
        ax.add_patch(FancyArrowPatch((1.75, yv - .10), (1.75, yv - ROW_GAP + .10),
                                     arrowstyle="-|>", mutation_scale=13, color=INK3, lw=1.3))
ax.set(xlim=(-.35, 11.6), ylim=(-2 * (ROW_H + ROW_GAP) - .45, ROW_H + 1.35),
       xticks=[], yticks=[])
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)
ax.grid(False)
ax.text(0, ROW_H + .92, "A protocol for trustworthy modelling", fontsize=13.5,
        fontweight="bold", color=INK)
ax.text(0, ROW_H + .42, "Orange steps are the ones most often skipped — and the ones that decide "
                        "whether the final number means anything.", fontsize=9.6, color=INK2)
plt.tight_layout(); plt.show()
''')

md(r'''
### The checklist

Before you believe a model — yours or anyone else's:

- [ ] **Censored values** excluded from regression, not silently converted
- [ ] **Units** harmonised; unconvertible rows dropped
- [ ] **Replicates** aggregated by median *before* splitting, and their spread reported
- [ ] **Noise ceiling** estimated, and the reported score compared against it
- [ ] **Split grouped by scaffold**, with zero scaffolds shared between folds — *verified in code*
- [ ] **Leakage quantified** by also running the naive split, so the gap is known
- [ ] **Baselines** reported: mean, context-only, linear
- [ ] **Hyperparameters tuned** on a validation fold, with one fold never touched until the end
- [ ] **Every feature block** justified by an ablation, not by plausibility
- [ ] **RMSE reported with $R^2$**, in physical units
- [ ] **Error tail** characterised, not averaged away
- [ ] **Limits stated**: where the model fails, and what it must not be used for

### The two sentences to remember

A model's value is not its score; it is the trustworthiness of its score. An honest $R^2 = 0.6$
will guide a real project, and an inflated $R^2 = 0.9$ will waste a year of synthesis.
''')

# ════════════════════════════════════════════════════════════════════ PART 13
md(r'''
---
# Part 13 — Workshop

Six exercises, roughly in order of difficulty. Each has a starter cell. Work on the dataset
loaded at the top (`df`, `FP`, `X`, `y`, `groups`).

> Compare your numbers with a partner. Where you disagree, one of you has a bug — finding it is
> the real exercise.
''')

md(r'''
### Exercise 1 — Measure the leakage yourself (15 min)

Run 5-fold CV three ways: random, scaffold-grouped, and grouped by `variant_id`.

1. Report $R^2$ and RMSE for each.
2. Rank them from most to least optimistic and explain the ordering.
3. Grouping by `variant_id` gives a *lower* score than scaffold grouping. What question does
   that number answer, and why is it the hardest one?
''')

code(r'''
# Exercise 1 — starter
schemes = {
    "random":   (KFold(5, shuffle=True, random_state=0), None),
    "scaffold": (GroupKFold(5), groups),
    "variant":  (GroupKFold(5), df.variant_id.to_numpy()),
}
for name, (splitter, g) in schemes.items():
    p = np.zeros(len(y))
    for tr, te in (splitter.split(X, y, g) if g is not None else splitter.split(X, y)):
        p[te] = HistGradientBoostingRegressor(**MODEL).fit(X[tr], y[tr]).predict(X[te])
    print(f"{name:9s} R² = {r2_score(y, p):.3f}   RMSE = {mean_squared_error(y, p)**.5:.3f}")

# YOUR ANSWER (2 and 3) in a comment here:
''')

md(r'''
### Exercise 2 — Find the noise ceiling in the surrogate (20 min)

Using `make_surrogate`, which exposes the hidden `pIC50_true`:

1. Generate datasets with `noise_sd` in $\{0.2, 0.5, 0.75, 1.0, 1.5\}$.
2. For each, compute the theoretical $R^2_{\max}$ and the $R^2$ your model actually achieves
   under a scaffold split.
3. Plot both against `noise_sd` on one axis.
4. The achieved score stays *below* the ceiling at every noise level. Name the two distinct
   reasons for that gap, and say which one more training data would fix.
5. What would you conclude about a model that *exceeds* its estimated ceiling on real data?
''')

code(r'''
# Exercise 2 — starter
for nsd in [0.2, 0.5, 0.75, 1.0, 1.5]:
    s, sfp = make_surrogate(n_scaffolds=260, noise_sd=nsd, seed=1)
    ys, gs = s.pIC50.to_numpy(), s.scaffold.to_numpy()
    Xs = np.hstack([sfp, pd.get_dummies(s.variant_id, dtype=np.float32).to_numpy()]).astype(np.float32)
    r2_ceiling = 1 - nsd ** 2 / ys.var()
    p = np.zeros(len(ys))
    for tr, te in GroupKFold(5).split(Xs, ys, gs):
        p[te] = HistGradientBoostingRegressor(**MODEL).fit(Xs[tr], ys[tr]).predict(Xs[te])
    print(f"noise {nsd:.2f}:  ceiling R² = {r2_ceiling:.3f}   achieved R² = {r2_score(ys, p):.3f}")

# TODO: make the plot, then answer question 4
''')

md(r'''
### Exercise 3 — Does molecular size predict potency? (20 min)

Larger molecules can make more contacts with the protein, but are also harder to turn into
oral drugs. Measure the relationship before assuming it.

1. Compute Spearman $\rho$ between `heavy_atoms` and `pIC50`.
2. Train a model on `heavy_atoms` *alone* under a scaffold split. What $R^2$?
3. Compute ligand efficiency $\mathrm{LE} = \mathrm{pIC_{50}} / N_{\text{heavy}}$ and
   correlate *that* with potency. Explain the sign.
4. Why would a model that ranks compounds purely by size be dangerous in a real project?
''')

code(r'''
# Exercise 3 — starter
print("ρ(size, potency) =", round(spearmanr(df.heavy_atoms, df.pIC50).statistic, 3))
# TODO: steps 2–4
''')

md(r'''
### Exercise 4 — Pick the threshold for a real decision (25 min)

Your team can synthesise **20 compounds**. You have predictions for all of them.

1. Rank by predicted potency and take the top 20. What fraction are truly active at
   $\mathrm{pIC_{50}} \ge 7$? At $\ge 8$?
2. Repeat for the top 100 and top 500. Plot precision against list length.
3. Precision falls as the list grows. Why is that *expected* rather than a defect?
4. Your project needs $\mathrm{IC_{50}} \le 10$ nM. Recompute at that threshold and state
   whether the model is good enough to decide for you.
''')

code(r'''
# Exercise 4 — starter
order = np.argsort(-p_scaf)
for k in (20, 100, 500):
    top = order[:k]
    print(f"top {k:4d}:  active@100nM {np.mean(y[top] >= 7):.0%}   active@10nM {np.mean(y[top] >= 8):.0%}")
# TODO: the plot and the written answers
''')

md(r'''
### Exercise 5 — Design a negative control (30 min)

Invent a feature you are confident is **useless** — a random number per molecule, the length of
its `mol_id`, anything. Then:

1. Add it to `X` and evaluate under the scaffold split. Does $R^2$ move?
2. Add **500** such random columns. Now what happens, and why?
3. A colleague adds a feature, sees $R^2$ rise from 0.571 to 0.578, and declares success.
   Design the experiment that decides whether that gain is real. State the number of repeats and
   the decision rule *before* running it.
4. Run your experiment. Was the colleague right?
''')

code(r'''
# Exercise 5 — starter
rng = np.random.default_rng(0)
junk = rng.normal(size=(len(y), 1)).astype(np.float32)
r2_junk, _ = eval_model(lambda: HistGradientBoostingRegressor(**MODEL),
                        np.hstack([X, junk]).astype(np.float32))
print(f"baseline R² = {r2_score(y, p_scaf):.4f}    with 1 junk column = {r2_junk:.4f}")
# TODO: steps 2–4
''')

md(r'''
### Exercise 6 — Write the model card (30 min)

Produce a one-page summary of your final model, structured as:

1. **Intended use** — the decisions it may inform
2. **Out-of-scope use** — three specific things it must not be used for, with the evidence
3. **Data** — size, sources, curation applied
4. **Evaluation** — split scheme, metrics with uncertainty, baselines
5. **Performance ceiling** — your $\sigma_{\text{exp}}$ estimate and the resulting band
6. **Known failure modes** — where the error concentrates, quantified
7. **Negative results** — features tested and rejected, with the numbers

A model card that omits section 7 is marketing.
''')

# ════════════════════════════════════════════════════════════════════ PART 14
md(r'''
---
# Part 14 — Exam

**40 points. Show your reasoning: a correct number with wrong reasoning earns half credit.**

Run the cell at the very end to check your answers — but write them all down first.
''')

md(r'''
## Section A — Concepts (12 points, 2 each)

**A1.** A colleague reports $R^2 = 0.94$ on a kinase potency dataset using a random 80/20 split.
Name the most likely problem and the single experiment that would confirm it.

**A2.** Why is $\mathrm{pIC_{50}}$ preferred over $\mathrm{IC_{50}}$? Give one statistical reason
and one physical reason.

**A3.** A dataset's labels have $\sigma_{\text{exp}} = 0.7$ log units and $\sigma_y = 1.4$.
Compute $R^2_{\max}$. A model reports $R^2 = 0.85$. What do you conclude?

**A4.** Explain why ECFP fingerprints remain competitive with graph neural networks on datasets
of a few thousand molecules.

**A5.** Your model's enrichment factor at the top 5% is 1.8, which sounds poor. Before concluding
the model is weak, what must you check?

**A6.** Define an activity cliff and explain why it cannot be fixed by a better featurisation of
2-D structure.

## Section B — Calculations (12 points, 3 each)

**B1.** A compound has $\mathrm{IC_{50}} = 45$ nM. Compute $\mathrm{pIC_{50}}$. A model predicts
7.9. By what fold factor is the prediction wrong?

**B2.** Two fingerprints have 80 and 95 set bits, sharing 50. Compute the Tanimoto similarity.
Would convention call these molecules similar?

**B3.** Replicate groups give $(n_g, s_g)$ of $(2, 0.5)$, $(3, 0.9)$, $(2, 1.2)$. Compute the
pooled $\sigma_{\text{exp}}$.

**B4.** A model has RMSE $= 0.65$ log units. Express the typical error as a fold-change in
$\mathrm{IC_{50}}$, and as kJ/mol of binding free energy at 300 K.

## Section C — Experimental design (10 points)

**C1.** (5 pts) You have 50,000 measurements across 80 kinases and want to predict potency for a
kinase with only 12 measurements. Design the evaluation. Which split, which baselines, and what
result would convince you the model transfers?

**C2.** (5 pts) Your team proposes adding 3-D docking features at a cost of 2 weeks of compute.
Design the smallest experiment that decides whether to proceed. State your decision rule and
required effect size *before* running it.

## Section D — Critique (6 points)

**D1.** (6 pts) A paper reports: *"Our deep learning model achieves $R^2 = 0.91$ and RMSE $= 0.31$
on held-out data, outperforming random forest ($R^2 = 0.78$). We used an 80/10/10 random split of
45,000 ChEMBL $\mathrm{IC_{50}}$ values across 200 targets, including all reported values."*

List every methodological problem and state, for each, which direction it biases the result.
''')

code(r'''
# ── Answer key ────────────────────────────────────────────────────────────────
# Write your answers first. Then run reveal("A1"), reveal("B3"), ... or reveal("all").
import base64, textwrap

_K = {
 "A1": "RGF0YSBsZWFrYWdlIHRocm91Z2ggYSByYW5kb20gc3BsaXQ6IGNsb3NlIGFuYWxvZ3VlcyBhcHBlYXIgaW4gYm90aCB0cmFpbiBhbmQgdGVzdCwgc28gdGhlIG1vZGVsIHJlY29nbmlzZXMgcmF0aGVyIHRoYW4gcHJlZGljdHMuIENvbmZpcm0gaXQgYnkgcmUtcnVubmluZyB0aGUgaWRlbnRpY2FsIG1vZGVsIHVuZGVyIGEgc2NhZmZvbGQtZ3JvdXBlZCBzcGxpdDsgYSBkcm9wIG9mIDAuMSsgaW4gUjIgaXMgdGhlIHNpZ25hdHVyZS4gKE91cnMgZHJvcHBlZCAwLjEyLik=",
 "A2": "U3RhdGlzdGljYWw6IGFzc2F5IGVycm9yIGlzIG11bHRpcGxpY2F0aXZlLCBzbyB0aGUgbG9nIG1ha2VzIGl0IGFkZGl0aXZlIGFuZCByb3VnaGx5IG5vcm1hbCwgd2hpY2ggbGVhc3Qtc3F1YXJlcyBhc3N1bWVzOyBpdCBhbHNvIHN0b3BzIHdlYWsgY29tcG91bmRzIGRvbWluYXRpbmcgdGhlIGxvc3MuIFBoeXNpY2FsOiBkRyA9IC1SVCBsbiBLLCBzbyBwSUM1MCBpcyBsaW5lYXIgaW4gYmluZGluZyBmcmVlIGVuZXJneSAtIG9uZSB1bml0IH4gNS43IGtKL21vbCBhdCAzMDAgSy4=",
 "A3": "UjJfbWF4ID0gMSAtIDAuNDkvMS45NiA9IDAuNzUuIEEgcmVwb3J0ZWQgMC44NSBleGNlZWRzIHRoZSBjZWlsaW5nLCBzbyBpdCBjYW5ub3QgYmUgZ2VudWluZSBza2lsbDogc3VzcGVjdCBsZWFrYWdlLCBkdXBsaWNhdGVkIHJlY29yZHMgYWNyb3NzIGZvbGRzLCBvciBhIHRlc3Qgc2V0IHdpdGggYXJ0aWZpY2lhbGx5IHdpZGUgc3ByZWFkLiAoQ2F2ZWF0OiBjaGVjayB0aGUgc2lnbWEgZXN0aW1hdGUgdG9vIC0gaXQgaXMgYW4gdXBwZXIgYm91bmQgaWYgbGFiZWxzIHdlcmUgbWVkaWFuLWFnZ3JlZ2F0ZWQuKQ==",
 "A4": "RUNGUCBpcyBtZXNzYWdlIHBhc3Npbmcgd2l0aCBhIGZpeGVkIGhhc2ggaW5zdGVhZCBvZiBsZWFybmVkIHdlaWdodHM6IHplcm8gcGFyYW1ldGVycyB0byBmaXQsIGhlbmNlIG5vdGhpbmcgdG8gb3ZlcmZpdC4gQSBHTk4gbXVzdCBsZWFybiBpdHMgcmVwcmVzZW50YXRpb24gZnJvbSB0aGUgc2FtZSBmZXcgdGhvdXNhbmQgbm9pc3kgbGFiZWxzLCB3aGljaCByYXJlbHkgcGF5cyBvZmYgYmVsb3cgfjEwNCBleGFtcGxlcy4=",
 "A5": "VGhlIGJhc2UgcmF0ZS4gRUYgaXMgY2FwcGVkIGF0IDEvKGZyYWN0aW9uIGFjdGl2ZSk7IGF0IDU0JSBhY3RpdmVzIHRoZSBtYXhpbXVtIGlzIDEuODUsIHNvIDEuOCBpcyBuZWFyLXBlcmZlY3QsIG5vdCBwb29yLiBVc2UgUk9DLUFVQyBhbmQgdG9wLWsgcHJlY2lzaW9uIHdoZW4gdGhlIGRhdGFzZXQgaXMgYWN0aXZlLWVuclc=",
 "A6": "VHdvIG1vbGVjdWxlcyB3aXRoIG5lYXItaWRlbnRpY2FsIDItRCBzdHJ1Y3R1cmUgKFRhbmltb3RvID4gMC45KSBidXQgcG90ZW5jaWVzIGRpZmZlcmluZyBieSBvcmRlcnMgb2YgbWFnbml0dWRlLiBBbnkgZmVhdHVyaXNhdGlvbiBvZiAyLUQgc3RydWN0dXJlIG1hcHMgdGhlbSB0byBuZWFyYnkgdmVjdG9ycywgc28gYSBzbW9vdGggZnVuY3Rpb24gY2Fubm90IHNlcGFyYXRlIHRoZW07IHRoZSBjYXVzZSBsaXZlcyBpbiAzLUQgYmluZGluZyBnZW9tZXRyeSBvciBwcm90b25hdGlvbiwgd2hpY2ggaXMgbm90IGluIHRoZSBpbnB1dC4=",
 "B1": "cElDNTAgPSA5IC0gbG9nMTAoNDUpID0gOSAtIDEuNjUzID0gNy4zNS4gRXJyb3IgPSA3LjkwIC0gNy4zNSA9IDAuNTUgbG9nIHVuaXRzLCBzbyAxMF4wLjU1ID0gMy41LWZvbGQgdG9vIHBvdGVudC4=",
 "B2": "VCA9IGMvKGErYi1jKSA9IDUwLygxODAtNTApID0gNTAvMTI1ID0gMC40MC4gQmVsb3cgdGhlIGNvbnZlbnRpb25hbCAwLjcgdGhyZXNob2xkLCBzbyBub3QgInNpbWlsYXIiIC0gcmVsYXRlZCBidXQgZGlzdGluY3QgY2hlbWlzdHJ5Lg==",
 "B3": "cG9vbGVkID0gc3FydCggWzEoMC4yNSkgKyAyKDAuODEpICsgMSgxLjQ0KV0gLyAoMSsyKzEpICkgPSBzcXJ0KDMuMzEvNCkgPSBzcXJ0KDAuODI3NSkgPSAwLjkxIGxvZyB1bml0cy4=",
 "B4": "MTBeMC42NSA9IDQuNS1mb2xkLiBJbiBlbmVyZ3k6IDAuNjUgeCA1Ljc0IGtKL21vbCA9IDMuNyBrSi9tb2wgLSBhYm91dCBvbmUgd2VhayBoeWRyb2dlbiBib25kLg==",
 "C1": "TGVhdmUtb25lLWtpbmFzZS1vdXQ6IGhvbGQgb3V0IEFMTCByb3dzIG9mIHRoZSB0YXJnZXQga2luYXNlLCB0cmFpbiBvbiB0aGUgb3RoZXIgNzkuIEJhc2VsaW5lcyAtIGdsb2JhbCBtZWFuLCBtZWFuIG9mIHRoZSAxMiBrbm93biBtZWFzdXJlbWVudHMsIGFuZCBhIG1vZGVsIHRyYWluZWQgb24gdGhvc2UgMTIgYWxvbmUuIFdpdGggODAga2luYXNlcyBhIHNlcXVlbmNlIHJlcHJlc2VudGF0aW9uIChFU00yKSBiZWNvbWVzIHdvcnRoIHRlc3RpbmcsIHVubGlrZSB0aGUgMzAtdmFyaWFudCBjYXNlLiBDb252aW5jaW5nIHJlc3VsdDogYmVhdHMgdGhlICIxMiBtZWFzdXJlbWVudHMgb25seSIgYmFzZWxpbmUgYWNyb3NzIG1vc3QgaGVsZC1vdXQga2luYXNlcywgdGVzdGVkIHdpdGggYSBwYWlyZWQgc2lnbiBvciBXaWxjb3hvbiB0ZXN0IC0gbm90IGEgd2luIG9uIG9uZSBraW5hc2UuIFdhdGNoIGZvciBhIHNpbmdsZSBsYXJnZSBraW5hc2UgZG9taW5hdGluZyBhIHJvdy13ZWlnaHRlZCBhdmVyYWdlLg==",
 "C2": "Q29tcHV0ZSBkb2NraW5nIGZvciBhIFNBTVBMRSAoc2F5IDIsMDAwIGNvbXBvdW5kcyksIG5vdCBhbGwgNTAsMDAwLiBFdmFsdWF0ZSBmaW5nZXJwcmludCB2cyBmaW5nZXJwcmludCtkb2NraW5nIHVuZGVyIHRoZSBzYW1lIHNjYWZmb2xkIHNwbGl0LCBzZXZlcmFsIHNlZWRzLCBwYWlyZWQgcGVyIGZvbGQuIEFsc28gcnVuIGRvY2tpbmctYWxvbmUgdnMgc2l6ZS1kZXNjcmlwdG9ycy1hbG9uZSB0byBjYXRjaCB0aGUgc2l6ZS1wcm94eSBjb25mb3VuZC4gRGVjaWRlIHRoZSBydWxlIGZpcnN0LCBlLmcuIHByb2NlZWQgb25seSBpZiBkZWx0YSBSMiA+IDAuMDIgY29uc2lzdGVudGx5IGFjcm9zcyBmb2xkcy4gT3VyIHJlc3VsdDogZGVsdGEgUjIgPSAtMC4wMDExLiBEbyBub3QgcHJvY2VlZC4=",
 "D1": "KDEpIFJhbmRvbSBzcGxpdCBvbiBhbmFsb2d1ZSBzZXJpZXMgLT4gbGVha2FnZTsgaW5mbGF0ZXMuICgyKSAyMDAgdGFyZ2V0cyBwb29sZWQgLT4gdGhlIG1vZGVsIGNhbiBzY29yZSB3ZWxsIGJ5IGxlYXJuaW5nIHBlci10YXJnZXQgbWVhbnM7IGluZmxhdGVzLiAoMykgImFsbCByZXBvcnRlZCB2YWx1ZXMiIGtlZXBzIGNlbnNvcmVkID4vPCBkYXRhIGFzIGlmIG1lYXN1cmVkOyBjb3JydXB0cyBsYWJlbHMgYW5kIHVzdWFsbHkgaW5mbGF0ZXMgYnkgYWRkaW5nIGVhc3ksIHJlcGVhdGVkIHRocmVzaG9sZCB2YWx1ZXMuICg0KSBEdXBsaWNhdGUgcmVjb3JkcyBvZiB0aGUgc2FtZSBjb21wb3VuZC10YXJnZXQgcGFpciBhY3Jvc3MgZm9sZHM7IGluZmxhdGVzLiAoNSkgUk1TRSAwLjMxIGlzIGJlbG93IHRoZSBpbnRlci1sYWIgcmVwcm9kdWNpYmlsaXR5IG9mIENoRU1CTCBJQzUwIGRhdGEgKH4wLjctMC45KSwgd2hpY2ggaXMgcGh5c2ljYWxseSBpbXBvc3NpYmxlIHdpdGhvdXQgbGVha2FnZSAtIHRoZSBzaW5nbGUgbW9zdCBkYW1uaW5nIG51bWJlci4gKDYpIE5vIG5vaXNlLWNlaWxpbmcgY2hlY2ssIG5vIHNjYWZmb2xkIHNwbGl0LCBubyB0cml2aWFsIGJhc2VsaW5lLCBhbmQgdGhlIFJGIGNvbXBhcmF0b3Igd2FzIHByb2JhYmx5IHVudHVuZWQuIE5ldCBkaXJlY3Rpb246IGV2ZXJ5IGZsYXcgcHVzaGVzIHRoZSByZXBvcnRlZCBzY29yZSBVUC4=",
}


def reveal(q="all"):
    ks = sorted(_K) if q == "all" else [q.upper()]
    for k in ks:
        if k not in _K:
            print(f"{k}: no key"); continue
        txt = base64.b64decode(_K[k]).decode()
        print(f"\n── {k} " + "─" * (72 - len(k)))
        print(textwrap.fill(txt, 86))


print("Answer key loaded. Write your answers, then call e.g. reveal('A3') or reveal('all').")
print("Exercises in Part 13 are open-ended by design and have no key — discuss them.")
''')

md(r'''
---
## Where to go next

| Direction | When it is worth it |
|---|---|
| **Chemprop / graph neural networks** | $> 10^4$ measurements per target, and the ceiling is not yet reached |
| **Uncertainty estimation** (conformal prediction, ensembles) | when the model must say *"I don't know"* — usually sooner than you think |
| **Multi-task learning** across targets | when related targets share chemistry |
| **Protein language models** (ESM2) | **many** proteins, not a few variants — see Part 10 |
| **Active learning** | you control which compounds get measured next |
| **Re-measuring the noisy tail** | the highest-value option whenever $\sigma_{\text{exp}}$ dominates |

### Further reading

- Bemis & Murcko, *J. Med. Chem.* **39** (1996) 2887 — scaffold definition
- Rogers & Hahn, *J. Chem. Inf. Model.* **50** (2010) 742 — extended-connectivity fingerprints
- Kramer *et al.*, *J. Med. Chem.* **55** (2012) 5165 — experimental reproducibility of ChEMBL $\mathrm{IC_{50}}$ data
- Sheridan, *J. Chem. Inf. Model.* **53** (2013) 783 — time-split validation
- Wu *et al.*, *Chem. Sci.* **9** (2018) 513 — MoleculeNet benchmarks and splitting

---

*The most valuable result in this notebook is a negative one: two plausible feature sets —
molecular docking and protein language model embeddings — were tested and rejected with numbers.
Knowing what does not work is what makes the rest believable.*
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "colab": {"provenance": [], "toc_visible": True},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
}
out = HERE / "Trustworthy_ML_for_Drug_Discovery.ipynb"
nbf.write(nb, str(out))
print(f"wrote {out}  ({len(cells)} cells, {out.stat().st_size/1024:.0f} KB)")
