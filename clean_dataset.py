"""Limpieza y normalización de egfr_alk_braf_merged para modelos de ML.

Salidas (en el mismo directorio):
  kinase_ml_dataset.csv   una fila por (estructura, diana, mutación, tipo de ensayo)
  variant_sequences.csv   variant_id -> secuencia de la proteína
  fingerprints_ecfp4.csv  mol_id -> Morgan r=2, 2048 bits (una fila por estructura)
  cleaning_log.txt        filas eliminadas en cada paso
"""
import ast
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
SRC = HERE / "egfr_alk_braf_merged With ALL docking scores, Cleaned.xlsx"

TARGETS = {
    "Epidermal growth factor receptor erbB1": "EGFR",
    "Serine/threonine-protein kinase B-raf": "BRAF",
    "ALK tyrosine kinase receptor": "ALK",
}
ASSAY_CATEGORY = {
    "single protein format": "biochemical",
    "protein format": "biochemical",
    "cell-free format": "biochemical",
    "cell membrane format": "biochemical",
    "cell-based format": "cellular",
    "assay format": "other",
}
PIC50_RANGE = (3.0, 12.0)   # 1 mM .. 1 pM
MAX_REPLICATE_STD = 1.0     # unidades log; grupos más dispersos se descartan
FP_RADIUS, FP_BITS = 2, 2048
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)   # train / valid / test
N_FOLDS = 5
SEED = 42

log = []


def step(msg, before, after):
    line = f"{msg}: {before} -> {after} (-{before - after})"
    log.append(line)
    print(line)


# ---------------------------------------------------------------- estructuras
_cleanup = rdMolStandardize.CleanupParameters()
_largest = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = rdMolStandardize.Cleanup(mol, _cleanup)
    mol = _largest.choose(mol)          # quita sales / contraiones
    mol = _uncharger.uncharge(mol)      # neutraliza
    return mol


def descriptors(mol):
    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "mol_weight": Descriptors.MolWt(mol),
        "clogp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


# ---------------------------------------------------------------- docking
def pose_stats(raw):
    """Recalcula estadísticas de docking descartando poses con puntuación >= 0 (choques)."""
    scores = np.array(ast.literal_eval(raw), dtype=float)
    valid = np.sort(scores[scores < 0])
    if valid.size == 0:
        return pd.Series(dtype=float)
    return pd.Series({
        "dock_best": valid[0],
        "dock_top3_mean": valid[:3].mean(),
        "dock_mean": valid.mean(),
        "dock_median": np.median(valid),
        "dock_std": valid.std(ddof=1) if valid.size > 1 else 0.0,
        "n_poses_valid": valid.size,
        "n_poses_discarded": scores.size - valid.size,
    })


# ---------------------------------------------------------------- scaffolds
def murcko(mol, smiles):
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaf or smiles   # moléculas acíclicas: cada una es su propio grupo


def scaffold_split(scaffolds, weights):
    """Split balanceado por scaffold (estilo chemprop): los scaffolds grandes van a train,
    el resto se reparte al azar. Las proporciones se miden en filas del dataset."""
    total = weights.sum()
    n_train, n_valid = SPLIT_FRACTIONS[0] * total, SPLIT_FRACTIONS[1] * total
    big = [s for s in scaffolds if weights[s] > SPLIT_FRACTIONS[2] * total / 2]
    small = [s for s in scaffolds if weights[s] <= SPLIT_FRACTIONS[2] * total / 2]
    random.Random(SEED).shuffle(small)
    assign, filled = {}, {"train": 0, "valid": 0}
    for s in big + small:
        if filled["train"] + weights[s] <= n_train:
            part = "train"
        elif filled["valid"] + weights[s] <= n_valid:
            part = "valid"
        else:
            part = "test"
        assign[s] = part
        filled[part] = filled.get(part, 0) + weights[s]
    return assign


def scaffold_folds(scaffolds, weights):
    """K folds agrupados por scaffold: cada grupo va al fold con menos filas."""
    order = sorted(scaffolds)
    random.Random(SEED).shuffle(order)
    order.sort(key=lambda s: -weights[s])   # sort estable: empates quedan en orden aleatorio fijo
    sizes, assign = [0] * N_FOLDS, {}
    for s in order:
        k = sizes.index(min(sizes))
        assign[s] = k
        sizes[k] += weights[s]
    return assign


def main():
    df = pd.read_excel(SRC)
    n0 = len(df)
    log.append(f"Filas de entrada: {n0}")
    df["target"] = df["target_pref_name"].map(TARGETS)
    df["mutation"] = df["assay_variant_mutation"]
    df["assay_category"] = df["bao_label"].map(ASSAY_CATEGORY)

    # 1. Solo valores exactos (los censurados >, < no sirven para regresión)
    n = len(df)
    df = df[df["standard_relation"].isin(["=", "~"])]
    step("Relación exacta (= o ~)", n, len(df))

    # 2. Estructuras estandarizadas
    mols = {s: standardize(s) for s in df["canonical_smiles"].unique()}
    n = len(df)
    df = df[df["canonical_smiles"].map(lambda s: mols[s] is not None)]
    step("SMILES válidos", n, len(df))
    df["smiles_std"] = df["canonical_smiles"].map(lambda s: Chem.MolToSmiles(mols[s]))
    mw_orig = {s: Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in df["canonical_smiles"].unique()}

    # 3. Unidades -> nM (ug/mL se convierte con el PM de la forma registrada; /uM se descarta)
    is_ugml = df["standard_units"] == "ug.mL-1"
    df.loc[is_ugml, "standard_value"] = (
        df.loc[is_ugml, "standard_value"] / df.loc[is_ugml, "canonical_smiles"].map(mw_orig) * 1e6
    )
    log.append(f"Convertidas de ug/mL a nM: {int(is_ugml.sum())}")
    n = len(df)
    df = df[df["standard_units"].isin(["nM", "ug.mL-1"]) & (df["standard_value"] > 0)]
    step("Unidades convertibles y valor > 0", n, len(df))

    # 4. pIC50 en rango físico
    df["pIC50"] = 9 - np.log10(df["standard_value"])
    n = len(df)
    df = df[df["pIC50"].between(*PIC50_RANGE)]
    step(f"pIC50 en [{PIC50_RANGE[0]}, {PIC50_RANGE[1]}]", n, len(df))

    # 5. Docking recalculado sin poses positivas
    dock = df["All docking scores"].apply(pose_stats)
    df = df.join(dock)
    n = len(df)
    df = df[df["dock_best"].notna()]
    step("Al menos una pose válida", n, len(df))
    log.append(f"Filas con poses descartadas (>= 0): {int((df['n_poses_discarded'] > 0).sum())}")

    # 6. Variantes -> id corto; la secuencia va en un archivo aparte
    seqs = (df[["target", "mutation", "variant_mutation_sequence"]].drop_duplicates()
            .sort_values(["target", "mutation"]).reset_index(drop=True))
    seqs["variant_id"] = seqs["target"] + "|" + seqs["mutation"].str.replace(", ", "+")
    df = df.merge(seqs, on=["target", "mutation", "variant_mutation_sequence"])

    # 7. Agregar réplicas por (estructura, variante, tipo de ensayo)
    key = ["smiles_std", "target", "mutation", "variant_id", "assay_category"]
    dock_cols = ["dock_best", "dock_top3_mean", "dock_mean", "dock_median", "dock_std", "n_poses_valid"]
    agg = df.groupby(key).agg(
        molecule_chembl_ids=("molecule_chembl_id", lambda s: ";".join(sorted(set(s)))),
        molecule_pref_name=("molecule_pref_name", lambda s: next((x for x in s if x != "Research Molecule"), "")),
        pIC50=("pIC50", "median"),
        pIC50_mean=("pIC50", "mean"),
        pIC50_std=("pIC50", "std"),
        n_measurements=("pIC50", "size"),
        n_documents=("document_chembl_id", "nunique"),
        first_year=("document_year", "min"),
        **{c: (c, "median") for c in dock_cols},
    ).reset_index()
    agg["pIC50_std"] = agg["pIC50_std"].fillna(0.0)
    step("Agregación de réplicas (filas -> grupos)", len(df), len(agg))
    n = len(agg)
    agg = agg[agg["pIC50_std"] <= MAX_REPLICATE_STD]
    step(f"Réplicas con std pIC50 <= {MAX_REPLICATE_STD}", n, len(agg))

    # 8. Descriptores
    smi_to_mol = {Chem.MolToSmiles(m): m for m in mols.values() if m is not None}
    desc = pd.DataFrame([{"smiles_std": s, **descriptors(smi_to_mol[s])} for s in agg["smiles_std"].unique()])
    agg = agg.merge(desc, on="smiles_std")

    # 9. Normalizaciones del docking
    agg["dock_LE"] = -agg["dock_best"] / agg["heavy_atoms"]            # eficiencia de ligando
    agg["dock_LE_size_corr"] = -agg["dock_best"] / agg["heavy_atoms"] ** 0.3
    # residuo del ajuste lineal dock_best ~ heavy_atoms dentro de cada diana (quita el sesgo por tamaño)
    uniq = agg.drop_duplicates(["smiles_std", "variant_id"])
    agg["dock_size_resid"] = np.nan
    for t, g in uniq.groupby("target"):
        slope, icpt = np.polyfit(g["heavy_atoms"], g["dock_best"], 1)
        m = agg["target"] == t
        agg.loc[m, "dock_size_resid"] = agg.loc[m, "dock_best"] - (slope * agg.loc[m, "heavy_atoms"] + icpt)
    # z-score dentro de cada variante (quita el sesgo sistemático de cada estructura modelada)
    stats = uniq.groupby("variant_id")["dock_best"].agg(["mean", "std"])
    agg["dock_z_variant"] = ((agg["dock_best"] - agg["variant_id"].map(stats["mean"]))
                             / agg["variant_id"].map(stats["std"])).fillna(0.0)  # variantes con 1 sola molécula

    # 10. Codificación de mutaciones (una columna binaria por mutación puntual)
    atomic = sorted({m for s in agg["mutation"] for m in s.split(", ") if m != "Wild Type"})
    for m in atomic:
        agg[f"mut_{m}"] = agg["mutation"].str.split(", ").map(lambda parts, m=m: int(m in parts))
    agg["n_mutations"] = agg["mutation"].map(lambda s: 0 if s == "Wild Type" else len(s.split(", ")))
    agg["is_wild_type"] = (agg["n_mutations"] == 0).astype(int)

    # 11. Id de estructura, scaffold de Murcko y particiones (una estructura nunca cruza particiones)
    uniq_smiles = sorted(agg["smiles_std"].unique())
    mol_ids = {s: f"M{i:05d}" for i, s in enumerate(uniq_smiles, 1)}
    agg["mol_id"] = agg["smiles_std"].map(mol_ids)
    scaf = {s: murcko(smi_to_mol[s], s) for s in uniq_smiles}
    agg["scaffold"] = agg["smiles_std"].map(scaf)
    weights = agg["scaffold"].value_counts()
    agg["split"] = agg["scaffold"].map(scaffold_split(list(weights.index), weights))
    agg["cv_fold"] = agg["scaffold"].map(scaffold_folds(list(weights.index), weights))
    log.append(f"Scaffolds de Murcko: {len(weights)}")

    # 12. Fingerprints Morgan (ECFP4) por estructura, en archivo aparte
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    fps = np.array([gen.GetFingerprintAsNumPy(smi_to_mol[s]) for s in uniq_smiles], dtype=np.uint8)
    fp_df = pd.DataFrame(fps, columns=[f"ecfp4_{i}" for i in range(FP_BITS)])
    fp_df.insert(0, "mol_id", [mol_ids[s] for s in uniq_smiles])
    fp_df.to_csv(HERE / "fingerprints_ecfp4.csv", index=False)

    front = ["mol_id", "smiles_std", "molecule_chembl_ids", "molecule_pref_name", "target", "mutation", "variant_id",
             "assay_category", "split", "cv_fold", "scaffold",
             "pIC50", "pIC50_mean", "pIC50_std", "n_measurements", "n_documents", "first_year"]
    agg = agg[front + [c for c in agg.columns if c not in front]]
    agg = agg.sort_values(["target", "variant_id", "assay_category", "pIC50"], ascending=[True, True, True, False])
    for c in agg.select_dtypes("float").columns:
        agg[c] = agg[c].round(4)

    agg.to_csv(HERE / "kinase_ml_dataset.csv", index=False)
    seqs[["variant_id", "target", "mutation", "variant_mutation_sequence"]].rename(
        columns={"variant_mutation_sequence": "sequence"}).to_csv(HERE / "variant_sequences.csv", index=False)

    log.append("")
    log.append(f"Dataset final: {len(agg)} filas, {agg['smiles_std'].nunique()} estructuras únicas, {agg.shape[1]} columnas")
    log.append(agg.groupby(["target", "assay_category"]).size().unstack(fill_value=0).to_string())
    log.append("")
    log.append(agg.groupby(["target", "split"]).size().unstack(fill_value=0).to_string())
    log.append("")
    log.append("Filas por cv_fold: " + agg["cv_fold"].value_counts().sort_index().to_dict().__repr__())
    (HERE / "cleaning_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log[-3:]))


if __name__ == "__main__":
    sys.exit(main())
