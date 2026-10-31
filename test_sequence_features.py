"""¿Aporta la SECUENCIA algo sobre el one-hot de mutaciones?

Con 30 variantes el one-hot está saturado: identifica cada variante y un modelo de árboles
puede representar cualquier función de ella. Dentro de esas 30, nada puede superarlo.
La secuencia solo puede ayudar donde el one-hot es ciego: una variante NO vista al entrenar.

Test: leave-one-variant-out. Se saca una variante mutante completa, se entrena con el resto
(WT incluido) y se predice. Es el escenario real: conoces tus compuestos y aparece una
mutación de resistencia nueva.

  A  one-hot        -> la variante nueva llega con todas las mut_* en cero
  B  secuencia      -> posición y cambio fisicoquímico, generalizan a mutaciones nuevas
  C  ambas

Si B o C ganan a A, representar la proteína por secuencia (ESM2 incluido) vale la pena.
Si no, no lo vale.

Salida: sequence_test_results.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
MIN_ROWS = 30
PARAMS = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
              min_samples_leaf=5, max_features=0.3, random_state=0)
DESC = ["heavy_atoms", "mol_weight", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds",
        "aromatic_rings", "fraction_csp3"]

# Propiedades por aminoácido: hidrofobicidad (Kyte-Doolittle), volumen (A^3), carga, polaridad
AA = {
    "A": (1.8, 88.6, 0, 0), "R": (-4.5, 173.4, 1, 1), "N": (-3.5, 114.1, 0, 1),
    "D": (-3.5, 111.1, -1, 1), "C": (2.5, 108.5, 0, 0), "Q": (-3.5, 143.8, 0, 1),
    "E": (-3.5, 138.4, -1, 1), "G": (-0.4, 60.1, 0, 0), "H": (-3.2, 153.2, 0.1, 1),
    "I": (4.5, 166.7, 0, 0), "L": (3.8, 166.7, 0, 0), "K": (-3.9, 168.6, 1, 1),
    "M": (1.9, 162.9, 0, 0), "F": (2.8, 189.9, 0, 0), "P": (-1.6, 112.7, 0, 0),
    "S": (-0.8, 89.0, 0, 1), "T": (-0.7, 116.1, 0, 1), "W": (-0.9, 227.8, 0, 0),
    "Y": (-1.3, 193.6, 0, 1), "V": (4.2, 140.0, 0, 0),
}


def seq_features(mutation, sequence, wt_sequence):
    """Rasgos de la mutación derivados de la secuencia, no de su nombre."""
    f = dict(n_point=0, n_indel=0, indel_len=0, pos_min=0.0, pos_max=0.0, pos_mean=0.0,
             d_hydro=0.0, d_volume=0.0, d_charge=0.0, d_polar=0.0,
             abs_d_hydro=0.0, abs_d_volume=0.0, len_ratio=len(sequence) / len(wt_sequence))
    if mutation == "Wild Type":
        return f
    f["indel_len"] = len(wt_sequence) - len(sequence)
    positions = []
    for m in mutation.split(", "):
        pm = re.fullmatch(r"([A-Z])(\d+)([A-Z])", m)
        if pm:
            old, pos, new = pm.group(1), int(pm.group(2)), pm.group(3)
            if old in AA and new in AA:
                f["n_point"] += 1
                positions.append(pos / len(wt_sequence))
                for key, i in [("d_hydro", 0), ("d_volume", 1), ("d_charge", 2), ("d_polar", 3)]:
                    f[key] += AA[new][i] - AA[old][i]
                f["abs_d_hydro"] += abs(AA[new][0] - AA[old][0])
                f["abs_d_volume"] += abs(AA[new][1] - AA[old][1])
        else:
            f["n_indel"] += 1
            pm2 = re.search(r"(\d+)", m)
            if pm2:
                positions.append(int(pm2.group(1)) / len(wt_sequence))
    if positions:
        f["pos_min"], f["pos_max"], f["pos_mean"] = min(positions), max(positions), float(np.mean(positions))
    return f


def main():
    d = pd.read_csv(HERE / "kinase_ml_dataset.csv")
    seqs = pd.read_csv(HERE / "variant_sequences.csv")
    z = np.load(HERE / "features_ligand.npz", allow_pickle=True)

    wt = {t: g[g.mutation == "Wild Type"].sequence.iloc[0] for t, g in seqs.groupby("target")}
    sf = pd.DataFrame([{"variant_id": r.variant_id, **seq_features(r.mutation, r.sequence, wt[r.target])}
                       for r in seqs.itertuples()]).set_index("variant_id")
    print("rasgos de secuencia por variante (muestra):", flush=True)
    print(sf.loc[["EGFR|T790M", "EGFR|L858R", "EGFR|L858R+T790M", "ALK|G1202R", "BRAF|V600E"],
                 ["n_point", "pos_mean", "d_hydro", "d_volume", "d_charge"]].round(3).to_string(), flush=True)

    idx = pd.Series(range(len(z["mol_id"])), index=z["mol_id"]).reindex(d["mol_id"]).to_numpy()
    lig = np.hstack([z["morgan2_count"][idx].astype(np.float32), d[DESC].to_numpy(np.float32)])
    lig = lig[:, lig.any(axis=0)]
    base = pd.concat([pd.get_dummies(d["target"], prefix="tgt", dtype=np.float32),
                      pd.get_dummies(d["assay_category"], prefix="assay", dtype=np.float32)],
                     axis=1).to_numpy(np.float32)
    onehot = d[[c for c in d.columns if c.startswith("mut_")]
               + ["n_mutations", "is_wild_type"]].astype(np.float32).to_numpy()
    seqf = sf.reindex(d["variant_id"]).to_numpy(np.float32)

    sets = {"A_onehot": np.hstack([lig, base, onehot]),
            "B_secuencia": np.hstack([lig, base, seqf]),
            "C_ambas": np.hstack([lig, base, onehot, seqf])}
    y = d["pIC50"].to_numpy()
    vid = d["variant_id"].to_numpy()

    counts = d[~d.variant_id.str.contains("Wild Type")].variant_id.value_counts()
    held = sorted(counts[counts >= MIN_ROWS].index)
    print(f"\nvariantes mutantes a dejar fuera: {len(held)} (>= {MIN_ROWS} filas)\n", flush=True)

    rows = []
    for v in held:
        te = vid == v
        tr = ~te
        line = [f"{v:34s} n={te.sum():5d}"]
        for name, X in sets.items():
            p = HistGradientBoostingRegressor(**PARAMS).fit(X[tr], y[tr]).predict(X[te])
            rmse = mean_squared_error(y[te], p) ** 0.5
            rows.append({"variante": v, "n": int(te.sum()), "encoding": name,
                         "RMSE": rmse, "R2": r2_score(y[te], p), "bias": (p - y[te]).mean()})
            line.append(f"{name}: RMSE={rmse:.3f}")
        print("  " + "  ".join(line), flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(HERE / "sequence_test_results.csv", index=False)

    print("\n=== Media ponderada por nº de filas ===", flush=True)
    piv = res.pivot(index="variante", columns="encoding", values="RMSE")
    n = res.pivot(index="variante", columns="encoding", values="n")["A_onehot"]
    for c in piv.columns:
        print(f"  {c:12s} RMSE ponderado = {np.average(piv[c], weights=n):.4f}", flush=True)
    print(f"\n  gana secuencia (B<A) en {(piv.B_secuencia < piv.A_onehot).sum()} de {len(piv)} variantes", flush=True)
    print(f"  gana combinada (C<A) en {(piv.C_ambas < piv.A_onehot).sum()} de {len(piv)} variantes", flush=True)


if __name__ == "__main__":
    main()
