"""¿Aportan los embeddings ESM2 algo sobre el one-hot de mutaciones?

Mismo protocolo que test_sequence_features.py (leave-one-variant-out, mismos hiperparámetros
y mismo bloque de ligando), para que los RMSE sean directamente comparables. El baseline
A_onehot se reutiliza de ese test en lugar de reentrenarlo.

Nota metodológica: los árboles no explotan bien la similitud en espacios de 640 dimensiones,
así que además del embedding crudo se prueba una reducción por PCA. La PCA se ajusta sobre
las 30 variantes sin usar etiquetas, así que no filtra información del target.

Predicción registrada antes de correr: ninguna variante ESM2 mejora el RMSE ponderado
de A_onehot en más de 0.02.

Salida: esm2_test_results.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
MIN_ROWS = 30
N_PCA = 8
PARAMS = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
              min_samples_leaf=5, max_features=0.3, random_state=0)
DESC = ["heavy_atoms", "mol_weight", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds",
        "aromatic_rings", "fraction_csp3"]


def main():
    d = pd.read_csv(HERE / "kinase_ml_dataset.csv")
    z = np.load(HERE / "features_ligand.npz", allow_pickle=True)
    e = np.load(HERE / "esm2_variant_embeddings.npz", allow_pickle=True)

    vids = e["variant_id"]
    emb, dlt, site = e["emb_mean"], e["emb_delta"], e["emb_site"]
    print(f"embeddings: {len(vids)} variantes x {emb.shape[1]} dims", flush=True)

    def red(M):   # PCA sin etiquetas sobre las 30 variantes: no filtra información del target
        return PCA(N_PCA, random_state=0).fit_transform(StandardScaler().fit_transform(M))

    pca_mean = pd.DataFrame(np.hstack([red(emb), red(dlt)]), index=vids)
    pca_site = pd.DataFrame(red(site), index=vids)
    raw_site = pd.DataFrame(site, index=vids)
    print(f"PCA: {N_PCA} componentes por bloque", flush=True)

    idx = pd.Series(range(len(z["mol_id"])), index=z["mol_id"]).reindex(d["mol_id"]).to_numpy()
    lig = np.hstack([z["morgan2_count"][idx].astype(np.float32), d[DESC].to_numpy(np.float32)])
    lig = lig[:, lig.any(axis=0)]
    base = pd.concat([pd.get_dummies(d["target"], prefix="tgt", dtype=np.float32),
                      pd.get_dummies(d["assay_category"], prefix="assay", dtype=np.float32)],
                     axis=1).to_numpy(np.float32)
    onehot = d[[c for c in d.columns if c.startswith("mut_")]
               + ["n_mutations", "is_wild_type"]].astype(np.float32).to_numpy()

    v = d["variant_id"]
    f_mean = pca_mean.reindex(v).to_numpy(np.float32)
    f_site = pca_site.reindex(v).to_numpy(np.float32)
    f_raw = raw_site.reindex(v).to_numpy(np.float32)

    sets = {
        "D_esm2_ventana": np.hstack([lig, base, f_mean]),       # enfoque ingenuo: señal diluida
        "G_esm2_sitio": np.hstack([lig, base, f_site]),         # señal de la mutación intacta
        "H_esm2_sitio_raw": np.hstack([lig, base, f_raw]),      # sin PCA, por si pierde algo
        "F_esm2_sitio_onehot": np.hstack([lig, base, f_site, onehot]),
    }
    y = d["pIC50"].to_numpy()
    vid = v.to_numpy()

    prev = pd.read_csv(HERE / "sequence_test_results.csv")
    held = sorted(prev[prev.encoding == "A_onehot"].variante)
    print(f"variantes a dejar fuera: {len(held)}\n", flush=True)

    rows = []
    for var in held:
        te = vid == var
        tr = ~te
        line = [f"{var:34s} n={te.sum():5d}"]
        for name, X in sets.items():
            p = HistGradientBoostingRegressor(**PARAMS).fit(X[tr], y[tr]).predict(X[te])
            rmse = mean_squared_error(y[te], p) ** 0.5
            rows.append({"variante": var, "n": int(te.sum()), "encoding": name,
                         "RMSE": rmse, "R2": r2_score(y[te], p)})
            line.append(f"{name.split('_', 1)[1]}: {rmse:.3f}")
        print("  " + "  ".join(line), flush=True)

    res = pd.concat([prev[["variante", "n", "encoding", "RMSE", "R2"]], pd.DataFrame(rows)])
    res.to_csv(HERE / "esm2_test_results.csv", index=False)

    piv = res.pivot(index="variante", columns="encoding", values="RMSE")
    n = res.pivot(index="variante", columns="encoding", values="n")["A_onehot"]
    no_braf = ~piv.index.str.contains("V600E")
    print("\n=== RMSE leave-one-variant-out ===", flush=True)
    print(f"{'encoding':20s} {'ponderado':>10s} {'sin V600E':>10s} {'simple':>8s} {'gana a A':>9s}", flush=True)
    for c in piv.columns:
        wins = "" if c == "A_onehot" else f"{int((piv[c] < piv.A_onehot).sum())}/{len(piv)}"
        print(f"{c:20s} {np.average(piv[c], weights=n):10.4f} "
              f"{np.average(piv[c][no_braf], weights=n[no_braf]):10.4f} "
              f"{piv[c].mean():8.4f} {wins:>9s}", flush=True)

    best = min((c for c in piv.columns if c.startswith(("D_", "E_", "F_"))),
               key=lambda c: np.average(piv[c], weights=n))
    gain = np.average(piv.A_onehot, weights=n) - np.average(piv[best], weights=n)
    print(f"\nmejor ESM2: {best}  mejora sobre one-hot = {gain:+.4f} RMSE", flush=True)
    print(f"predicción registrada (<= 0.02): {'FALLA' if gain > 0.02 else 'SE CUMPLE'}", flush=True)


if __name__ == "__main__":
    main()
