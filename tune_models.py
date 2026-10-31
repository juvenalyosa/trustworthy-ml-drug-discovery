"""Mejora del modelo de pIC50 sin datos nuevos: representaciones y ajuste de hiperparámetros.

Fase A  compara bloques de fingerprints con los MISMOS hiperparámetros del baseline,
        para que el R2 sea comparable con el 0.5719 de ECFP4 binario (CV 5 folds por scaffold).
Fase B  busca hiperparámetros usando SOLO folds 0-2 (train) y fold 3 (valid).
        El fold 4 no se toca, así queda un test limpio.
Fase C  evalúa la mejor configuración con CV completa y un ensamble de semillas.

Salida: tuning_results.csv, oof_predictions_v2.csv
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
N_FOLDS = 5
BASELINE = dict(max_iter=400, learning_rate=0.06, random_state=0)   # igual que train_models.py
DESC = ["heavy_atoms", "mol_weight", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds",
        "aromatic_rings", "fraction_csp3"]
GRID = {
    "max_iter": [400, 800, 1500],
    "learning_rate": [0.03, 0.06, 0.1],
    "max_leaf_nodes": [31, 63, 127],
    "min_samples_leaf": [5, 20, 50],
    "l2_regularization": [0.0, 1.0, 10.0],
    "max_features": [0.3, 0.6, 1.0],
}
N_SEARCH = 15
rows = []


def metrics(y, p):
    return {"R2": r2_score(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "MAE": mean_absolute_error(y, p), "spearman": spearmanr(y, p).statistic,
            "pct_within_1log": (np.abs(y - p) < 1).mean() * 100}


def cv_oof(X, y, folds, params, seeds=(0,)):
    """Predicciones out-of-fold; con varias semillas promedia (ensamble)."""
    oof = np.zeros(len(y))
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        preds = [HistGradientBoostingRegressor(**{**params, "random_state": s}).fit(X[tr], y[tr]).predict(X[te])
                 for s in seeds]
        oof[te] = np.mean(preds, axis=0)
    return oof


def main():
    d = pd.read_csv(HERE / "kinase_ml_dataset.csv")
    z = np.load(HERE / "features_ligand.npz", allow_pickle=True)
    assert list(z["mol_id"]) == sorted(d["mol_id"].unique())
    idx = pd.Series(range(len(z["mol_id"])), index=z["mol_id"]).reindex(d["mol_id"]).to_numpy()

    ctx = pd.concat([
        pd.get_dummies(d["target"], prefix="tgt", dtype=np.float32),
        pd.get_dummies(d["assay_category"], prefix="assay", dtype=np.float32),
        d[[c for c in d.columns if c.startswith("mut_")] + ["n_mutations", "is_wild_type"]].astype(np.float32),
    ], axis=1).to_numpy(np.float32)
    desc = d[DESC].to_numpy(np.float32)

    def block(*names, with_desc=False):
        parts = [ctx] + [z[n][idx] for n in names] + ([desc] if with_desc else [])
        X = np.hstack([p.astype(np.float32) for p in parts])
        return X[:, X.any(axis=0)]   # fuera columnas vacías: no aportan y solo cuestan tiempo

    y = d["pIC50"].to_numpy()
    folds = d["cv_fold"].to_numpy()
    tgt = d["target"].to_numpy()

    # ---------------------------------------------------------------- Fase A
    print("=== Fase A: bloques de fingerprints (hiperparámetros del baseline) ===", flush=True)
    blocks = {
        "morgan2_count": block("morgan2_count"),
        "morgan3_count": block("morgan3_count"),
        "morgan2+maccs+avalon+desc": block("morgan2_count", "maccs", "avalon", with_desc=True),
        "morgan3+maccs+avalon+desc": block("morgan3_count", "maccs", "avalon", with_desc=True),
    }
    oof_store, best_block, best_r2 = {}, None, -9
    for name, X in blocks.items():
        t = time.time()
        oof = cv_oof(X, y, folds, BASELINE)
        m = metrics(y, oof)
        rows.append({"phase": "A", "config": name, "params": "baseline", "target": "ALL", **m})
        for tn in ["EGFR", "BRAF", "ALK"]:
            rows.append({"phase": "A", "config": name, "params": "baseline", "target": tn,
                         **metrics(y[tgt == tn], oof[tgt == tn])})
        oof_store[name] = oof
        print(f"  {name}: {X.shape[1]} cols  R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  ({time.time()-t:.0f}s)", flush=True)
        if m["R2"] > best_r2:
            best_r2, best_block = m["R2"], name
    print(f"  -> mejor bloque: {best_block} (R2={best_r2:.4f})", flush=True)
    X = blocks[best_block]

    # ---------------------------------------------------------------- Fase B
    print("\n=== Fase B: búsqueda de hiperparámetros (train=folds 0-2, valid=fold 3) ===", flush=True)
    tr, va = np.isin(folds, [0, 1, 2]), folds == 3
    rng = np.random.default_rng(0)
    configs = [{k: GRID[k][int(i)] for k, i in zip(GRID, idxs)}
               for idxs in rng.integers(0, 3, size=(N_SEARCH, len(GRID)))]
    configs.insert(0, {k: v for k, v in BASELINE.items() if k != "random_state"})

    best_params, best_va = None, -9
    for c in configs:
        t = time.time()
        p = HistGradientBoostingRegressor(**c, random_state=0).fit(X[tr], y[tr]).predict(X[va])
        r2 = r2_score(y[va], p)
        rows.append({"phase": "B", "config": best_block, "params": json.dumps(c, sort_keys=True),
                     "target": "VALID_fold3", **metrics(y[va], p)})
        print(f"  R2_valid={r2:.4f}  {c}  ({time.time()-t:.0f}s)", flush=True)
        if r2 > best_va:
            best_va, best_params = r2, c
    print(f"  -> mejores: {best_params}  (R2_valid={best_va:.4f})", flush=True)

    # ---------------------------------------------------------------- Fase C
    print("\n=== Fase C: configuración final ===", flush=True)
    te4 = folds == 4
    tr4 = ~te4
    p4 = HistGradientBoostingRegressor(**best_params, random_state=0).fit(X[tr4], y[tr4]).predict(X[te4])
    rows.append({"phase": "C", "config": best_block, "params": "tuned", "target": "TEST_fold4_limpio",
                 **metrics(y[te4], p4)})
    print(f"  fold 4 (nunca usado en el tuning): R2={r2_score(y[te4], p4):.4f}", flush=True)

    for label, seeds in [("tuned", (0,)), ("tuned_ensamble5", (0, 1, 2, 3, 4))]:
        t = time.time()
        oof = cv_oof(X, y, folds, best_params, seeds=seeds)
        m = metrics(y, oof)
        oof_store[label] = oof
        rows.append({"phase": "C", "config": best_block, "params": label, "target": "ALL", **m})
        for tn in ["EGFR", "BRAF", "ALK"]:
            rows.append({"phase": "C", "config": best_block, "params": label, "target": tn,
                         **metrics(y[tgt == tn], oof[tgt == tn])})
        print(f"  {label}: R2={m['R2']:.4f} RMSE={m['RMSE']:.4f} "
              f"within1log={m['pct_within_1log']:.1f}%  ({time.time()-t:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    for c in ["R2", "RMSE", "MAE", "spearman", "pct_within_1log"]:
        res[c] = res[c].round(4)
    res.to_csv(HERE / "tuning_results.csv", index=False)
    keep = ["mol_id", "target", "mutation", "variant_id", "assay_category", "scaffold", "cv_fold", "pIC50"]
    pd.concat([d[keep], pd.DataFrame(oof_store).round(4)], axis=1).to_csv(HERE / "oof_predictions_v2.csv", index=False)
    json.dump(best_params, open(HERE / "best_params.json", "w"), indent=2)

    print("\n=== Resumen (todas las dianas) ===", flush=True)
    print(res[res.target == "ALL"][["phase", "config", "params", "R2", "RMSE", "pct_within_1log"]]
          .to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
