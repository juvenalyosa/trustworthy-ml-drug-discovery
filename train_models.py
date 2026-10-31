"""Baseline de regresión de pIC50 con validación cruzada de 5 folds por scaffold.

Compara juegos de variables para medir cuánto aporta el docking sobre la estructura,
y repite un juego con folds aleatorios para cuantificar la fuga de datos.

Salidas:
  model_results.csv    una fila por (juego de variables, modelo, diana) con las métricas
  oof_predictions.csv  predicciones out-of-fold para analizar errores
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
SEED = 0
N_FOLDS = 5
DOCK = ["dock_best", "dock_top3_mean", "dock_mean", "dock_std", "dock_LE", "dock_size_resid", "dock_z_variant"]
DESC = ["heavy_atoms", "mol_weight", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds",
        "aromatic_rings", "fraction_csp3"]


def metrics(y, p):
    return {
        "n": len(y),
        "R2": r2_score(y, p),
        "RMSE": mean_squared_error(y, p) ** 0.5,
        "MAE": mean_absolute_error(y, p),
        "spearman": spearmanr(y, p).statistic,
        "pct_within_1log": (np.abs(y - p) < 1).mean() * 100,
    }


def run_cv(X, y, folds, model_fn, label, feats, target, results, oof_store):
    """Entrena un modelo por fold y evalúa con las predicciones out-of-fold agrupadas."""
    oof = np.full(len(y), np.nan)
    times = []
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        t = time.time()
        oof[te] = model_fn().fit(X[tr], y[tr]).predict(X[te])
        times.append(time.time() - t)
        print(f"  [{feats}/{label}] fold {k}: {time.time() - t:.0f}s", flush=True)
    oof_store[f"{feats}__{label}"] = oof
    row = {"features": feats, "model": label, "split": "scaffold", "target": "ALL",
           "fit_seconds": round(np.mean(times), 1), **metrics(y, oof)}
    results.append(row)
    for t_name, m in [(t, target == t) for t in ["EGFR", "BRAF", "ALK"]]:
        results.append({"features": feats, "model": label, "split": "scaffold", "target": t_name,
                        "fit_seconds": np.nan, **metrics(y[m], oof[m])})
    print(f"  -> {feats}/{label}: R2={row['R2']:.3f} RMSE={row['RMSE']:.3f}", flush=True)
    return oof


def main():
    d = pd.read_csv(HERE / "kinase_ml_dataset.csv")
    fp = pd.read_csv(HERE / "fingerprints_ecfp4.csv")

    # Contexto del ensayo: diana, formato y mutaciones. Va en todos los juegos de variables,
    # porque el mismo compuesto tiene distinta potencia según la variante medida.
    ctx = pd.concat([
        pd.get_dummies(d["target"], prefix="tgt", dtype=np.float32),
        pd.get_dummies(d["assay_category"], prefix="assay", dtype=np.float32),
        d[[c for c in d.columns if c.startswith("mut_")] + ["n_mutations", "is_wild_type"]].astype(np.float32),
    ], axis=1)
    ecfp = d[["mol_id"]].merge(fp, on="mol_id", how="left").drop(columns="mol_id").astype(np.float32)

    sets = {
        "ctx_only": ctx,
        "dock_desc": pd.concat([ctx, d[DOCK + DESC].astype(np.float32)], axis=1),
        "ecfp": pd.concat([ctx, ecfp], axis=1),
        "ecfp_dock_desc": pd.concat([ctx, ecfp, d[DOCK + DESC].astype(np.float32)], axis=1),
    }
    y = d["pIC50"].to_numpy()
    folds = d["cv_fold"].to_numpy()
    target = d["target"].to_numpy()
    print(f"{len(d)} filas | juegos de variables: " + ", ".join(f"{k}={v.shape[1]}" for k, v in sets.items()), flush=True)

    results, oof_store = [], {}
    gbm = lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=SEED)

    # Referencia trivial: predecir siempre la media del fold de entrenamiento.
    dummy = np.concatenate([np.full((folds == k).sum(), y[folds != k].mean()) for k in range(N_FOLDS)])
    order = np.concatenate([np.where(folds == k)[0] for k in range(N_FOLDS)])
    base = np.empty(len(y)); base[order] = dummy
    results.append({"features": "ninguna", "model": "media", "split": "scaffold", "target": "ALL",
                    "fit_seconds": 0.0, **metrics(y, base)})

    for feats, X in sets.items():
        Xa = X.to_numpy(np.float32)
        run_cv(Xa, y, folds, gbm, "HistGB", feats, target, results, oof_store)
        if feats in ("ecfp", "ecfp_dock_desc"):   # lineal: referencia clásica en QSAR
            run_cv(Xa, y, folds, lambda: RidgeCV(alphas=np.logspace(-1, 3, 9)), "Ridge",
                   feats, target, results, oof_store)

    # Mismo modelo con folds aleatorios por fila: mide la fuga de datos de un split ingenuo.
    rng = np.random.default_rng(SEED)
    rand_folds = rng.integers(0, N_FOLDS, len(y))
    X = sets["ecfp_dock_desc"].to_numpy(np.float32)
    oof = np.full(len(y), np.nan)
    for k in range(N_FOLDS):
        tr, te = rand_folds != k, rand_folds == k
        oof[te] = gbm().fit(X[tr], y[tr]).predict(X[te])
        print(f"  [aleatorio] fold {k} listo", flush=True)
    oof_store["ecfp_dock_desc__random"] = oof
    results.append({"features": "ecfp_dock_desc", "model": "HistGB", "split": "aleatorio", "target": "ALL",
                    "fit_seconds": np.nan, **metrics(y, oof)})

    res = pd.DataFrame(results)
    for c in ["R2", "RMSE", "MAE", "spearman", "pct_within_1log"]:
        res[c] = res[c].round(4)
    res.to_csv(HERE / "model_results.csv", index=False)

    keep = ["mol_id", "target", "mutation", "variant_id", "assay_category", "scaffold", "cv_fold", "pIC50"]
    pd.concat([d[keep], pd.DataFrame(oof_store).round(4)], axis=1).to_csv(HERE / "oof_predictions.csv", index=False)

    print("\n=== Resultados (todas las dianas, split por scaffold) ===", flush=True)
    print(res[res.target == "ALL"].drop(columns="target").to_string(index=False), flush=True)
    print("\n=== Por diana (HistGB) ===", flush=True)
    print(res[(res.target != "ALL") & (res.model == "HistGB")]
          .pivot(index="features", columns="target", values="R2").to_string(), flush=True)


if __name__ == "__main__":
    main()
