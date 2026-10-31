"""Clasificación y ranking de compuestos: priorizar en lugar de predecir el valor exacto.

Evita el problema de los activity cliffs (basta ordenar bien, no acertar la potencia) y es
lo que de verdad se usa para seleccionar qué sintetizar.

La pregunta central: ¿un clasificador dedicado gana a usar el modelo de regresión como
ranker? Se comparan los dos con el MISMO split por scaffold de 5 folds.

Hiperparámetros: se reutilizan los de la regresión (best_params.json) con menos iteraciones,
sin volver a buscar; queda anotado para no confundirlo con un tuning propio.

Salida: ranking_results.csv, ranking_predictions.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
N_FOLDS = 5
THRESHOLDS = {"100nM": 7.0, "10nM": 8.0}
DESC = ["heavy_atoms", "mol_weight", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds",
        "aromatic_rings", "fraction_csp3"]
CLF_PARAMS = dict(max_iter=800, learning_rate=0.06, max_leaf_nodes=63,
                  min_samples_leaf=5, l2_regularization=0.0, max_features=0.3)


def enrichment(y_true, score, frac):
    """Veces que se concentran los activos en el top `frac` del ranking, sobre el azar."""
    k = max(1, int(round(frac * len(score))))
    top = np.argsort(-score)[:k]
    base = y_true.mean()
    return (y_true[top].mean() / base) if base > 0 else np.nan


def rank_metrics(y_true, score, pic50):
    m = {
        "n": len(y_true),
        "pct_activos": y_true.mean() * 100,
        "ROC_AUC": roc_auc_score(y_true, score),
        "PR_AUC": average_precision_score(y_true, score),
        "spearman_vs_pIC50": spearmanr(pic50, score).statistic,
    }
    for f in (0.01, 0.05, 0.10):
        m[f"EF_{int(f*100)}pct"] = enrichment(y_true, score, f)
    k = max(1, int(round(0.05 * len(score))))
    m["precision_top5pct"] = y_true[np.argsort(-score)[:k]].mean() * 100
    return m


def main():
    d = pd.read_csv(HERE / "kinase_ml_dataset.csv")
    z = np.load(HERE / "features_ligand.npz", allow_pickle=True)
    reg = pd.read_csv(HERE / "oof_predictions_v2.csv")["tuned_ensamble5"].to_numpy()

    idx = pd.Series(range(len(z["mol_id"])), index=z["mol_id"]).reindex(d["mol_id"]).to_numpy()
    ctx = pd.concat([
        pd.get_dummies(d["target"], prefix="tgt", dtype=np.float32),
        pd.get_dummies(d["assay_category"], prefix="assay", dtype=np.float32),
        d[[c for c in d.columns if c.startswith("mut_")] + ["n_mutations", "is_wild_type"]].astype(np.float32),
    ], axis=1).to_numpy(np.float32)
    X = np.hstack([ctx, z["morgan3_count"][idx].astype(np.float32),
                   z["maccs"][idx].astype(np.float32), z["avalon"][idx].astype(np.float32),
                   d[DESC].to_numpy(np.float32)])
    X = X[:, X.any(axis=0)]

    pic50 = d["pIC50"].to_numpy()
    folds = d["cv_fold"].to_numpy()
    tgt = d["target"].to_numpy()
    print(f"{X.shape[0]} filas x {X.shape[1]} columnas", flush=True)

    rows, preds = [], {"pIC50": pic50, "ranker_regresion": reg}
    for name, thr in THRESHOLDS.items():
        y = (pic50 >= thr).astype(int)
        # Clasificador dedicado, out-of-fold
        prob = np.zeros(len(y))
        for k in range(N_FOLDS):
            tr, te = folds != k, folds == k
            prob[te] = HistGradientBoostingClassifier(**CLF_PARAMS, random_state=0) \
                .fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            print(f"  [{name}] fold {k} listo", flush=True)
        preds[f"prob_{name}"] = prob
        preds[f"activo_{name}"] = y

        for label, score in [("clasificador", prob), ("ranker_regresion", reg)]:
            rows.append({"umbral": name, "pIC50_min": thr, "modelo": label, "target": "ALL",
                         **rank_metrics(y, score, pic50)})
            for tn in ["EGFR", "BRAF", "ALK"]:
                m = tgt == tn
                rows.append({"umbral": name, "pIC50_min": thr, "modelo": label, "target": tn,
                             **rank_metrics(y[m], score[m], pic50[m])})
        r = pd.DataFrame(rows)
        print(r[(r.umbral == name) & (r.target == "ALL")]
              [["modelo", "ROC_AUC", "PR_AUC", "EF_5pct", "precision_top5pct"]].round(4).to_string(index=False),
              flush=True)

    res = pd.DataFrame(rows)
    for c in res.select_dtypes("float").columns:
        res[c] = res[c].round(4)
    res.to_csv(HERE / "ranking_results.csv", index=False)
    keep = ["mol_id", "target", "variant_id", "assay_category", "cv_fold", "scaffold"]
    pd.concat([d[keep], pd.DataFrame(preds).round(4)], axis=1).to_csv(HERE / "ranking_predictions.csv", index=False)
    with open(HERE / "ranking_params.json", "w") as fh:
        json.dump({"clf_params": CLF_PARAMS, "nota": "reutilizados de la regresion, sin tuning propio"}, fh, indent=2)

    print("\n=== Resumen (todas las dianas) ===", flush=True)
    print(res[res.target == "ALL"][["umbral", "modelo", "pct_activos", "ROC_AUC", "PR_AUC",
                                    "EF_1pct", "EF_5pct", "precision_top5pct"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
