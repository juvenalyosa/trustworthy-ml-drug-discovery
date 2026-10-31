"""Embeddings ESM2 de las 30 variantes.

ESM2 admite ~1024 posiciones y ALK tiene 1620 aa, así que no se embebe la proteína entera:
se toma una ventana que cubre todos los sitios mutados de cada diana con margen.

La ventana se ancla por los DOS extremos (inicio desde el N-terminal, fin desde el C-terminal),
de modo que una deleción dentro de la ventana la acorta de forma natural y ESM2 la "ve".

Para cada variante se guardan tres representaciones:
  emb_mean   media de los embeddings por residuo de la ventana
  emb_delta  emb_mean menos el del WT de su diana
  emb_site   delta EN las posiciones mutadas, contra el WT en las mismas posiciones

emb_site es necesario porque promediar 452 residuos diluye el cambio de uno solo: una
mutación puntual mueve emb_mean apenas 0.5-2% de su norma. emb_site no sufre esa dilución.

Salida: esm2_variant_embeddings.npz
"""
import re
from pathlib import Path

# torch y esm van PRIMERO: importar numpy/pandas antes provoca un segfault por conflicto
# de runtimes de OpenMP (torch 2.2.2 en macOS Intel).
import torch
import esm

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL = "esm2_t30_150M_UR50D"
MARGIN = 150          # aa de contexto a cada lado del sitio mutado más externo
MAX_WINDOW = 1000     # límite de posiciones de ESM2


def windows(seqs):
    """Ventana por diana, en coordenadas del WT, que cubre todos sus sitios mutados."""
    out = {}
    for t, g in seqs.groupby("target"):
        pos = []
        for mutation in g.mutation:
            if mutation == "Wild Type":
                continue
            for m in mutation.split(", "):
                n = re.search(r"(\d+)", m)
                if n:
                    pos.append(int(n.group(1)))
        wt_len = len(g[g.mutation == "Wild Type"].sequence.iloc[0])
        lo = max(0, min(pos) - MARGIN)
        hi = min(wt_len, max(pos) + MARGIN)
        if hi - lo > MAX_WINDOW:                      # recorta centrado si se pasa
            mid = (lo + hi) // 2
            lo, hi = mid - MAX_WINDOW // 2, mid + MAX_WINDOW // 2
        out[t] = (lo, hi, wt_len)
        print(f"  {t}: sitios {min(pos)}-{max(pos)} -> ventana {lo}-{hi} ({hi - lo} aa)", flush=True)
    return out


def main():
    seqs = pd.read_csv(HERE / "variant_sequences.csv")
    print("ventanas por diana:", flush=True)
    win = windows(seqs)

    print(f"\ncargando {MODEL}...", flush=True)
    # La carga vía hub de fair-esm segfaulta con torch 2.2.2 en macOS Intel;
    # se lee el checkpoint del caché con la ruta explícita, que sí funciona.
    ckpt = Path.home() / ".cache/torch/hub/checkpoints" / f"{MODEL}.pt"
    if ckpt.exists():
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(ckpt))
    else:
        model, alphabet = getattr(esm.pretrained, MODEL)()
    model.eval()
    layer = model.num_layers
    batch_converter = alphabet.get_batch_converter()

    def residues(seq, name):
        _, _, toks = batch_converter([(name, seq)])
        with torch.no_grad():
            rep = model(toks, repr_layers=[layer])["representations"][layer]
        return rep[0, 1: len(seq) + 1].numpy()             # sin el token BOS

    # primero el WT de cada diana: hace de referencia para los deltas
    wt_res, wt_sub = {}, {}
    for t, (lo, hi, wt_len) in win.items():
        s = seqs[(seqs.target == t) & (seqs.mutation == "Wild Type")].sequence.iloc[0][lo:hi]
        wt_res[t], wt_sub[t] = residues(s, f"{t}_WT"), s

    rows = []
    for r in seqs.itertuples():
        lo, hi, wt_len = win[r.target]
        # ancla inicio por el N-terminal y fin por el C-terminal: los indels acortan la ventana
        sub = r.sequence[lo: len(r.sequence) - (wt_len - hi)]
        res = residues(sub, r.variant_id)
        shift = wt_len - len(r.sequence)      # aa perdidos por deleciones (negativo si inserción)

        site = np.zeros(res.shape[1], dtype=np.float32)
        if r.mutation != "Wild Type":
            deltas = []
            for m in r.mutation.split(", "):
                num = re.search(r"(\d+)", m)
                if not num:
                    continue
                p = int(num.group(1))
                i_wt, i_var = p - 1 - lo, p - 1 - shift - lo
                if 0 <= i_wt < len(wt_res[r.target]) and 0 <= i_var < len(res):
                    deltas.append(res[i_var] - wt_res[r.target][i_wt])
            if deltas:
                site = np.mean(deltas, axis=0)

        rows.append({"variant_id": r.variant_id, "target": r.target, "len": len(sub),
                     "emb": res.mean(0), "site": site})
        print(f"  {r.variant_id:34s} ventana {len(sub)} aa  |site delta|={np.linalg.norm(site):.3f}", flush=True)

    df = pd.DataFrame(rows)
    E = np.vstack(df.emb.to_numpy())
    S = np.vstack(df.site.to_numpy())
    wt_emb = {t: E[(df.target == t).to_numpy() & df.variant_id.str.contains("Wild Type").to_numpy()][0]
              for t in df.target.unique()}
    D = E - np.vstack([wt_emb[t] for t in df.target])

    np.savez_compressed(HERE / "esm2_variant_embeddings.npz",
                        variant_id=df.variant_id.to_numpy(), emb_mean=E, emb_delta=D, emb_site=S)
    print(f"\nguardado: {E.shape[0]} variantes x {E.shape[1]} dims (mean, delta, site)", flush=True)

    rel_d = np.linalg.norm(D, axis=1) / np.linalg.norm(E, axis=1) * 100
    rel_s = np.linalg.norm(S, axis=1) / np.linalg.norm(E, axis=1) * 100
    print("\nseñal relativa al tamaño del embedding (%):", flush=True)
    print(f"  delta de la ventana completa: mediana {np.median(rel_d[rel_d > 0]):.2f}%", flush=True)
    print(f"  delta en el sitio mutado    : mediana {np.median(rel_s[rel_s > 0]):.2f}%", flush=True)


if __name__ == "__main__":
    main()
