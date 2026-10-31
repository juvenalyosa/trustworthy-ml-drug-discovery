"""Genera representaciones del ligando más ricas que el ECFP4 binario.

Morgan contado conserva cuántas veces aparece cada subestructura, no solo si aparece;
MACCS y Avalon aportan subestructuras definidas a mano que Morgan no codifica igual.

Salida: features_ligand.npz  (matrices por bloque + mol_id, alineado con mol_id del dataset)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent

BLOCKS = {
    "morgan2_count": lambda m: rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, countSimulation=False).GetCountFingerprintAsNumPy(m),
    "morgan3_count": lambda m: rdFingerprintGenerator.GetMorganGenerator(
        radius=3, fpSize=2048, countSimulation=False).GetCountFingerprintAsNumPy(m),
    "maccs": lambda m: np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8),
    "avalon": lambda m: np.array(pyAvalonTools.GetAvalonFP(m, nBits=512), dtype=np.uint8),
}


def main():
    d = pd.read_csv(HERE / "kinase_ml_dataset.csv", usecols=["mol_id", "smiles_std"])
    mols = d.drop_duplicates("mol_id").sort_values("mol_id").reset_index(drop=True)
    print(f"{len(mols)} estructuras únicas", flush=True)

    rd_mols = [Chem.MolFromSmiles(s) for s in mols["smiles_std"]]
    assert all(m is not None for m in rd_mols)

    out = {"mol_id": mols["mol_id"].to_numpy()}
    for name, fn in BLOCKS.items():
        M = np.vstack([fn(m) for m in rd_mols])
        # Los contados se recortan a 255 para que quepan en uint8 (valores mayores son rarísimos).
        M = np.clip(M, 0, 255).astype(np.uint8)
        out[name] = M
        nz = (M > 0).sum(0)
        print(f"{name}: {M.shape[1]} columnas, {(nz > 0).sum()} no vacías, "
              f"max cuenta {M.max()}", flush=True)

    np.savez_compressed(HERE / "features_ligand.npz", **out)
    print(f"guardado features_ligand.npz "
          f"({(HERE / 'features_ligand.npz').stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
