"""
Bootstrap para Kaggle Notebook.

Uso:
1. Criar Kaggle Notebook.
2. Copiar este arquivo para uma celula.
3. Rodar. Ele anexa o NIH via kagglehub, acha Data_Entry_2017.csv
   e gera um manifesto pequeno com split por Patient ID.
"""

from pathlib import Path
import shutil
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub", "pandas", "scikit-learn", "tabulate"])

import kagglehub
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATASET = "nih-chest-xrays/data"
WORK = Path("/kaggle/working") if Path("/kaggle").exists() else Path.cwd()

dataset_path = Path(kagglehub.dataset_download(DATASET))
print("Dataset path:", dataset_path)

csv_files = list(dataset_path.rglob("Data_Entry_2017.csv"))
if not csv_files:
    raise FileNotFoundError("Data_Entry_2017.csv nao encontrado no dataset anexado.")

metadata_dir = WORK / "dados" / "raw" / "nih_metadata"
processed_dir = WORK / "dados" / "processed"
results_dir = WORK / "resultados"
for d in [metadata_dir, processed_dir, results_dir]:
    d.mkdir(parents=True, exist_ok=True)

data_csv = csv_files[0]
shutil.copy2(data_csv, metadata_dir / "Data_Entry_2017.csv")
print("Copied:", metadata_dir / "Data_Entry_2017.csv")

df = pd.read_csv(data_csv)
df["Patient Age"] = pd.to_numeric(df["Patient Age"], errors="coerce")
df = df[df["Patient Age"].between(0, 120, inclusive="both")].copy()

pos = df[df["Finding Labels"].astype(str).str.split("|").map(lambda labels: "Cardiomegaly" in labels)].copy()
neg = df[df["Finding Labels"].eq("No Finding")].copy()
pos["target"] = 1
pos["target_label"] = "Cardiomegaly"
neg["target"] = 0
neg["target_label"] = "No Finding"

n = min(1200, len(pos), len(neg))
subset = pd.concat([
    pos.sample(n=n, random_state=565535),
    neg.sample(n=n, random_state=565535),
], ignore_index=True).sample(frac=1, random_state=565535)

gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=565535)
train_idx, temp_idx = next(gss1.split(subset, groups=subset["Patient ID"]))
train = subset.iloc[train_idx].copy()
temp = subset.iloc[temp_idx].copy()

gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=565536)
val_rel, test_rel = next(gss2.split(temp, groups=temp["Patient ID"]))
val = temp.iloc[val_rel].copy()
test = temp.iloc[test_rel].copy()

train["split"] = "train"
val["split"] = "val"
test["split"] = "test"
manifest = pd.concat([train, val, test], ignore_index=True)

sets = {s: set(x["Patient ID"].astype(str)) for s, x in manifest.groupby("split")}
assert not (sets["train"] & sets["val"])
assert not (sets["train"] & sets["test"])
assert not (sets["val"] & sets["test"])

manifest_path = processed_dir / "data_manifest_plan.csv"
manifest.to_csv(manifest_path, index=False)

summary = []
summary.append("# EDA NIH - Cardiomegaly vs No Finding\n")
summary.append(f"Dataset path: `{dataset_path}`\n")
summary.append(f"Manifest: `{manifest_path}`\n")
summary.append(f"Imagens: {len(manifest)}\n")
summary.append(f"Pacientes unicos: {manifest['Patient ID'].nunique()}\n")
summary.append("\n## Split x classe\n")
summary.append(pd.crosstab(manifest["split"], manifest["target_label"]).to_markdown())
summary.append("\n\n## View Position x classe\n")
summary.append(pd.crosstab(manifest["View Position"], manifest["target_label"]).to_markdown())
summary.append("\n\n## Prova anti-vazamento\n")
summary.append("- train_val overlap: 0\n- train_test overlap: 0\n- val_test overlap: 0\n")

summary_path = results_dir / "eda_nih_resumo.md"
summary_path.write_text("\n".join(summary), encoding="utf-8")

print("Manifest saved:", manifest_path)
print("Summary saved:", summary_path)
print("OK: split por Patient ID sem vazamento.")
