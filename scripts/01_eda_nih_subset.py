from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "dados" / "raw" / "nih_metadata" / "Data_Entry_2017.csv"
OUT_DIR = ROOT / "dados" / "processed"
RESULTS_DIR = ROOT / "resultados"


def has_label(value: str, label: str) -> bool:
    return label in str(value).split("|")


def split_by_patient(df: pd.DataFrame, seed: int = 565535) -> pd.DataFrame:
    splitter_1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, temp_idx = next(splitter_1.split(df, groups=df["Patient ID"]))

    train = df.iloc[train_idx].copy()
    temp = df.iloc[temp_idx].copy()

    splitter_2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 1)
    val_rel_idx, test_rel_idx = next(splitter_2.split(temp, groups=temp["Patient ID"]))

    val = temp.iloc[val_rel_idx].copy()
    test = temp.iloc[test_rel_idx].copy()

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    return pd.concat([train, val, test], ignore_index=True)


def summarize(df: pd.DataFrame) -> str:
    lines = []
    lines.append("# EDA NIH Chest X-ray14 - Cardiomegaly vs No Finding")
    lines.append("")
    lines.append(f"Total imagens no manifesto: {len(df):,}")
    lines.append(f"Pacientes unicos: {df['Patient ID'].nunique():,}")
    lines.append("")
    lines.append("## Classes")
    lines.append(df["target_label"].value_counts().to_markdown())
    lines.append("")
    lines.append("## Split x classe")
    lines.append(pd.crosstab(df["split"], df["target_label"]).to_markdown())
    lines.append("")
    lines.append("## Split x pacientes unicos")
    lines.append(df.groupby("split")["Patient ID"].nunique().to_markdown())
    lines.append("")
    if "Patient Gender" in df.columns:
        lines.append("## Genero x classe")
        lines.append(pd.crosstab(df["Patient Gender"], df["target_label"]).to_markdown())
        lines.append("")
    if "View Position" in df.columns:
        lines.append("## View Position x classe")
        lines.append(pd.crosstab(df["View Position"], df["target_label"]).to_markdown())
        lines.append("")
        lines.append("## View Position x split")
        lines.append(pd.crosstab(df["split"], df["View Position"]).to_markdown())
        lines.append("")
    if "Patient Age" in df.columns:
        lines.append("## Idade sanitizada por classe")
        lines.append(df.groupby("target_label")["Patient Age"].describe().to_markdown())
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--n-per-class", type=int, default=1200)
    parser.add_argument(
        "--positive-policy",
        choices=["any-cardiomegaly", "isolated-cardiomegaly"],
        default="any-cardiomegaly",
    )
    parser.add_argument("--seed", type=int, default=565535)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"Image Index", "Finding Labels", "Patient ID"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in NIH CSV: {sorted(missing)}")

    if "Patient Age" in df.columns:
        df["Patient Age"] = pd.to_numeric(df["Patient Age"], errors="coerce")
        df = df[df["Patient Age"].between(0, 120, inclusive="both")].copy()

    if args.positive_policy == "isolated-cardiomegaly":
        positives = df[df["Finding Labels"].eq("Cardiomegaly")].copy()
    else:
        positives = df[df["Finding Labels"].map(lambda x: has_label(x, "Cardiomegaly"))].copy()

    negatives = df[df["Finding Labels"].eq("No Finding")].copy()

    positives["target"] = 1
    positives["target_label"] = "Cardiomegaly"
    negatives["target"] = 0
    negatives["target_label"] = "No Finding"

    n = min(args.n_per_class, len(positives), len(negatives))
    if n < 50:
        raise SystemExit(f"Too few samples after filtering: n_per_class={n}")

    pos_sample = positives.sample(n=n, random_state=args.seed)
    neg_sample = negatives.sample(n=n, random_state=args.seed)
    subset = pd.concat([pos_sample, neg_sample], ignore_index=True)
    subset = subset.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    manifest = split_by_patient(subset, seed=args.seed)

    patient_sets = {
        split: set(part["Patient ID"].astype(str))
        for split, part in manifest.groupby("split")
    }
    overlaps = {
        "train_val": patient_sets.get("train", set()) & patient_sets.get("val", set()),
        "train_test": patient_sets.get("train", set()) & patient_sets.get("test", set()),
        "val_test": patient_sets.get("val", set()) & patient_sets.get("test", set()),
    }
    if any(overlaps.values()):
        raise SystemExit(f"Patient leakage detected: {overlaps}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUT_DIR / "data_manifest_plan.csv"
    manifest.to_csv(manifest_path, index=False)

    summary = summarize(manifest)
    summary += "\n## Prova anti-vazamento\n\n"
    summary += "- train_val overlap: 0\n"
    summary += "- train_test overlap: 0\n"
    summary += "- val_test overlap: 0\n"
    summary += f"\nPolitica de positivo: `{args.positive_policy}`\n"
    summary += f"Amostra por classe planejada: `{n}`\n"

    summary_path = RESULTS_DIR / "eda_nih_resumo.md"
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Manifest saved: {manifest_path}")
    print(f"Summary saved: {summary_path}")
    print("No patient leakage detected.")


if __name__ == "__main__":
    main()

