from __future__ import annotations

import zipfile
from pathlib import Path

DATASET = "nih-chest-xrays/data"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dados" / "raw" / "nih_metadata"
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    "Data_Entry_2017.csv",
    "BBox_List_2017.csv",
    "BBoxlist2017.csv",
    "train_val_list.txt",
    "test_list.txt",
    "README_ChestXray.pdf",
]


def unzip_downloads() -> None:
    for z in OUT.glob("*.zip"):
        with zipfile.ZipFile(z) as archive:
            archive.extractall(OUT)
        z.unlink()


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    files = api.dataset_list_files(DATASET).files
    names = sorted({f.name for f in files})

    (OUT / "kaggle_files_list.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Dataset files listed: {len(names)}")
    print(f"List saved: {OUT / 'kaggle_files_list.txt'}")

    lower_to_name = {Path(name).name.lower(): name for name in names}
    downloaded = []
    missing = []

    for candidate in CANDIDATES:
        match = lower_to_name.get(candidate.lower())
        if not match:
            missing.append(candidate)
            continue
        print(f"Downloading metadata file: {match}")
        api.dataset_download_file(DATASET, file_name=match, path=str(OUT), force=False, quiet=False)
        downloaded.append(match)
        unzip_downloads()

    print("")
    print("Downloaded:")
    for name in downloaded:
        print(f"- {name}")
    if missing:
        print("")
        print("Not found with expected names:")
        for name in missing:
            print(f"- {name}")

    data_entry = next(OUT.glob("**/Data_Entry_2017.csv"), None)
    if not data_entry:
        raise SystemExit("Data_Entry_2017.csv not found. Check Kaggle terms/token or file names.")


if __name__ == "__main__":
    main()

