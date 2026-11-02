from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

from transformer.paths import DATASET_DIR

# --------------------------------------------------------------------------------------------------

def load_iwslt2017_en_de() -> DatasetDict:
    """
    Download/cache train/valid/test sets for German-to-English using HF's built-in Parquet loader.
    """
    base = "hf://datasets/IWSLT/iwslt2017@refs/convert/parquet/iwslt2017-en-de"
    ds = load_dataset(
        "parquet",
        data_files={
            split: f"{base}/{split}/*.parquet"
            for split in ("train", "validation", "test")
        },
        cache_dir=str(Path(__file__).parent / ".cache" / "datasets"),
    )
    return ds.map(
        lambda row: {
            "src": row["translation"]["de"],
            "tgt": row["translation"]["en"]
        },
        remove_columns=["translation"]
    )


def generate_add_dataset(seed: int = 0) -> DatasetDict:
    """
    Generate all combinations of three-digit addition.
    """
    return load_from_disk(DATASET_DIR / "add-3d")



DATASETS = {
    "iwslt2017-de-en": load_iwslt2017_en_de,
    "add-3d": generate_add_dataset,
}
