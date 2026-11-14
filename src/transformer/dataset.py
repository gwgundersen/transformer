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


def load_tiny_stories(seed: int = 0) -> DatasetDict:
    """
    Download/cache train/valid/test sets for TinyStories using HF's built-in Parquet loader.

    TinyStories only ships train and validation splits, so split validation 50/50 into
    validation/test.
    """
    base = "hf://datasets/roneneldan/TinyStories@refs/convert/parquet/default"
    ds = load_dataset(
        "parquet",
        data_files={
            split: f"{base}/{split}/*.parquet"
            for split in ("train", "validation")
        },
        cache_dir=str(Path(__file__).parent / ".cache" / "datasets"),
    )
    holdout = ds["validation"].train_test_split(test_size=0.5, seed=seed, shuffle=True)

    # FIXME: Each row is a single "text" column. The rest of the pipeline expects src/tgt pairs, so
    # decide how to map plain LM data onto that (e.g. empty src, text as tgt).
    return DatasetDict(
        train=ds["train"],
        validation=holdout["train"],
        test=holdout["test"],
    )


def get_loader(name: str) -> DatasetDict:
    def func():
        return load_from_disk(DATASET_DIR / name)
    return func


DATASETS = {
    "add3d": get_loader("add3d"),
    "add3d-zpad": get_loader("add3d-zpad"),
    "add3d-zpad-oversample": get_loader("add3d-zpad-oversample"),
    "iwslt2017-de-en": load_iwslt2017_en_de,
    "tiny-stories": load_tiny_stories,
}
