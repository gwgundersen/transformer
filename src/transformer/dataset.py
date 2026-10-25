from pathlib import Path

from datasets import load_dataset

# --------------------------------------------------------------------------------------------------

def load_iwslt2017_en_de():
    """
    Download/cache train/valid/test sets for German-to-English using HF's built-in Parquet loader.
    """
    base = "hf://datasets/IWSLT/iwslt2017@refs/convert/parquet/iwslt2017-en-de"
    return load_dataset(
        "parquet",
        data_files={
            split: f"{base}/{split}/*.parquet"
            for split in ("train", "validation", "test")
        },
        cache_dir=str(Path(__file__).parent / ".cache" / "datasets"),
    )


DATASETS = {
    "iwslt2017-de-en": load_iwslt2017_en_de,
}
