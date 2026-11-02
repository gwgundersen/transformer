from datasets import Dataset, DatasetDict

from transformer.paths import DATASET_DIR

# --------------------------------------------------------------------------------------------------

SEED = 0

if __name__ == "__main__":
    """
    Generate max 3-digit addition dataset.
    """
    dct = {"src": [], "tgt": []}
    for x in range(1000):
        for y in range(1000):
            dct["src"].append(f"{x}+{y}=")
            dct["tgt"].append(f"{x+y}")
    ds = Dataset.from_dict(dct)

    # 80/10/10 train/cv/test split
    split = ds.train_test_split(test_size=0.2, seed=SEED)
    holdout = split["test"].train_test_split(test_size=0.5, seed=SEED)
    dd = DatasetDict(
        train=split["train"],
        validation=holdout["train"],
        test=holdout["test"],
    )
    dd.save_to_disk(DATASET_DIR / "add-3d")
