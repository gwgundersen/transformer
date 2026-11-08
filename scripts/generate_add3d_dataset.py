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
            # zpad everything so that the sequences are fixed length. This makes the problem easier
            # for the transformer and results easier to intrepret. So each string is 14 tokens:
            # [BOS,X,X,X,+,Y,Y,Y,=,A,A,A,A,EOS]
            xz = str(x).zfill(3)
            yz = str(y).zfill(3)
            az = str(x + y).zfill(4)
            dct["src"].append(f"{xz}+{yz}=")
            dct["tgt"].append(f"{az}")
    ds = Dataset.from_dict(dct)

    # 80/10/10 train/cv/test split
    split = ds.train_test_split(test_size=0.2, seed=SEED, shuffle=True)
    holdout = split["test"].train_test_split(test_size=0.5, seed=SEED, shuffle=True)
    dd = DatasetDict(
        train=split["train"],
        validation=holdout["train"],
        test=holdout["test"],
    )
    dd.save_to_disk(DATASET_DIR / "add-3d")
