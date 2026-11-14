import numpy as np

from datasets import Dataset, DatasetDict, concatenate_datasets

from transformer.arith import count_carry_cascades
from transformer.paths import DATASET_DIR

# --------------------------------------------------------------------------------------------------

N_DIGITS = 3


def generate_dataset(*, zpad: bool, oversample: bool, seed: int) -> DatasetDict:
    """
    Generate max 3-digit addition dataset.
    """
    dct = {"src": [], "tgt": [], "cascades": []}
    for x in range(10**N_DIGITS):
        for y in range(10**N_DIGITS):
            # Generate both versions for ablations.
            if zpad:
                # zpad everything so that the sequences are fixed length. This makes the problem
                # easier for the transformer and results easier to intrepret. So each string is 14
                # tokens: [BOS,X,X,X,+,Y,Y,Y,=,A,A,A,A,EOS]
                xz = str(x).zfill(N_DIGITS)
                yz = str(y).zfill(N_DIGITS)
                az = str(x + y).zfill(N_DIGITS+1)
            else:
                xz = x
                yz = y
                az = str(x + y)

            dct["src"].append(f"{xz}+{yz}=")
            dct["tgt"].append(f"{az}")
            dct["cascades"].append(count_carry_cascades(x, y))
    ds = Dataset.from_dict(dct)

    # 80/10/10 train/cv/test split
    split = ds.train_test_split(test_size=0.2, seed=seed, shuffle=True)
    holdout = split["test"].train_test_split(test_size=0.5, seed=seed, shuffle=True)

    # Oversample hard problems in the training data only.
    train = split["train"]
    if oversample:
        _log_subset_percentages(train)
        train = _oversample_cascades(train, seed=seed)
        _log_subset_percentages(train)

    return DatasetDict(
        train=train,
        validation=holdout["train"],
        test=holdout["test"],
    )


def _log_subset_percentages(train: Dataset) -> None:
    """
    Print statistics about cascade-based subsets.
    """
    for n in (0, 1, 2):
        frac = (np.asarray(train["cascades"]) == n).sum().item() / len(train)
        print(f"{100 * frac:.2f}% of examples have {n} cascades")


def _oversample_cascades(train: Dataset, seed: int) -> Dataset:
    """
    Oversample problems with 1 or 2 cascades. From the training data:

    91.01 % have zero cascades
     8.55 % have one cascade
     0.45 % have two cascades

    So 2x for 1 cascade and 20x for 2 cascades to show the model more hard cases.
    """
    # FIXME: We could do this automatically, but currently this is hard-coded based on 3-digit
    # statistics.
    out = []
    cascades_to_mult = {0: 1, 1: 2, 2: 20}
    for nc, mult in cascades_to_mult.items():
        subset = train.filter(lambda row: row["cascades"] == nc)
        out += [subset] * mult
    return concatenate_datasets(out).shuffle(seed=seed)
    

# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    for zpad in (False, True):
        for oversample in (False, True):
            print("=" * 80)
            print(f"{zpad=}, {oversample=}")
            dataset = generate_dataset(zpad=zpad, oversample=oversample, seed=0)
            fname = f"add3d{'-zpad' if zpad else ''}{'-oversample' if oversample else ''}"
            print(f"Saving dataset to {fname}")
            dataset.save_to_disk(DATASET_DIR / fname)
