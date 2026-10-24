import json
import numpy as np

from flax import nnx
import jax
import jax.numpy as jnp
import optax
import time

from transformer.checkpoints import save_checkpoint
from transformer.config import IWSLTConfig
from transformer.dataset import load_iwslt_en_de
from transformer.decode import greedy_decode
from transformer.model import Transformer
from transformer.tokenizer import (
    build_vocab, prepare, make_src_mask, make_tgt_mask, BOS_ID, EOS_ID, PAD_ID
)
from transformer.train import batches, train_step
from transformer.vocab import load_vocab


# --------------------------------------------------------------------------------------------------

BATCH_SIZE = 128
SAVE_EVERY = 1


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")

    data = load_iwslt_en_de()
    train_data = data["train"]

    vocab_de = load_vocab("de")
    vocab_en = load_vocab("en")
    en_id_to_word = {i: w for w, i in vocab_en.items()}

    config = IWSLTConfig()

    model = Transformer(
        src_vocab=vocab_de,
        tgt_vocab=vocab_en,
        dropout_rate=0.0
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adam(1e-4),
        wrt=nnx.Param,
    )

    train_data = data["train"].map(
        prepare,
        fn_kwargs={"vocab": {"de": vocab_de, "en": vocab_en}, "max_length": config.max_length},
        remove_columns=data["train"].column_names,
        load_from_cache_file=True,
    )

    # HuggingFace's API will return a list. Convert to a block of numpy arrays.
    train_data = train_data.with_format("numpy")
    src_data = np.asarray(train_data["src"])
    tgt_data = np.asarray(train_data["tgt"])

    def print_sentence(x):
        print(" ".join([en_id_to_word[int(i)] for i in x if i != PAD_ID]))

    rng = np.random.default_rng(seed=0)
    step = 0
    losses = []
    for epoch in range(1000):

        for batch in batches(src_data, tgt_data, batch_size=BATCH_SIZE, rng=rng):
            loss = train_step(model, optimizer, batch)
            losses.append(loss)
            step +=1 
        
        if epoch % SAVE_EVERY == 0:
            print(f"{epoch}\t{float(losses[-1]):.4f}")
            save_checkpoint(
                "iwslt",
                model,
                optimizer,
                losses,
                step
            )

            tgt_input = batch["tgt"][:, :-1]
            src_mask = make_src_mask(batch["src"])
            tgt_mask = make_tgt_mask(tgt_input)

            y_true = tgt_input[0]
            y_hat = greedy_decode(
                model, 
                src=batch["src"][:1],
                src_mask=src_mask[:1],
                max_length=config.max_length,
            )
            print_sentence(y_true)
            print_sentence(y_hat)
