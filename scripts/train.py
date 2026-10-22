import json
import numpy as np

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from transformer.checkpoints import save_checkpoint
from transformer.config import IWSLTConfig
from transformer.dataset import load_iwslt_en_de
from transformer.decode import greedy_decode
from transformer.model import Transformer
from transformer.tokenizer import build_vocab, prepare, make_tgt_mask, BOS_ID, EOS_ID, PAD_ID
from transformer.train import batches, train_step
from transformer.vocab import load_vocab


# --------------------------------------------------------------------------------------------------

SAVE_EVERY = 100
# Small but try to prevent memory fragmentation on Colab GPU
BATCH_SIZE = 16


if __name__ == "__main__":
    print(f"Jax devices: {jax.devices()}")

    data = load_iwslt_en_de()
    train_data = data["train"].to_list()

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

    # FIXME: What is the max length?
    train_data = data["train"].map(
        prepare,
        fn_kwargs={"vocab": {"de": vocab_de, "en": vocab_en}, "max_length": config.max_length},
        remove_columns=data["train"].column_names,
        load_from_cache_file=True,
    )

    # FIXME: Explain.
    train_data = train_data.with_format("jax")

    def print_sentence(x):
        print(" ".join([en_id_to_word[int(i)] for i in x if i != PAD_ID]))

    step = 0
    losses = []
    for epoch in range(1000):

        for batch in batches(train_data, batch_size=BATCH_SIZE):
            loss = train_step(model, optimizer, batch)
            losses.append(loss)
            if step % SAVE_EVERY == 0:
                print(f"Saving checkpoint at {step=}")
                save_checkpoint(
                    f"iwslt_{step}",
                    model,
                    optimizer,
                    step=step
                )
            step +=1 
        
        print(f"{epoch}\t{loss}")

        y_true = batch["tgt_input"][0]
        y_hat = greedy_decode(
            model, 
            src=batch["src"][:1],
            src_mask=batch["src_mask"][:1],
        )
        print_sentence(y_true)
        print_sentence(y_hat)
