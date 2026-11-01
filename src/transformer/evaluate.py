from transformer.inputs import Split
from transformer.model import Model
from transformer.train import compute_loss

# --------------------------------------------------------------------------------------------------

def evaluate(model: Model, split: Split) -> float:
    """
    Evaluates a model given a particular dataset split.
    """
    model.eval()
    loss_sum = 0
    n_tokens = 0
    for batch in get_eval_batches(split.src, split.tgt, split.buckets):
        l, n = compute_loss(model, batch)
        loss_sum += l
        n_tokens += n
    return loss_sum / n_tokens
