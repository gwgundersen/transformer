from transformer.inputs import Split
from transformer.model import Transformer
from transformer.train import compute_loss


def evaluate(model: Transformer, split: Split) -> float:
    """
    """
    model.eval()
    loss_sum = 0
    n_tokens = 0
    for batch in get_eval_batches(split.src, split.tgt, split.buckets):
        l, n = compute_loss(model, batch["src"], batch["tgt"])
        loss_sum += l
        n_tokens += n
    return loss_sum / n_tokens
        
    
