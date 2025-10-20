import torch.nn.functional as F

def mlm_loss(predictions, labels):

    return F.cross_entropy(
        predictions.view(-1, predictions.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
