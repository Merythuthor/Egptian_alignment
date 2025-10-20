

from .simcse_loss import simcse_loss, simcse_batch_golden_loss
from .mlm_loss import mlm_loss
from .early_fusion_loss import early_fusion_loss




def load_loss(loss_type):
    if loss_type == "mlm":
        from .mlm_loss import mlm_loss
        return mlm_loss
    elif loss_type == "simcse":
        from .simcse_loss import simcse_loss
        return simcse_loss
    elif loss_type == "early_fusion":
        from .early_fusion_loss import early_fusion_loss
        return early_fusion_loss
    elif loss_type == "english_anchor":
        from .english_anchor_loss import EnglishAnchorLossWrapper
        return EnglishAnchorLossWrapper
    elif loss_type == "simcse_batchgolden":
        from .simcse_loss import simcse_loss_wrapper
        return simcse_loss_wrapper
    elif loss_type == "transformer_multitask":
        from .mlm_loss import mlm_loss
        return mlm_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

