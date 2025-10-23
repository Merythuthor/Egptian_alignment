


from .mlm_loss import mlm_loss




def load_loss(loss_type):
    if loss_type == "mlm":
        from .mlm_loss import mlm_loss
        return mlm_loss

    elif loss_type == "transformer_multitask":
        from .mlm_loss import mlm_loss
        return mlm_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

