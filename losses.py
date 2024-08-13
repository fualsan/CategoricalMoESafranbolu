import torch

def categorical_loss(_category_logits, _target_type_ids):
    categorical_dist = torch.distributions.Categorical(logits=_category_logits)
    log_probs = categorical_dist.log_prob(_target_type_ids)
    return -(log_probs.mean())