import torch
import torch.nn as nn

check_batchnorm_condition = lambda k: "running_mean" in k or "running_var" in k or "num_batches_tracked" in k


@torch.no_grad()
def get_weights(model: nn.Module, flattened: bool = True):
    if flattened:
        return torch.cat([p.view(-1) for p in model.parameters()])
    else:
        state_dict = model.state_dict(keep_vars=True)
        return {k: v for k, v in state_dict.items() if not check_batchnorm_condition(k)}


@torch.no_grad()
def get_grads(model: nn.Module):
    return torch.cat(
        [p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1) for p in model.parameters()]
    )


@torch.no_grad()
def set_weights(model: nn.Module, weights: torch.Tensor):
    state_dict = model.state_dict(keep_vars=True)
    # The index keeps track of location of current weights that is being un-flattened.
    index = 0
    with torch.no_grad():
        v: torch.Tensor
        for k, v in state_dict.items():
            if check_batchnorm_condition(k):
                continue
            param_count = v.numel()
            param_shape = v.shape
            state_dict[k] = nn.Parameter(weights[index : index + param_count].reshape(param_shape))
            index += param_count
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def set_grads(model: nn.Module, grads: torch.Tensor):
    state_dict = model.state_dict(keep_vars=True)
    index = 0
    for k, v in state_dict.items():
        # ignore batchnorm params
        if check_batchnorm_condition(k):
            continue
        param_count = v.numel()
        param_shape = v.shape
        state_dict[k].grad = grads[index : index + param_count].view(param_shape).clone()
        index += param_count
    model.load_state_dict(state_dict)
    return model
