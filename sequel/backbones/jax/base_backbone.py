import jax.numpy as jnp
from flax import linen as nn
from jax.lax import dynamic_update_slice


def other_fun():
    return -10e10


class BaseBackbone(nn.Module):
    """Inits the BaseBackbone class. This class defines the Jax base class for neural networks. All models
    should inherit from this class. Inherits from flax.nn.Module and the BaseCallback class that endows callbacks
    for each stage of training, e.g., before and after trainining/validation steps/epochs/tasks etc.

    Attributes:
        multihead (bool, optional): Set to True if the backbone is multi-headed. Defaults to False.
        classes_per_task (Optional[int], optional): The number of classes per task. Defaults to None.
        masking_value (float, optional): The value that replaces the logits. Only used if multihead is set to True. Defaults to -10e10.

    Note:
        Currently, the BaseBackbone only considers tasks with equal number of classes.
    """

    masking_value = -10e10
    classes_per_task: int
    multihead: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray, task_ids: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        raise NotImplementedError

    def select_output_head(self, x, task_ids):
        assert self.multihead
        assert isinstance(x, jnp.ndarray)
        mask = jnp.ones_like(x)
        z = jnp.zeros((1, self.classes_per_task))
        for i, task_id in enumerate(task_ids):
            task_id = task_id - 1
            mask = dynamic_update_slice(mask, z, (i, task_id * self.classes_per_task))

        x = jnp.where(mask, other_fun(), x)

        return x


class BackboneWrapper(BaseBackbone):
    model: nn.Module
    masking_value = -10e10
    classes_per_task: int
    multihead: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray, task_ids: jnp.ndarray = None, training: bool = True) -> jnp.ndarray:
        x = self.model(x, training)
        if self.multihead and task_ids is not None:
            x = self.select_output_head(x, task_ids)
        return x
