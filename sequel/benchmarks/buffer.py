import random
from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import Tensor

PyJaxTensor = Union[torch.Tensor, np.ndarray]


class BufferDataset(torch.utils.data.Dataset):
    def __init__(self, return_logits: bool = False) -> None:
        super().__init__()
        self._data = []
        self._labels = []
        self._task_ids = []
        self._logits = []

        self.return_logits = return_logits

    @property
    def data(self):
        if isinstance(self._data[0], torch.Tensor):
            return torch.stack(self._data)
        elif isinstance(self._data[0], np.ndarray):
            return np.stack(self._data)
        else:
            raise TypeError

    @property
    def labels(self):
        if isinstance(self._labels[0], torch.Tensor):
            return torch.stack(self._labels)
        elif isinstance(self._labels[0], np.ndarray):
            return np.stack(self._labels)
        else:
            raise TypeError

    @property
    def task_ids(self):
        if isinstance(self._task_ids[0], torch.Tensor):
            return torch.stack(self._task_ids)
        elif isinstance(self._task_ids[0], np.ndarray):
            return np.stack(self._task_ids)
        else:
            raise TypeError

    @property
    def logits(self):
        if isinstance(self._logits[0], torch.Tensor):
            return torch.stack(self._logits)
        elif isinstance(self._logits[0], np.ndarray):
            return np.stack(self._logits)
        elif isinstance(self._logits[0], jax.xla.DeviceArray):
            return np.stack(self._logits)
            return jnp.stack(self._logits)
        else:
            raise TypeError

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> Any:
        x = self.data[index]
        y = self.labels[index]
        t = self.task_ids[index]
        if self.return_logits:
            assert self.return_logits
            logits = self.logits[index]
            return x, y, t, logits
        else:
            return x, y, t

    def update(self, x: Tensor, y: Union[int, Tensor], t: Union[int, Tensor], logits: Tensor = None):

        if isinstance(x, torch.Tensor):
            x = x.detach().squeeze().unsqueeze(0)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y)
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            if logits is not None and self.return_logits:
                logits = logits.detach()

        elif isinstance(x, np.ndarray):
            # x = x.detach().squeeze().unsqueeze(0)
            # logits = logits.detach()
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            if not isinstance(t, np.ndarray):
                t = np.array(t)

        self._data.append(x)
        self._labels.append(y)
        self._task_ids.append(t)
        if self.return_logits:
            self._logits.append(logits)

    def replace(self, index, x: Tensor, y: Union[int, Tensor], t: Union[int, Tensor], logits: Tensor = None):
        # if not isinstance(y, torch.Tensor):
        #     y = torch.tensor(y)
        # if not isinstance(t, torch.Tensor):
        #     t = torch.tensor(t)

        if isinstance(x, torch.Tensor):
            x = x.detach().squeeze().unsqueeze(0)
            if logits is not None and self.return_logits:
                logits = logits.detach()

        self._data[index] = x
        self._labels[index] = y
        self._task_ids[index] = t
        if self.return_logits:
            self._logits[index] = logits


class Buffer:
    def __init__(self, memory_size: int, return_logits: bool = False) -> None:
        self.memory_size = memory_size
        self.dataset = BufferDataset(return_logits)
        self.num_seen_samples = 0

        self.return_logits = return_logits

    # def prepare_data(self, )

    def add_data(self, x: Tensor, y: Tensor, t: Tensor, logits: Optional[Tensor] = None) -> None:
        if logits is None:
            logits = [None for _ in range(len(x))]
        if (isinstance(y, Tensor) and y.numel() == 1) or (isinstance(y, np.ndarray) and y.size == 1):
            x, y, t, logits = [x], [y], [t], [logits]
        for xx, yy, tt, ll in zip(x, y, t, logits):
            self.num_seen_samples += 1
            if len(self.dataset) < self.memory_size:
                self.dataset.update(xx, yy, tt, ll)
            else:
                # reservoir sampling
                index = random.randint(0, self.num_seen_samples)
                if index < self.memory_size:
                    self.dataset.replace(index, xx, yy, tt, ll)

    def sample_from_buffer(self, batch_size: int) -> Tuple[Tensor]:
        batch_size = min(batch_size, len(self.dataset))
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        indices = indices[:batch_size]

        return self.dataset[indices]

    def __len__(self) -> int:
        return len(self.dataset)

    def augment_batch_with_memory(
        self, x: Tensor, y: Tensor, t: Tensor, batch_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if len(self.dataset) == 0:
            return x, y, t

        if batch_size is None:
            batch_size = len(x)

        mem_x, mem_y, mem_t = self.sample_from_buffer(batch_size)
        assert type(x) == type(mem_x)
        if isinstance(x, torch.Tensor):
            x = torch.cat([x.clone(), mem_x])
            y = torch.cat([y.clone(), mem_y]).long()
            t = torch.cat([t, mem_t]).long()
        elif isinstance(x, np.ndarray):
            x = np.concatenate([x, mem_x])
            y = np.concatenate([y, mem_y])
            t = np.concatenate([t, mem_t])
        return x, y, t
