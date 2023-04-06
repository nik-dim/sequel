from typing import Tuple, Union

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class ContinualDataset(torch.utils.data.Dataset):
    def __init__(self, task_id: int, *args, **kwargs) -> None:
        """Inits the ContinualDataset class.

        Args:
            task_id (int): The id of the current task.
        """
        super().__init__(*args, **kwargs)
        self.task_id = task_id

    def __getitem__(self, index: int) -> Tuple[Tensor, Union[Tensor, int], int]:
        x, y = super().__getitem__(index=index)
        return x, y, self.task_id


class ContinualConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class ContinualSubsetDataset(torch.utils.data.Subset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        targets = self.dataset.targets
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)
        self.targets = targets[self.indices]


class ContinualVisionDataset(Dataset):
    def __init__(self, task_id: int, data, targets, transform=None, target_transform=None) -> None:
        self.task_id = task_id
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[Tensor, Union[Tensor, int], int]:
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.task_id

    def __len__(self) -> int:
        return len(self.targets)


class SplitDataset(Dataset):
    def __init__(self, task_id, classes_per_split, dataset, fixed_class_order=None):
        self.task_id = task_id
        self.classes_per_split = classes_per_split
        self.fixed_class_order = torch.tensor(fixed_class_order) if fixed_class_order is not None else None

        self.__build_split(dataset, task_id)

    def __build_split(self, dataset: Dataset, task_id: int):
        low = (task_id - 1) * self.classes_per_split
        high = task_id * self.classes_per_split
        targets = dataset.targets
        if isinstance(targets, list):
            targets = torch.tensor(targets)

        if self.fixed_class_order is not None:
            targets = targets.apply_(lambda x: self.fixed_class_order.argsort()[x])
        self.indices = torch.where(torch.logical_and(targets >= low, targets < high))[0]
        self.dataset = ContinualSubsetDataset(dataset, self.indices)
        self.targets = self.dataset.targets

    def __getitem__(self, index: int) -> Tuple[Tensor, Union[Tensor, int], int]:
        img, target = self.dataset[index]
        return img, target, self.task_id

    def __len__(self) -> int:
        return len(self.dataset)
