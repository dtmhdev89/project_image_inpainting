import os
import torch
from torchvision import datasets, transforms


class DatasetLoader:
    """Dataset Loader"""

    def __init__(self, batch_size: int = 64) -> None:
        self._batch_size = batch_size
        os.makedirs(os.path.join("dataset"), exist_ok=True)
        self._save_output = os.path.join("dataset")

    def build_trainset(
        self,
        train_transforms: transforms.Compose | None = None
    ):
        """Prepare trainset"""

        if not train_transforms:
            train_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])

        trainset = datasets.MNIST(
            root=self._save_output,
            train=True,
            download=True,
            transform=train_transforms
        )

        return trainset

    def build_train_loader(self, trainset):
        """Build train loader"""

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True
        )

        return train_loader
