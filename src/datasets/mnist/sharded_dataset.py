from torch.utils.data import Dataset
from torchvision import transforms
import typing as T


class ShardedMNIST(Dataset):
    """Dataset for creating specific distribution shards of MNIST."""

    def __init__(
        self,
        dataset: Dataset,
        indices: T.List[int],
        transform: T.Optional[transforms.Compose] = None,
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label
