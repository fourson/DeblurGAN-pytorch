from torchvision import transforms
from PIL import Image

from . import dataset
from base.base_data_loader import BaseDataLoader


class GoProDataLoader(BaseDataLoader):
    """
    GoPro data loader
    """

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers):
        transform = transforms.Compose([
            transforms.Resize([360, 640], Image.BICUBIC),  # downscale by a factor of two (720*1280 -> 360*640)
            transforms.ToTensor(),  # convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
        ])
        self.dataset = dataset.GoProDataset(data_dir, transform=transform, height=360, width=640, fine_size=256)

        super(GoProDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
