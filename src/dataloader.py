from PIL import Image
import pathlib
from typing import List
from torchvision import transforms
from torch.utils.data import Dataset

ACCEPTED_IMAGE_EXTS = ['.jpg', '.png']


def get_transformation():
    '''
    torchvision transforms that take a list of `n` PIL image(s) and output
    tensor of shape `N x W X H X 3` \n
    Then following with normalize 
    and further transform will be applied.
    Return
    ------
    return torch image transforms
    '''
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation = transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    return transform

class MyDataLoader(Dataset):
    def __init__(self, image_root):
        self.image_root = pathlib.Path(image_root)
        self.image_list = self._get_image_list(self.image_root)
        self.transform = get_transformation()
    # Get the list of images in 1 folder

    def _get_image_list(self, root: pathlib.Path) -> List[pathlib.Path]:
        image_list = []
        for entry in root.rglob('*'):  # Use rglob to recursively search
            if entry.is_file() and entry.suffix.lower() in ACCEPTED_IMAGE_EXTS:
                image_list.append(entry)
        return sorted(image_list, key=lambda x: x.name)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        _img = self.image_list[index]
        _img = Image.open(_img)
        _img = _img.convert("RGB")
        return self.transform(_img), str(self.image_list[index])