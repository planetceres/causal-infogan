import torch.utils.data as data
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.utils import save_image
from scipy.ndimage.morphology import grey_dilation
import torch

def filter_background(x):
    x[:, (x < 0.3).any(dim=0)] = 0.0
    return x

def dilate(x):
    x = x.squeeze(0).numpy()
    x = grey_dilation(x, size=3)
    x = x[None, :, :]
    return torch.from_numpy(x)


transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    # transforms.RandomRotation(360),
    transforms.ToTensor(),
    filter_background,
    lambda x: x.mean(dim=0)[None, :, :],
    dilate,
    transforms.Normalize((0.5,), (0.5,)),
])
dataset = ImageFolder('data/rope/full_data', transform=transform)
loader = data.DataLoader(dataset, batch_size=64, shuffle=True,
                         pin_memory=True, num_workers=2)

batch = next(iter(loader))[0]
print(batch.shape)
save_image(batch * 0.5 + 0.5, 'example_dilate.png', nrow=8)
