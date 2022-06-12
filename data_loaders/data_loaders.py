from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

class CIFAR10DataLoader(DataLoader):
    def __init__(self, root, train=True, batch_size=8, shuffle=False, num_workers=0):
        #color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        trsfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomApply([scolor_jitter], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        
        if train:
            self.dataset = CIFAR10(root=root, train=train, download=True, transform=trsfm)
        else:
            self.dataset = CIFAR10(root=root, train=train, download=True, transform=test_trsfm)
        super().__init__(
            dataset=self.dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    