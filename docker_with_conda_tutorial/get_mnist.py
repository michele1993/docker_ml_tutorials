import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_data(data_path: str = '/data', batch_s: int = 64) -> (DataLoader, DataLoader):
    """ 
    Method to load mnist dataset from Pytoch library and organised them in bacthes
    Args:
        batch_s: size of the batch
    """

    # Download data
    training_data = datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform=ToTensor()
    )

    test_data = datasets.MNIST(
            root=data_path,
            train=False,
            download=True,
            transform=ToTensor()
    )

        
    # Use DataLoader to get data organised in random batches
    train_dataloader = DataLoader(training_data,batch_size=batch_s,shuffle=True)#, num_workers=2)
    test_dataloader = DataLoader(test_data,batch_size=batch_s,shuffle=True)#, num_workers=2)

    return train_dataloader, test_dataloader

