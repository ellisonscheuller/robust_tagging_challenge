import torch

class PUPPIDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            data: torch.Tensor, 
            labels: torch.Tensor, 
            device: torch.device
        ):
        self.data = data
        self.device = device
        self.labels = labels  
    
    def __len__(self):
        return self.data.shape[0]
    
    def make_padding_mask(self, x):
        return x[:, 0] == 0  

    def __getitem__(self, idx):
        to_return = (self.data[idx], self.make_padding_mask(self.data[idx]))
        to_return += (self.labels[idx],) if self.labels is not None else ()
        return to_return

class PFCandsDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            data: torch.Tensor, 
            labels: torch.Tensor, 
            device: torch.device, 
        ):
        self.data = data
        self.device = device
        self.labels = labels
    
    def __len__(self) -> int:
        return self.data.shape[0]
    
    def make_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., 0] == 0  

    def __getitem__(self, idx: int) -> tuple:
        to_return = (self.data[idx], self.make_padding_mask(self.data[idx]))
        to_return += (self.labels[idx],) if self.labels is not None else ()
        return to_return

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, in_dataset, aug_dataset):
        self.in_dataset = in_dataset
        self.aug_dataset = aug_dataset

    def __len__(self) -> int:
        return len(self.in_dataset)

    def __getitem__(self, idx: int) -> tuple:
        x_in, mask_in, label = self.in_dataset[idx]
        x_aug, mask_aug, _ = self.aug_dataset[idx]
        return x_in, mask_in, x_aug, mask_aug, label