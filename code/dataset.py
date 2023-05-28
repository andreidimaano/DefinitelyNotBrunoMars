from torch.utils.data import Dataset
import h5py

class MusicDataset(Dataset):
    def __init__(self, file, transform=None):
        self.file = file
        self.transform = transform
        
        with h5py.File(file, 'r') as hf:
            self.data = hf['data'][:]
            self.labels = hf['labels'][:]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        if self.transform is not None:
            x = self.transform(x)
        
        return x,y