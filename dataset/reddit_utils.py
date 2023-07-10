from torch.utils.data import Dataset


class Reddit_Dataset(Dataset):
    def __init__(self, data):
        self.seq_len = 50
        self.text = data

    def __len__(self):
        return len(self.text) // 50

    def __getitem__(self, idx):
        i = idx * self.seq_len
        src = self.text[i : i + self.seq_len]
        target = self.text[i + 1 : i + self.seq_len + 1]
        return src, target
