import os
import cv2
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.img_list = os.listdir(path)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.path, img_name)
        img = cv2.imread(img_path)
        if self.transform:
            img = self.transform(img)
        return img


if __name__ == '__main__':
    data_path = '../RoofColor/'
    img_list = os.listdir(data_path)
    for _, img in enumerate(img_list):
        os.rename(data_path + img, data_path + str(_) + '.png')
