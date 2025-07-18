import torch
import cv2
from torch import nn
from dataset import MyDataset
from model import MyModel
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(90),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 512)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = MyDataset(r"E:\All picture\N-elevation", transform=train_transform)  #这里路径选图片地址
valid_dataset = MyDataset(r"E:\All picture\N-elevation", transform=valid_transform)  #这里路径选图片地址
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=70, shuffle=False)



if __name__ == '__main__':
    cuda = 0
    model = MyModel().cuda(cuda)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.load_state_dict(torch.load('./Checkpoints/windowbest.pth', map_location=f'cuda:{cuda}'))
    best_loss = 100
    for epoch in range(8000):
        train_loss = 0
        valid_loss = 0
        model.train()
        for data in train_loader:
            data = data.cuda(cuda)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'Epoch {epoch} training loss: {train_loss}')

        model.eval()
        with torch.no_grad():
            for data in valid_loader:
                data = data.cuda(cuda)
                output = model(data)
                loss = criterion(output, data)
                save_img = output[0].permute(1, 2, 0).cpu().detach().numpy()
                # denormalize
                save_img = save_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                cv2.imwrite('example.png', 255*save_img)
                valid_loss += loss.item()
            print(f'Epoch {epoch} validation loss: {valid_loss}')
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'Checkpoints/aerialbest1.pth')