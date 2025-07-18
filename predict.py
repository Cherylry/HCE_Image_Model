import os
import cv2
import timm
import torch
import numpy as np
from main import valid_transform
from collections import OrderedDict
from torch.nn import functional as F

img_folder = r"C:\Users\10756\Desktop\Aerial_407"     # 这里是图片地址，图片文件夹最好放在同一个文件夹（AutoEncoderCluster）
feature_folder = './Aerial12Feature/'   #下次跑代码改个数字就行
ckpt_path = './Checkpoints/aerialbest1.pth'  #这个不要碰

if not os.path.exists(feature_folder):
    os.mkdir(feature_folder)

model = timm.create_model('resnet50', pretrained=True, features_only=True)
old_state_dict = torch.load(ckpt_path, map_location='cpu')
new_state_dict = OrderedDict()
for k, v in old_state_dict.items():
    if k[0] == 'e':
        new_state_dict[k[8:]] = v
model.load_state_dict(new_state_dict)
model.eval()
model.cuda()


img_list = os.listdir(img_folder)

with torch.no_grad():
    for _, img_name in enumerate(img_list):
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = valid_transform(img)
        img = img.unsqueeze(0)
        img = img.cuda()
        feature = model(img)[-1]
        feature = F.adaptive_avg_pool2d(feature, 1)
        feature = feature[0].cpu().detach().numpy()
        np.save(feature_folder + img_name[:-4] + '.npy', feature)

