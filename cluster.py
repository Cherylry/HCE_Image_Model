import os
import cv2
import shutil
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['figure.dpi'] = 200
os.environ["OMP_NUM_THREADS"] = "4"
# current_list = os.listdir('./')
# current_list = [i for i in current_list if 'Feature' in i]
# 定义聚类的图来源
# img_path = "../Data/Window4/"
img_path = r"C:\Users\10756\Desktop\Aerial_407"  #这里图片地址
output_path = './Aerial12ClusterResults/'    #这里改数字即可
# 定义特征的来源
feature_path = "./Aerial12Feature/"    #这里改数字即可


data_list = os.listdir(feature_path)
# 获取图像名字列表
data_list = os.listdir(feature_path)
for _, f_name in enumerate(data_list):
    # 读取图像
    f_path = os.path.join(feature_path, f_name)
    f = np.load(f_path)
    # 减小特征维度
    # 预测图转化为单通道

    feature = f.reshape(1, -1)
    if _ == 0:
        features = feature
    else:
        features = np.concatenate((features, feature), axis=0)

# 创建文件夹
if not os.path.exists(output_path):
    os.mkdir(output_path)

# KMeans定义多次聚类类别数
square_sum = [0] * 3
for n in range(3, 8):
    # 定义KMeans聚类方法 并进行聚类
    Methods = KMeans(n_clusters=n, random_state=0, n_init='auto').fit(features)

    # 定义类别输出路径
    class_path = os.path.join(output_path, f'n_{n}')
    if not os.path.exists(class_path):
        os.mkdir(class_path)

    # 创建类别文件夹
    for i in range(n):
        if not os.path.exists(os.path.join(class_path, f'c_{i}')):
            os.mkdir(os.path.join(class_path, f'c_{i}'))

    # 将聚类结果复制到对应的文件夹
    for __, label in enumerate(Methods.labels_):
        file_name_base = data_list[__][:-4]  # 去掉 .npy
        source_path = os.path.join(img_path, file_name_base + '.jpg')
        target_path = os.path.join(class_path, f'c_{label}', file_name_base + '.jpg')

        if not os.path.exists(source_path):
            print(f"⚠️ Source not found: {source_path}")
            continue

        try:
            shutil.copy(source_path, target_path)
            print(f"✅ Copied to: {target_path}")
        except Exception as e:
            print(f"❌ Failed to copy: {source_path} ➜ {target_path}, reason: {e}")

    # Features 进行 TSNE 降维
    tsne = TSNE(n_components=2, random_state=0)
    tsne_features = tsne.fit_transform(features)

    # 将降维后的标签直接可视化
    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=Methods.labels_)
    plt.savefig('./pics/' + feature_path[2:-1] + f'_{n}.jpg')
    plt.show()

    if n ==6:
        # 保存 tsne 降维后的二维坐标及类别编号
        df = {
            "id": [data_list[i][:-4] for i in range(len(data_list))],
            "x": tsne_features[:, 0],
            "y": tsne_features[:, 1],
            "label": Methods.labels_
        }
        df = pd.DataFrame(df)
        df.to_csv(os.path.join(output_path, "cluster6_tsne_result.csv"), index=False, encoding='utf-8-sig')
        print("📁 聚类结果 CSV 已保存为 cluster6_tsne_result.csv")
