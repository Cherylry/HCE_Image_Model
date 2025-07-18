import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from rrr import CatBoostClassifier, Pool
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency  # 添加这行导入
from sklearn.model_selection import StratifiedKFold
import numpy as np

# 读取数据
data_path = r'C:\Users\10756\Desktop\2\machine\streetscape\加入鸟瞰街景.csv'  # 请确认路径和文件名
data = pd.read_csv(data_path, encoding='GBK')

# 显示数据前几行，确保数据读取正确
print(data.head())

# 自变量和因变量
continuous_columns = ['Number of residents', 'Number of elderly people', 'Number of children']
categorical_columns = [
    'Sex ratio', 'Residential Type', 'Year of construction', 'Building storey',
    'Apartment Layout', 'Total Floor Area', 'Heating method', 'Cooling method', 'Cooking method',
    'east', 'west', 'south', 'north', 'Aerial view'  # 添加新的分类变量
]

# 将'7'作为有效类别处理，不做任何替换，直接作为分类变量
# 对分类变量中的 7 替换为 NaN 并将 NaN 转换为字符串 'NaN'
for col in ['east', 'west', 'south', 'north', 'Aerial view']:
    data[col] = data[col].replace(7, np.nan).fillna('NaN').astype(str)

# 对连续变量进行标准化
scaler = StandardScaler()
data[continuous_columns] = scaler.fit_transform(data[continuous_columns])

# 对因变量进行编码（分类变量需要整数编码）
label_encoder = LabelEncoder()
data['Annual average carbon emissions'] = label_encoder.fit_transform(data['Annual average carbon emissions'])

# 设置自变量（X）和因变量（y）
X = data[continuous_columns + categorical_columns]
y = data['Annual average carbon emissions']

# 检查类别分布
print("Target variable (y) distribution:")
print(y.value_counts())

# 过滤掉样本数量少于 2 的类别
valid_classes = y.value_counts()[y.value_counts() >= 2].index
X = X[y.isin(valid_classes)]
y = y[y.isin(valid_classes)]

# 切分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 将数据转换为 CatBoost 数据格式
train_pool = Pool(X_train, y_train, cat_features=categorical_columns)
test_pool = Pool(X_test, y_test, cat_features=categorical_columns)

# 定义 CatBoost 分类模型，改进参数
model = CatBoostClassifier(
    iterations=1000,  # 减少迭代次数，但使用更好的参数
    learning_rate=0.005,  # 增加学习率
    depth=10,  # 减少深度以防止过拟合
    loss_function='MultiClass',
    random_seed=42,
    verbose=100,
    l2_leaf_reg=1,  # L2 正则化
    min_data_in_leaf=1,  # 每个叶子节点的最小数据量，以防止过拟合
    auto_class_weights='Balanced',  # 继续使用类别平衡
    bagging_temperature=0.9,  # 增强随机化
    border_count=128,  # 增加边界数
    max_ctr_complexity=4,  # 最大化 CTR 特征复杂度
)

# 添加交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in skf.split(X, y):
    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(X_fold_train, y_fold_train, cat_features=categorical_columns)
    val_pool = Pool(X_fold_val, y_fold_val, cat_features=categorical_columns)

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)
    pred = model.predict(X_fold_val)
    score = accuracy_score(y_fold_val, pred)
    cv_scores.append(score)

print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

# 预测
y_pred = model.predict(X_test)
y_pred_classes = y_pred.flatten()  # 将预测结果转换为一维数组

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Weighted F1-Score: {f1}')

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# 平滑处理混淆矩阵，避免零频数导致计算错误
smoothed_conf_matrix = conf_matrix + 1  # 添加 1 作为平滑值

# 卡方检验计算 P 值
chi2_stat, p_value, dof, expected = chi2_contingency(smoothed_conf_matrix)
print(f'Chi-squared Test Statistic: {chi2_stat}')
print(f'P-value for Chi-squared Test: {p_value}')
print(f'Degrees of Freedom: {dof}')

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 特征重要性
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.get_feature_importance(train_pool)
}).sort_values(by='Importance', ascending=False)

feature_importance['Importance'] /= feature_importance['Importance'].sum()

# 打印特征重要性
print(feature_importance)

# 保存特征重要性为 CSV 文件
feature_importance.to_csv('feature_importance_catboost_classification.csv', index=False)

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()
# 保存模型文件
model_path = r"C:\Users\10756\Desktop\catboost_model_with_streetscape.cbm"
model.save_model(model_path)
print("模型已保存。")