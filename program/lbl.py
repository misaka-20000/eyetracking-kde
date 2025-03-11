import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import numpy as np

# 假设数据
X = np.random.rand(100, 2)  # 随机生成100个二维数据点
y = np.random.randint(0, 4, 100)  # 随机生成0到3之间的级别数组（共4个级别）

# 使用t-SNE降维（如果数据是高维的，可以用t-SNE降维）
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 绘制t-SNE图，使用不同的颜色表示不同级别
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', edgecolor='k')

# 添加颜色条
plt.colorbar(scatter, label='Level')

# 标题和轴标签
plt.title('t-SNE with Level Coloring')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

plt.show()