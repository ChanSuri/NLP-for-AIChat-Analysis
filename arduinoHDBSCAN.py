import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
import numpy as np
from sklearn.model_selection import KFold

# 1. 数据预处理
def preprocess_text(text):
    # 去除标点符号、特殊字符并小写化
    text = re.sub(r'[^\w\s]', '', text.lower())
    # 去除停用词
    stop_words = set(stopwords.words("english"))
    words = [word for word in text.split() if  len(word) > 2  and word not in stop_words]
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

df = pd.read_csv('dataset/ProjectText.csv')
df["cleaned_text"] = df['one_liner'].apply(preprocess_text)


# 2. 特征提取 (使用 BERT)
model = SentenceTransformer('all-mpnet-base-v2')
text_embeddings = model.encode(df["cleaned_text"].tolist())
# model = SentenceTransformer('all-MiniLM-L6-v2')
# text_embeddings = model.encode(df["cleaned_text"].tolist())

###########################################################################

# # 3. 定义调参空间
# min_cluster_sizes = [5, 10, 15]  # 尝试不同的 min_cluster_size
# min_samples_values = [1, 5, 10]  # 尝试不同的 min_samples
# n_neighbors_values = [5, 10, 15]  # UMAP 的邻居数
# pca_components = [5, 10, 15]  # PCA 降维的维度数

# # 4. 定义交叉验证
# kf = KFold(n_splits=3, shuffle=True, random_state=42)  # 设置 3 折交叉验证

# # 5. 调整参数并评估 Silhouette Score
# best_score = -1
# best_params = {}

# # 进行交叉验证和参数调优
# for min_cluster_size in min_cluster_sizes:
#     for min_samples in min_samples_values:
#         for n_neighbors in n_neighbors_values:
#             for pca_component in pca_components:
#                 # 使用 UMAP 降维
#                 reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=pca_component, random_state=42)
#                 reduced_embeddings = reducer.fit_transform(text_embeddings)

#                 # 使用 HDBSCAN 聚类
#                 clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
                
#                 # 存储每一折的轮廓系数
#                 fold_silhouette_scores = []

#                 for train_idx, val_idx in kf.split(reduced_embeddings):
#                     # 在训练集上进行聚类
#                     train_embeddings = reduced_embeddings[train_idx]
#                     clusterer.fit(train_embeddings)

#                     # 预测聚类结果
#                     val_embeddings = reduced_embeddings[val_idx]
#                     val_labels = clusterer.fit_predict(val_embeddings)

#                     # 计算并存储 Silhouette Score
#                     if len(set(val_labels)) > 1:  # 确保有多个簇
#                         score = silhouette_score(val_embeddings, val_labels)
#                         fold_silhouette_scores.append(score)

#                 # 计算每种参数组合的平均 Silhouette Score
#                 if fold_silhouette_scores:
#                     avg_silhouette_score = np.mean(fold_silhouette_scores)
#                     print(f"min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
#                           f"n_neighbors={n_neighbors}, pca_components={pca_component} -> "
#                           f"Average Silhouette Score: {avg_silhouette_score}")

#                     # 如果当前组合的 Silhouette Score 更高，则更新最佳参数
#                     if avg_silhouette_score > best_score:
#                         best_score = avg_silhouette_score
#                         best_params = {
#                             "min_cluster_size": min_cluster_size,
#                             "min_samples": min_samples,
#                             "n_neighbors": n_neighbors,
#                             "pca_components": pca_component
#                         }

# # 6. 输出最佳参数
# print(f"\nBest Parameters: {best_params}")
# print(f"Best Silhouette Score: {best_score}")

############################################################

# 使用 UMAP 降维 (可选)
reducer = umap.UMAP(n_neighbors=5, n_components=20, random_state=42)
reduced_embeddings = reducer.fit_transform(text_embeddings)


# 3. 使用 HDBSCAN 聚类
# 使用 HDBSCAN 聚类，增大 min_cluster_size 和 min_samples
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10, metric='euclidean')
df['cluster'] = clusterer.fit_predict(reduced_embeddings)

# 输出聚类结果
print("Cluster counts (including noise):")
print(df['cluster'].value_counts())

# 计算聚类的 Silhouette Score（排除噪声点）
non_noise_points = df['cluster'] != -1  # 排除噪声点
if non_noise_points.any():
    silhouette_avg = silhouette_score(text_embeddings[non_noise_points], df['cluster'][non_noise_points])
    print(f"Silhouette Score (excluding noise): {silhouette_avg}")
else:
    print("No clusters were identified, only noise.")


# 4. 提取关键词 (使用 TF-IDF + LDA)
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words="english",ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(df["cleaned_text"])

lda = LatentDirichletAllocation(n_components=len(set(df['cluster'])) - (1 if -1 in df['cluster'].values else 0), random_state=0)
lda.fit(tfidf_matrix)

# 每个类别提取关键词
def display_topics(model, feature_names, num_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Cluster {topic_idx}"] = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
    return topics

num_top_words = 10
feature_names = tfidf_vectorizer.get_feature_names_out()
topics = display_topics(lda, feature_names, num_top_words)

print("Cluster Keywords:")
for cluster, keywords in topics.items():
    print(f"{cluster}: {keywords}")
