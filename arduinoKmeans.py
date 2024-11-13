import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import umap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
from nltk import pos_tag, word_tokenize
from sentence_transformers import SentenceTransformer

# # 下载 NLTK 词性标注所需的资源
# nltk.download("averaged_perceptron_tagger")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("punkt")

# 加载停用词
stop_words = set(stopwords.words("english"))

# 初始化词形还原器
lemmatizer = WordNetLemmatizer()

# 自定义的文本预处理函数
def preprocess_text(text):
    # 去除标点符号、特殊字符并小写化
    text = re.sub(r'[^\w\s]', '', text.lower())
    # 标记词性
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    # 仅保留名词和形容词等非动词
    filtered_words = [
        lemmatizer.lemmatize(word) for word, pos in pos_tags
        if word not in stop_words and pos.startswith(('N','J'))  # 只保留名词(N)和形容词(J)
    ]
    return " ".join(filtered_words)

# def preprocess_text(text):
#     # 去除标点符号、特殊字符并小写化
#     text = re.sub(r'[^\w\s]', '', text.lower())
#     # 去除停用词
#     stop_words = set(stopwords.words("english"))
#     words = [word for word in text.split() if  len(word) > 2  and word not in stop_words]
#     # 词形还原
#     lemmatizer = WordNetLemmatizer()
#     words = [lemmatizer.lemmatize(word) for word in words]
#     return " ".join(words)

df = pd.read_csv('dataset/ProjectText.csv')
df["cleaned_text"] = df['one_liner'].apply(preprocess_text)


# 2. 特征提取 (使用 BERT)
# model = SentenceTransformer('all-mpnet-base-v2')
# text_embeddings = model.encode(df["cleaned_text"].tolist())
model = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = model.encode(df["cleaned_text"].tolist())

# 使用 UMAP 降维 (可选)
reducer = umap.UMAP(n_neighbors=30, n_components=2, random_state=42) #n_neighbors bigger, cluster bigger
reduced_embeddings = reducer.fit_transform(text_embeddings)

# 3. 寻找最佳 K
# 定义 K 值范围和交叉验证
min_k = 4  # 最小的聚类数
max_k = 20  # 最大的聚类数
best_k = None
best_silhouette_score = -1

# 使用 KMeans 进行不同 K 值的聚类
for k in range(min_k, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)
    
    # 计算 Silhouette Score
    score = silhouette_score(reduced_embeddings, cluster_labels)
    print(f"K={k} -> Silhouette Score: {score}")
    
    if score > best_silhouette_score:
        best_silhouette_score = score
        best_k = k

print(f"\nBest K (with highest Silhouette Score): {best_k}")
print(f"Best Silhouette Score: {best_silhouette_score}")

# 4. 使用最佳 K 值进行 KMeans 聚类
best_kmeans = KMeans(n_clusters=best_k, random_state=42)
df['cluster'] = best_kmeans.fit_predict(reduced_embeddings)

# 输出聚类结果
print("Cluster counts:")
print(df['cluster'].value_counts())

# 计算聚类的 Silhouette Score
silhouette_avg = silhouette_score(reduced_embeddings, df['cluster'])
print(f"Silhouette Score for best K: {silhouette_avg}")

# 提取关键词 (使用 TF-IDF + LDA) 通过调整 max_df 和 min_df，可以减少在所有文档中都出现的普遍词（max_df 限制），以及只在少数文档中出现的稀有词（min_df 限制）。

# filter useless words 
def filter_keywords(keywords, min_length=2, common_threshold=0.5):
    filtered = [kw for kw in keywords if len(kw) > min_length]
    return filtered

tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=5,stop_words="english", ngram_range=(1, 3))  #
tfidf_matrix = tfidf_vectorizer.fit_transform(df["cleaned_text"])

lda = LatentDirichletAllocation(n_components=best_k, random_state=0)
lda.fit(tfidf_matrix)

# 每个类别提取关键词
def display_topics(model, feature_names, num_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_keywords = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
        topics[f"Cluster {topic_idx}"] = filter_keywords(topic_keywords)
    return topics

num_top_words = 10
feature_names = tfidf_vectorizer.get_feature_names_out()
topics = display_topics(lda, feature_names, num_top_words)

print("Cluster Keywords:")
for cluster, keywords in topics.items():
    print(f"{cluster}: {keywords}")

import matplotlib.pyplot as plt

# 使用 UMAP 降维后的结果和聚类标签绘制散点图
plt.figure(figsize=(10, 8))

# 绘制每个点的散点图，颜色根据聚类标签来区分
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=df['cluster'], cmap='viridis', s=50, alpha=0.6)

# 添加色条和标签
plt.colorbar(label='Cluster Label')
plt.title('KMeans Clustering Visualization (Best K)')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')

# 显示图形
plt.show()

import seaborn as sns

# 计算每个点的密度
sns.kdeplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], cmap='Blues', shade=True, alpha=0.3)

# 绘制散点图
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=df['cluster'], cmap='viridis', s=50, alpha=0.6)
plt.colorbar(label='Cluster Label')
plt.title('KMeans Clustering with Density Plot')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.show()
