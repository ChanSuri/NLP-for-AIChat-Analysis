import torch
from transformers import BertModel, BertTokenizer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
import re
from sklearn.preprocessing import StandardScaler  # 或者使用 MinMaxScaler
from nltk.stem import PorterStemmer,WordNetLemmatizer
import seaborn as sns
from sklearn.metrics import confusion_matrix
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# 加载 BERT 基础模型和分词器
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()


# 读取CSV文件
df = pd.read_csv('dataset/question_num.csv')
texts = df['question'].tolist()

labels = df['category_num']
label_counts = labels.value_counts().sort_index()

#stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


#text preprocess
def preprocess_texts(texts):
    
    processed_texts = []
    for text in texts:
        # 转为小写
        text = text.lower()
        # 去除标点和数字
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # 去除停用词
        text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
        # 词干提取
        #text = ' '.join([stemmer.stem(word) for word in text.split()])
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        processed_texts.append(text)
    return processed_texts

# 预处理文本
texts = preprocess_texts(texts)

# 使用 TF-IDF 提取特征
vectorizer = TfidfVectorizer(max_features=1000, min_df=2)  # max_features 可以根据需要调整
X_tfidf = vectorizer.fit_transform(texts)

#text into vector
def get_embedding(text):
    # Tokenize 输入文本
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    
    # 获取 BERT 输出
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 使用 BERT 的 [CLS] 向量作为文本的嵌入表示
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return cls_embedding

# 获取 BERT 嵌入
embeddings = torch.stack([get_embedding(text) for text in texts])
# 将 BERT 嵌入转换为稠密矩阵
bert_embeddings = embeddings.cpu().numpy()  # 转换为 NumPy 数组
# 将 BERT 嵌入和 TF-IDF 特征结合
X_combined = np.hstack((bert_embeddings, X_tfidf.toarray()))  # 将稠密 BERT 嵌入与稀疏 TF-IDF 特征结合

#standerdize
# 数据标准化
scaler = StandardScaler()  # 或者使用 MinMaxScaler()
X_scaled = scaler.fit_transform(X_combined)  # 进行标准化

#使用 PCA 降维至 50 维
# pca = PCA(n_components=50, random_state=42)
# re_embeddings = pca.fit_transform(X_combined)  # 转换为 NumPy 数组
# re_embeddings = pca.fit_transform(bert_embeddings)  # 转换为 NumPy 数组


# 使用 UMAP 降维至 50 维 BETTER THAN PCA
re_embeddings = UMAP(n_components=20, random_state=42).fit_transform(X_scaled)

#KMEANS###################################################################################
# 绘制轮廓系数随K值的变化，find best k
def find_optimal_k(embeddings, k_range):
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
        scores.append(silhouette_score(embeddings, kmeans.labels_))
    return scores

# find optimal k
k_range = range(3, 11)
optimal_k_scores = find_optimal_k(re_embeddings,k_range)
plt.plot(k_range, optimal_k_scores,marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Optimal K Selection")
plt.xticks(k_range)  # Set x-ticks to match k values
plt.grid()
plt.show()

num_clusters = optimal_k_scores.index(max(optimal_k_scores))+3
print(f'Number of best clusters num: {num_clusters} : {max(optimal_k_scores)}')

# Kmeans
kmeans = KMeans(n_clusters=num_clusters,random_state=42)

# 进行聚类
kmeans.fit(re_embeddings)
cluster_labels = kmeans.labels_

# HERE: 0:sport 1:politics 2:tech 3:business 4:entertainment
# label_map = {0: 'tech'401, 1: 'sport'511, 2: 'business'510, 3: 'politics'417, 4: 'entertainment'386}
cluster_labels.replace({0: 1, 1: 3, 2: 0, 3: 2 })
####################################################################################

def get_top_keywords(texts, labels, cluster_num, n_words=5): #num of keywords
    # 过滤出属于指定聚类的文本
    cluster_texts = [texts[i] for i in range(len(texts)) if labels[i] == cluster_num]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, min_df=2)
    X = vectorizer.fit_transform(cluster_texts)
    # 获取词汇表和特征得分
    keywords = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1
    sorted_keywords = [keywords[i] for i in scores.argsort()[-n_words:]]
    return sorted_keywords


# 示例获取第一个聚类的关键词
for cluster_num in range(num_clusters):
    print(f"Top Keywords for Cluster {cluster_num}:", get_top_keywords(texts, cluster_labels, cluster_num=cluster_num))

# Step 2: 使用 t-SNE 将 UMAP 输出降至 2 维
# 可视化聚类结果
def plot_clusters(embeddings, cluster_labels, num_clusters):
    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for i in range(num_clusters):
        points = embeddings_2d[cluster_labels == i]
        plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {i}", alpha=0.6)
    plt.title('Visualization of Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.show()


# 绘制聚类结果
plot_clusters(re_embeddings, cluster_labels, num_clusters)

def plot_labels(cluster_labels,label_counts):
    unique, counts = np.unique(cluster_labels, return_counts=True)

    # 创建条形图
    plt.figure(figsize=(10, 5))

    # 绘制真实标签数量
    bars_actual = plt.bar(label_counts.index - 0.2, label_counts.values, width=0.4, label='Actual Labels', align='center')

    # 绘制预测标签数量
    bars_predicted = plt.bar(unique + 0.2, counts, width=0.4, label='Predicted Labels', align='center')

    # 添加标题和标签
    plt.title('Comparison of Actual and Predicted Labels')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    custom_labels = ['sport', 'business', 'tech', 'politics', 'entertainment']
    plt.xticks(range(len(label_counts)), custom_labels)  # 设置x轴刻度
    plt.legend()

    # 在柱子上方添加数字标签
    for bar in bars_actual:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

    for bar in bars_predicted:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

    # 显示图表
    plt.tight_layout()
    plt.show()

plot_labels(cluster_labels,label_counts)

# 创建可视化混淆矩阵的函数
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

cm = confusion_matrix(labels, cluster_labels)
class_names = ['tech','sport', 'business','politics', 'entertainment']
plot_confusion_matrix(cm, class_names)
accuracy = accuracy_score(labels, cluster_labels)
print(f'Predict accuracy: {accuracy:.2f}') 
#Predict accuracy: 0.94