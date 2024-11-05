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
import re
from nltk.stem import PorterStemmer,WordNetLemmatizer
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 加载 BERT 基础模型和分词器
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()


# 读取CSV文件
df = pd.read_csv('dataset/question_num.csv')
texts = df['question'].tolist()

# PCA HERE: 0:sport 1:business 2:tech 3:politics 4:entertainment
# label_map = {0: 'tech'401, 1: 'sport'511, 2: 'business'510, 3: 'politics'417, 4: 'entertainment'386}
labels = df['category_num'].replace({0: 2, 1: 0, 2: 1})
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


from sklearn.preprocessing import StandardScaler  # 或者使用 MinMaxScaler

# 数据标准化
scaler = StandardScaler()  # 或者使用 MinMaxScaler()
X_scaled = scaler.fit_transform(X_combined)  # 进行标准化


#使用 PCA 降维至 50 维
# pca = PCA(n_components=50, random_state=42)
# re_embeddings = pca.fit_transform(X_combined)  # 转换为 NumPy 数组
# re_embeddings = pca.fit_transform(bert_embeddings)  # 转换为 NumPy 数组


# # 使用 UMAP 降维至 50 维 BETTER THAN PCA n_components=50, random_state=42
# re_embeddings = UMAP(n_neighbors=15, min_dist=0.1, n_components=2).fit_transform(X_combined)


# 绘制轮廓系数随K值的变化，find best k
def find_optimal_k(embeddings, k_range):
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
        scores.append(silhouette_score(embeddings, kmeans.labels_))
    return scores
#######################################################
# Define a range of n_components to test
n_components_range = [2, 5, 10, 20, 30, 50]

# Store results for evaluation
umap_results = {}

# Iterate over each n_components value
for n_components in n_components_range:
    print(f"Testing UMAP with n_components={n_components}")
    
    # Apply UMAP
    re_embeddings = UMAP(n_components=n_components, random_state=42).fit_transform(X_scaled)

    # Determine optimal K using silhouette score
    k_range = range(3, 11)
    optimal_k_scores = find_optimal_k(re_embeddings, k_range)
    
    # Store the best silhouette score for the current n_components
    best_score = max(optimal_k_scores)
    best_k = optimal_k_scores.index(best_score) + 3  # Get best K
    umap_results[n_components] = {'best_k': best_k, 'best_score': best_score}

# Display the results
for n_components, result in umap_results.items():
    print(f"n_components: {n_components}, Best K: {result['best_k']}, Silhouette Score: {result['best_score']:.4f}")

#result：
# n_components: 2, Best K: 5, Silhouette Score: 0.5794
# n_components: 5, Best K: 5, Silhouette Score: 0.5994
# n_components: 10, Best K: 5, Silhouette Score: 0.6000
# n_components: 20, Best K: 5, Silhouette Score: 0.6018
# n_components: 30, Best K: 5, Silhouette Score: 0.5929
# n_components: 50, Best K: 5, Silhouette Score: 0.5989