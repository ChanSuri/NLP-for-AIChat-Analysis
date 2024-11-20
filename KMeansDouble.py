import numpy as np
import pandas as pd
import torch
import re
import umap
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from keybert import KeyBERT
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# nltk.download("wordnet")
# nltk.download("punkt")
# nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
custom_stop_words = {"the", "one", "uno", "una", "para", "une", "que", "com", "por", "del", "ist", "und", "die", "das", "der", "arduino", "questo",'con','per',}
stop_words.update(custom_stop_words)

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9.,!?;:'\"()\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]# if len(token) > 2 and token not in stop_words
    preprocessed_text = " ".join(lemmatized_tokens)
    return preprocessed_text

df = pd.read_csv('dataset/ProjectText.csv')
df["cleaned_text"] = df['one_liner'].apply(preprocess_text)

# 加载RoBERTa模型
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 获取嵌入
def get_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

text_embeddings = get_embeddings(df["cleaned_text"].tolist(), tokenizer, model)

# UMAP降维
reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=42)
reduced_embeddings = reducer.fit_transform(text_embeddings)

# 自动选择最佳 K 值
def find_best_k(data, max_k=10):
    best_k, best_score = 0, -1
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        if score > best_score:
            best_k, best_score = k, score
    print(f"Best K: {best_k}, Silhouette Score: {best_score}")
    return best_k

# 初次聚类（粗分类）
best_k = find_best_k(reduced_embeddings)
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['broad_cluster'] = kmeans.fit_predict(reduced_embeddings)

print("First Big Cluster counts:")
print(df['broad_cluster'].value_counts())

# 粗聚类可视化
plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=df['broad_cluster'], cmap='viridis', marker='o')
plt.title('Broad Categories Clustering (KMeans)', fontsize=16)
plt.colorbar(label='Cluster Label')
plt.show()

# 二次细分聚类
fine_grained_labels = {}
umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, random_state=42)

for broad_label in df['broad_cluster'].unique():
    indices = df[df['broad_cluster'] == broad_label].index
    cluster_embeddings = text_embeddings[indices]

    # 降维
    reduced_embeddings = umap_reducer.fit_transform(cluster_embeddings)
    
    # 自动选择最佳 K 值
    best_k = find_best_k(reduced_embeddings)
    
    # 细分聚类
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    sub_labels = kmeans.fit_predict(reduced_embeddings)
    fine_grained_labels[broad_label] = sub_labels
    df.loc[indices, 'fine_cluster'] = sub_labels

    print(f"Broad Cluster {broad_label}:")
    cluster_counts = pd.Series(sub_labels).value_counts()
    for sub_label, count in cluster_counts.items():
        print(f"  Fine Cluster {sub_label}: {count}")
    
    # 可视化细分聚类
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=sub_labels, cmap='tab10', marker='o', alpha=0.8)
    plt.title(f'Fine-Grained Clustering for Broad Cluster {broad_label}', fontsize=16)
    plt.colorbar(scatter, label='Sub-Cluster Label')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.show()

# 关键词提取
kw_model = KeyBERT()

def extract_keywords_for_clusters(texts, labels):
    keywords_per_cluster = {}
    for label in np.unique(labels):
        cluster_texts = [texts[idx] for idx, l in enumerate(labels) if l == label]
        if len(cluster_texts) > 0:
            joined_text = " ".join(cluster_texts)
            keywords = kw_model.extract_keywords(joined_text, top_n=10)
            filtered_keywords = [kw[0] for kw in keywords if kw[0].lower() not in stop_words]
            keywords_per_cluster[label] = filtered_keywords
    return keywords_per_cluster

# 显示关键词
broad_category_keywords = extract_keywords_for_clusters(df["cleaned_text"].tolist(), df['broad_cluster'])
print("Broad Category Keywords:")
for label, keywords in broad_category_keywords.items():
    print(f"Cluster {label}: {keywords}")

# 细分类别关键词
for broad_label, sub_labels in fine_grained_labels.items():
    indices = df[df['broad_cluster'] == broad_label].index
    texts_in_broad_cluster = df.loc[indices, "cleaned_text"].tolist()
    sub_keywords = extract_keywords_for_clusters(texts_in_broad_cluster, sub_labels)
    print(f"\nBroad Cluster {broad_label} Sub-Clusters:")
    for sub_label, keywords in sub_keywords.items():
        print(f"  Sub-Cluster {sub_label}: {keywords}")

# # 生成词云
# def generate_wordcloud(text, title):
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.title(title, fontsize=16)
#     plt.axis('off')
#     plt.show()

# for broad_label in df['broad_cluster'].unique():
#     texts = df[df['broad_cluster'] == broad_label]["cleaned_text"].tolist()
#     generate_wordcloud(" ".join(texts), title=f"Broad Cluster {broad_label}")
