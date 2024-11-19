import numpy as np
import pandas as pd
import torch
import re
import umap
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score

stop_words = set(stopwords.words("english"))
custom_stop_words = {"the", "one","uno","una","para","une","que","com",'por','del','ist','und','die','das','der','arduino','questo'}
stop_words.update(custom_stop_words)

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9.,!?;:'\"()\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = " ".join(lemmatized_tokens)
    return preprocessed_text

df = pd.read_csv('dataset/ProjectText.csv')
df["cleaned_text"] = df['one_liner'].apply(preprocess_text)

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        encoder_outputs = model(**inputs)
    embeddings = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

text_embeddings = get_embeddings(df["cleaned_text"].tolist())

# First round of clustering with UMAP + KMeans
reducer = umap.UMAP(n_neighbors=20, n_components=2, random_state=42)
reduced_embeddings = reducer.fit_transform(text_embeddings)

min_k = 2
max_k = 10
best_k = None
best_silhouette_score = -1

# Find the best K for first round clustering
for k in range(min_k, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)
    score = silhouette_score(reduced_embeddings, cluster_labels)
    if score > best_silhouette_score:
        best_silhouette_score = score
        best_k = k

print(f"Best K (first round clustering): {best_k}, Score: {best_silhouette_score}")

# Perform first round of clustering
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['big_cluster'] = kmeans.fit_predict(reduced_embeddings)

# Visualize first round of clustering
plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=df['big_cluster'], cmap='viridis', s=50, alpha=0.6)
plt.colorbar(label='Big Cluster Label')
plt.title('First Round of Clustering (UMAP + KMeans)')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.show()

# Now, for each large cluster, refine the clustering
small_cluster_keywords = {}
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# 使用原始的embeddings进行降维
def reduce_dimensionality_with_pca(embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

# 对每个大聚类进行细分聚类并生成图表
small_cluster_keywords = {}

for cluster in df['big_cluster'].unique():
    # Filter texts for this cluster
    cluster_texts = df[df['big_cluster'] == cluster]["cleaned_text"].tolist()
    
    # Get embeddings for the cluster (use original embeddings)
    cluster_embeddings = get_embeddings(cluster_texts)
    
    # 使用PCA降维
    cluster_reduced_embeddings = reduce_dimensionality_with_pca(cluster_embeddings, n_components=2)
    
    # 使用DBSCAN进行细分聚类（替代KMeans）
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels_small = dbscan.fit_predict(cluster_reduced_embeddings)
    
    # Perform keyword extraction (LDA) as before
    tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=5, stop_words="english", ngram_range=(1, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(cluster_texts)
    
    lda = LatentDirichletAllocation(n_components=len(set(cluster_labels_small)), random_state=0)
    lda.fit(tfidf_matrix)
    
    def filter_keywords(keywords, min_length=2):
        filtered_keywords = [kw for kw in keywords if len(kw) > min_length and kw not in stop_words]
        tagged_keywords = pos_tag(filtered_keywords)
        useful_keywords = [kw for kw, pos in tagged_keywords if pos.startswith(('N', 'J', 'V'))] 
        return useful_keywords

    def display_topics(model, feature_names, num_top_words):
        topics = {}
        for topic_idx, topic in enumerate(model.components_):
            topic_keywords = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
            topics[f"Small Cluster {topic_idx}"] = filter_keywords(topic_keywords)
        return topics

    num_top_words = 10
    feature_names = tfidf_vectorizer.get_feature_names_out()
    topics = display_topics(lda, feature_names, num_top_words)
    small_cluster_keywords[cluster] = topics

    # 可视化每个大聚类的小聚类分布
    plt.figure(figsize=(8, 6))
    plt.scatter(cluster_reduced_embeddings[:, 0], cluster_reduced_embeddings[:, 1], c=cluster_labels_small, cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(label='Small Cluster Label')
    plt.title(f'Second Round of Clustering for Big Cluster {cluster} (DBSCAN + PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

# Display small cluster keywords
for cluster, topics in small_cluster_keywords.items():
    print(f"\nBig Cluster {cluster} - Small Clusters Keywords:")
    for small_cluster, keywords in topics.items():
        print(f"{small_cluster}: {keywords}")
