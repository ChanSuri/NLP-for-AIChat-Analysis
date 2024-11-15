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
# nltk.download("wordnet")
# nltk.download("punkt")

stop_words = set(stopwords.words("english"))
custom_stop_words = {"the", "one","uno","una","para"}
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


# # 自定义的文本预处理函数
# def preprocess_text(text):
#     # 去除标点符号、特殊字符并小写化
#     text = re.sub(r'[^\w\s]', '', text.lower())
#     # 标记词性
#     tokens = word_tokenize(text)
#     pos_tags = pos_tag(tokens)
#     # 仅保留名词和形容词等非动词
#     filtered_words = [
#         lemmatizer.lemmatize(word) for word, pos in pos_tags
#         #if word not in stop_words and pos.startswith(('N','J'))  # 只保留名词(N)和形容词(J)
#     ]
#     return " ".join(filtered_words)

df = pd.read_csv('dataset/ProjectText.csv')
df["cleaned_text"] = df['one_liner'].apply(preprocess_text)

# RoBERTa generate embedding
model_name = "roberta-base" # 3:0.53 / 5:0.48
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define a function to get sentence embeddings
def get_embeddings(texts):
    # Tokenize without specifying decoder inputs
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        encoder_outputs = model(**inputs)
    # Mean pooling on the encoder's last hidden states
    embeddings = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Get embeddings for the cleaned text
text_embeddings = get_embeddings(df["cleaned_text"].tolist())

# Use UMAP for dimensionality reduction
reducer = umap.UMAP(n_neighbors=20, n_components=2, random_state=42)
reduced_embeddings = reducer.fit_transform(text_embeddings)

# Find the best K for clustering
min_k = 2
max_k = 10
best_k = None
best_silhouette_score = -1

for k in range(min_k, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)
    score = silhouette_score(reduced_embeddings, cluster_labels)
    print(f"K={k} -> Silhouette Score: {score}")
    if score > best_silhouette_score:
        best_silhouette_score = score
        best_k = k

print(f"\nBest K (with highest Silhouette Score): {best_k}")
print(f"Best Silhouette Score: {best_silhouette_score}")

# Cluster with the best K
best_kmeans = KMeans(n_clusters=best_k, random_state=42)
df['cluster'] = best_kmeans.fit_predict(reduced_embeddings)

print("Cluster counts:")
print(df['cluster'].value_counts())

# filter useless words 
def filter_keywords(keywords, min_length=2):
    """
    :param keywords: List of keywords generated by the model
    :param min_length: Minimum length of the keywords to consider
    :return: Filtered list of keywords
    """
    filtered_keywords = [kw for kw in keywords if len(kw) > min_length and kw not in stop_words]
    tagged_keywords = pos_tag(filtered_keywords)
    useful_keywords = [kw for kw, pos in tagged_keywords if pos.startswith(('N', 'J','V'))] 
    
    return useful_keywords


tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=5,stop_words="english", ngram_range=(1, 3))  #
tfidf_matrix = tfidf_vectorizer.fit_transform(df["cleaned_text"])

lda = LatentDirichletAllocation(n_components=best_k, random_state=0)
lda.fit(tfidf_matrix)

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


# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=df['cluster'], cmap='viridis', s=50, alpha=0.6)
plt.colorbar(label='Cluster Label')
plt.title('KMeans Clustering Visualization with RoBERT Embeddings')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.show()


