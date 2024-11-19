import numpy as np
import pandas as pd
import torch
import re
import umap
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from keybert import KeyBERT
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# nltk.download("wordnet")
# nltk.download("punkt")
stop_words = set(stopwords.words("english"))
custom_stop_words = {"the", "one", "uno", "una", "para","arduino"}
stop_words.update(custom_stop_words)

lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9.,!?;:'\"()\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = " ".join(lemmatized_tokens)
    return preprocessed_text

# Load dataset and preprocess text
df = pd.read_csv('dataset/ProjectText.csv')
df["cleaned_text"] = df['one_liner'].apply(preprocess_text)

# Load RoBERTa model and tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Generate embeddings using RoBERTa
def get_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Generate embeddings for the dataset
text_embeddings = get_embeddings(df["cleaned_text"].tolist(), tokenizer, model)

# Dimensionality reduction using UMAP
reducer = umap.UMAP(n_neighbors=20, n_components=2, random_state=42)
reduced_embeddings = reducer.fit_transform(text_embeddings)

# First clustering: KMeans for broad categories
best_k = 3  # Adjust based on data
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['broad_cluster'] = kmeans.fit_predict(reduced_embeddings)
score = silhouette_score(reduced_embeddings, df['broad_cluster'])

print(f"Score: {score}")
print("First Big Cluster counts:")
print(df['broad_cluster'].value_counts())

# Visualize broad clusters
plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=df['broad_cluster'], cmap='viridis', marker='o')
plt.title('Broad Categories Clustering (KMeans)', fontsize=16)
plt.colorbar(label='Cluster Label')
plt.show()

# Second clustering: GMM for fine-grained categories within each broad cluster
fine_grained_labels = {}
pca = PCA(n_components=2)  # For visualization later

for broad_label in df['broad_cluster'].unique():
    # Filter embeddings for the current broad cluster
    indices = df[df['broad_cluster'] == broad_label].index
    cluster_embeddings = text_embeddings[indices]

    # Apply GMM for fine-grained clustering
    gmm = GaussianMixture(n_components=3, random_state=42)  # Adjust n_components as needed
    sub_labels = gmm.fit_predict(cluster_embeddings)
    fine_grained_labels[broad_label] = sub_labels

    # Assign fine-grained cluster labels to the dataframe
    df.loc[indices, 'fine_cluster'] = sub_labels

    # Visualize fine-grained clusters for the current broad cluster
    reduced_sub_embeddings = pca.fit_transform(cluster_embeddings)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_sub_embeddings[:, 0], reduced_sub_embeddings[:, 1], c=sub_labels, cmap='tab10', marker='o')
    plt.title(f'Fine-Grained Clustering for Broad Cluster {broad_label}', fontsize=16)
    plt.colorbar(label='Sub-Cluster Label')
    plt.show()

# Keyword extraction using KeyBERT
kw_model = KeyBERT()

def extract_keywords_for_clusters(texts, labels):
    keywords_per_cluster = {}
    for label in np.unique(labels):
        cluster_texts = [texts[idx] for idx, l in enumerate(labels) if l == label]
        if len(cluster_texts) > 0:
            keywords = kw_model.extract_keywords(" ".join(cluster_texts), top_n=5)
            keywords_per_cluster[label] = [kw[0] for kw in keywords]
    return keywords_per_cluster

# Extract and display keywords for broad categories
broad_category_keywords = extract_keywords_for_clusters(df["cleaned_text"].tolist(), df['broad_cluster'])
print("Broad Category Keywords:")
for label, keywords in broad_category_keywords.items():
    print(f"Cluster {label}: {keywords}")

# Extract and display keywords for fine-grained categories
print("\nFine-Grained Category Keywords:")
for broad_label, sub_labels in fine_grained_labels.items():
    indices = df[df['broad_cluster'] == broad_label].index
    texts_in_broad_cluster = df.loc[indices, "cleaned_text"].tolist()
    sub_keywords = extract_keywords_for_clusters(texts_in_broad_cluster, sub_labels)
    print(f"Broad Cluster {broad_label} Sub-Clusters:")
    for sub_label, keywords in sub_keywords.items():
        print(f"  Sub-Cluster {sub_label}: {keywords}")
