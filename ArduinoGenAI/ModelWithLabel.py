from umap import UMAP
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler  # or MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.decomposition import PCA

# 1. 数据读取
df = pd.read_csv('dataset/Translated_Text.csv')
texts = df['question_translated'].dropna().tolist()  # 获取聊天文本并去除缺失值

# 2. 文本预处理函数
def preprocess_text(text):
    # 预处理：小写化，去除停用词，标点符号等
    text = text.lower()
    text = ' '.join([word for word in nltk.word_tokenize(text) if word.isalnum()])
    return text

# 对每个文本进行预处理
processed_texts = [preprocess_text(text) for text in texts]

# 3. 使用BERT获取文本嵌入
model_name = "roberta-base"#"facebook/bart-base""distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # 获取句子级别的BERT嵌入（平均最后一层的输出）
        embeddings = outputs.pooler_output 
        # embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# 获取所有文本的BERT嵌入
bert_embeddings = get_bert_embeddings(processed_texts)

# 4. 使用K-means聚类进行聚类分析
num_clusters = 4  # 设定簇的数量
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(bert_embeddings)

# 输出聚类结果
df['cluster'] = kmeans.labels_
print(df[['question_translated', 'cluster']])
# 5. 使用TF-IDF和Logistic Regression进行文本分类（假设你有标签数据）
# df['label']是预定义的标签列

X_train, X_test, y_train, y_test = train_test_split(processed_texts, df['label'], test_size=0.3, random_state=42)

# 使用TF-IDF向量化文本数据
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 使用Logistic Regression进行训练
clf = LogisticRegression()
clf.fit(X_train_tfidf, y_train)

# 预测并评估模型
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# 6. 评估聚类效果
silhouette_avg = silhouette_score(bert_embeddings, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg}")

# 可视化聚类结果
def plot_clusters(embeddings, labels, num_clusters):
    # pca = PCA(n_components=2)
    # reduced_embeddings = pca.fit_transform(embeddings)
    reducer = UMAP(n_components=2, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)


    # 创建一个散点图
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='Set2', s=100, alpha=0.7, edgecolor='k')
    plt.title(f'K-Means Clustering of Chat Texts (n_clusters={num_clusters})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.show()

# 绘制聚类图
plot_clusters(bert_embeddings, df['cluster'], num_clusters)

# 可视化分类结果（仅适用于有标签数据的情感分类示例）
def plot_classification_results(y_test, y_pred):
    # 使用混淆矩阵来展示分类结果
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['create code', 'fix error', 'explain','content'], yticklabels=['create code', 'fix error', 'explain','content'])
    plt.title('Confusion Matrix for Sentiment Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# 绘制分类结果图（情感分类示例）
plot_classification_results(y_test, y_pred)
