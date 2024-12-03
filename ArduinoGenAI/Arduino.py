from umap import UMAP
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler  # or MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.decomposition import PCA
import joblib
import umap


# 1. 数据读取
df = pd.read_csv('dataset/Translated_Text.csv', sep=None, engine='python')
texts = df['question_translated'].dropna().tolist()  # 获取聊天文本并去除缺失值

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
bert_embeddings = get_bert_embeddings(texts) # 768 dimensions

from sklearn.decomposition import PCA

# PCA 降维
pca = PCA(n_components=50, random_state=42)  # 降到 50 维
reduced_embeddings = pca.fit_transform(bert_embeddings)

# # Use UMAP for dimensionality reduction
# reducer = umap.UMAP(n_neighbors=20, n_components=2, random_state=42)
# reduced_embeddings = reducer.fit_transform(bert_embeddings)

# 4. 使用K-means聚类进行聚类分析
num_clusters = 4  # 设定簇的数量
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(reduced_embeddings)

# 输出聚类结果
df['cluster'] = kmeans.labels_
print(df[['question_translated', 'cluster']])

cluster_to_label = {0: 'Knowledge', 1: 'Code Generation', 2: 'Fix error', 3: 'other'}
df['label'] = df['cluster'].map(cluster_to_label)

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

X = reduced_embeddings
y = df['label_encoded']

# 5. 使用Logistic Regression进行文本分类（假设你有标签数据）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # 使用Logistic Regression进行训练
# clf = LogisticRegression()
# clf.fit(X_train, y_train)
# # 预测并评估模型
# y_pred = clf.predict(X_test)
# print("LR:" + classification_report(y_test, y_pred))

# random forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("分类模型性能报告:")
print("RF:" + classification_report(y_test, y_pred, target_names=le.classes_))

# predict new question
# def predict_request(new_request):
#     # 使用 RoBERTa 提取新请求的嵌入
#     new_request_embedding = get_bert_embeddings([new_request], tokenizer, model)
#     # 分类模型预测
#     predicted_label_encoded = rf.predict(new_request_embedding)[0]
#     # 转换为文本标签
#     predicted_label = le.inverse_transform([predicted_label_encoded])[0]
#     return predicted_label
# new_request = "How to create a code for LED？"
# predicted_category = predict_request(new_request)


# 6. 评估聚类效果
silhouette_avg = silhouette_score(reduced_embeddings, kmeans.labels_)
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
plot_clusters(reduced_embeddings, df['cluster'], num_clusters)

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


# joblib.dump(kmeans, 'kmeans_model.pkl')
# joblib.dump(rf, 'rf_model.pkl')
# joblib.dump(le, 'label_encoder.pkl')
# torch.save(model.state_dict(), 'roberta_model.pth')
# torch.save(tokenizer, 'roberta_tokenizer.pt')