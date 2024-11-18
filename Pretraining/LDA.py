import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import re
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 可使用 t-SNE 或 PCA 降维到 2 维，再进行散点图可视化
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#text preprocess
def preprocess_texts(texts):
    stemmer = PorterStemmer()
    #lemmatizer = WordNetLemmatizer()
    processed_texts = []
    for text in texts:
        # 转为小写
        text = text.lower()
        # 去除标点和数字
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # 去除停用词
        text = ' '.join([word for word in text.split() if  len(word) > 2 and word not in ENGLISH_STOP_WORDS])
        # 词干提取
        text = ' '.join([stemmer.stem(word) for word in text.split()])
        #text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        processed_texts.append(text)
    return processed_texts

def calculate_coherence_score(lda_model, texts, dictionary):
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()

def calculate_perplexity(lda_model, corpus):
    return lda_model.log_perplexity(corpus)

# 选择降维方法：t-SNE 或 PCA
def visualize_clusters(embeddings, labels, method='tsne'):
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)

    embeddings_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique_labels = set(labels)
    for label in unique_labels:
        plt.scatter(
            embeddings_2d[labels == label, 0],
            embeddings_2d[labels == label, 1],
            label=f"Cluster {label}",
            alpha=0.6
        )
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.title(f"KMeans Clustering Visualization using {method.upper()}")
    plt.legend()
    plt.show()

def plot_coherence_perplexity(topic_range,coherence_scores,perplexity_scores):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Coherence Score
    color = 'tab:blue'
    ax1.set_xlabel('Number of Topics')
    ax1.set_ylabel('Coherence Score', color=color)
    ax1.plot(topic_range, coherence_scores, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Perplexity Score
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Perplexity', color=color)
    ax2.plot(topic_range, perplexity_scores, marker='x', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Coherence Score and Perplexity vs. Number of Topics")
    plt.show()

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
    custom_labels = ['Technology', 'Business', 'Sport', 'Entertainment', 'Politics']
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

# 创建可视化混淆矩阵的函数
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


if __name__ == '__main__':
    # 读取CSV文件
    df = pd.read_csv('dataset/question_num.csv')
    texts = df['question'].tolist()
    
    labels = df['category_num']#.replace({1: 2, 2: 1, 3: 4, 4: 3})
    label_counts = labels.value_counts().sort_index()

    # 文本预处理（同之前的预处理函数）
    texts = preprocess_texts(texts)  # 使用您之前定义的 preprocess_texts 函数

    # # Tokenize each processed text into a list of words
    # tokenized_texts = [text.split() for text in texts]

    # # Create a dictionary and corpus
    # dictionary = corpora.Dictionary(tokenized_texts)
    # corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    # # Determine the optimal number of topics
    # topic_range = range(2, 10)
    # coherence_scores = []
    # perplexity_scores = []

    # for n in topic_range:
    #     lda_model = LdaModel(corpus, num_topics=n, id2word=dictionary, passes=10, random_state=42)
    #     coherence = calculate_coherence_score(lda_model, tokenized_texts, dictionary)
    #     perplexity = calculate_perplexity(lda_model, corpus)
        
    #     coherence_scores.append(coherence)
    #     perplexity_scores.append(perplexity)
    #     #print(f"Number of Topics: {n} - Coherence Score: {coherence}, Perplexity: {perplexity}")
    # plot_coherence_perplexity(topic_range,coherence_scores,perplexity_scores)

    # 创建词频矩阵
    vectorizer = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(texts)

    # 定义并训练 LDA 模型
    n_topics = 5  # 根据数据的多样性调整主题数
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    # 获取每个文档的主题分布矩阵
    doc_topic_distributions = lda.transform(X)

    # 使用 KMeans 聚类主题分布
    kmeans = KMeans(n_clusters=n_topics, random_state=42)
    kmeans.fit(doc_topic_distributions)
    cluster_labels = kmeans.labels_

    # 计算 Silhouette Score
    silhouette_avg = silhouette_score(doc_topic_distributions, cluster_labels)
    print(f"Silhouette Score with LDA: {silhouette_avg}")

    # 打印每个主题的主要关键词
    n_top_words = 5
    terms = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords = [terms[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"Top keywords for topic #{topic_idx}: {top_keywords}")


    #HERE: 0:technology 1:business 2:sport 3:entertainment 4:politics
    # label_map = {0: 'tech', 1: 'sport', 2: 'business', 3: 'politics', 4: 'entertainment'}
    # label_counts = label_counts.reindex([0, 2, 1, 4, 3])
    # label_counts = label_counts.reset_index(drop=True)
    # print(label_counts)
    # plot_labels(cluster_labels,label_counts)

    # cm = confusion_matrix(labels, cluster_labels)
    # class_names = ['Technology', 'Business', 'Sport', 'Entertainment','Politics']
    # plot_confusion_matrix(cm, class_names)

    # 可视化 KMeans 聚类结果
    visualize_clusters(doc_topic_distributions, cluster_labels, method='tsne')  # 或 method='pca'