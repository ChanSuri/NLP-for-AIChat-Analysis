import nltk
import pandas as pd
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# 下载 NLTK 停用词
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 示例文档数据
df = pd.read_csv('dataset/Translated_Text.csv')
documents = df['question_translated']

# 文本预处理函数
def preprocess(text):
    """对文本进行分词、去停用词和简单清理"""
    return [word for word in simple_preprocess(text) if word not in stop_words]

# 预处理文档
processed_docs = [preprocess(doc) for doc in documents]

# 创建字典和语料库
dictionary = Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# 打印预处理结果
print("预处理后的文档:")
print(processed_docs)
print("\nBag-of-Words 表示:")
print(corpus)

# 训练 LDA 模型
num_topics = 3  # 设置主题数量
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    passes=10,  # 训练轮次
    random_state=42
)

# 输出每个主题的关键词
print("\nLDA 模型的主题关键词:")
for idx, topic in lda_model.print_topics(num_words=5):
    print(f"主题 {idx + 1}: {topic}")

# 分析文档的主题分布
print("\n每个文档的主题分布:")
for i, doc in enumerate(corpus):
    doc_topics = lda_model.get_document_topics(doc)
    print(f"文档 {i + 1}: {doc_topics}")

# 可视化主题
print("\n生成主题模型的可视化...")
lda_display = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(lda_display, 'lda_visualization.html')  # 保存为 HTML 文件
print("可视化已保存为 'lda_visualization.html'，请在浏览器中打开。")
