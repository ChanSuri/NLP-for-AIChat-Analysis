import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from googletrans import Translator
from langdetect import detect
import pandas as pd
import nltk

# stop_words = set(stopwords.words("english"))
# custom_stop_words = {"the", "one","uno","una","para"}
# stop_words.update(custom_stop_words)

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return text  # NULL

    try:
        text = re.sub(r'\n.*', '', text)
        text = re.sub(r'\.*', '', text)
        text = re.sub(r'(?i)#include\s+<[^>]+>', ' ', text)  # 匹配 C/C++ 的 include 语句
        text = re.sub(r'[A-Za-z_]+\([^)]*\)', 'code', text)  # 匹配函数调用形式的代码
        text = re.sub(r'https?://\S+', 'website', text)      # 替换 URL
        tokens = nltk.word_tokenize(text)  # 分词
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # 词形还原
        text = " ".join(lemmatized_tokens)  # 合并为字符串
        
    except Exception as e:
        print(f"Error during preprocessing: {text}, Error: {e}")
        text = text[:100]
        return text  # 发生异常时返回原始文本
    return text

translator = Translator()
def translate_to_english(text):
    try:
        if not isinstance(text, str) or text.strip() == "":
            return text
        
        text = re.sub(r'https?://\S+', 'website', text)
        text = re.sub(r'[A-Za-z_]+\([^)]*\)', 'code', text)
        detected_lang = detect(text)

        if detected_lang == "en":
            return text

        translated = translator.translate(text, src=detected_lang, dest="en")
        return translated.text

    except Exception as e:
        print(f"Error processing text: {text}, Error: {e}")
        return text


df = pd.read_csv('dataset/ArduinoGenAI.csv')

#df["cleaned_text"] = df['d_question-submitted'].apply(preprocess_text)
df['question_translated'] = df['d_question-submitted'].apply(translate_to_english)

# 保存结果
df.to_csv('dataset/Translated_Text.csv', index=False)
