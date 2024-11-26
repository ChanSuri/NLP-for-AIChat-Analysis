import re
from nltk.stem import WordNetLemmatizer
from googletrans import Translator
from translatepy import Translator as TranslatepyTranslator
from langdetect import detect
import pandas as pd
import nltk


lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return text  # NULL

    try:
        text = text.lower()

        # remove Binary num
        text = re.sub(r'B[01]{8}(,\s*|\s*)', '', text)

        # 统一移除代码相关内容：PROGMEM数据块、函数代码块、#include语句、函数调用
        text = re.sub(
            r'const\s+progmem\s+uint8_t\s+\w+\[\]\s*=\s*{[^}]*};|'  # PROGMEM 数据块
            r'void\s+\w+\s*{[^}]*}|'
            r'(?i)#include\s+<[^>]+>|'
            r'[A-Za-z_]+\([^)]*\)',  # 函数调用
            'code', text, flags=re.DOTALL 
        )

        # 替换 URL 为 "url"
        text = re.sub(r'https?://\S+', 'url', text)

        text = re.sub(r'/[\w/.-]+', 'path', text)

        # 移除多余空格和换行符
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = nltk.word_tokenize(text)  # 分词
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # 词形还原
        text = " ".join(lemmatized_tokens)  # 合并为字符串
        
    except Exception as e:
        print(f"Error during preprocessing: {text}, Error: {e}")
        return text  # return orginal text while error
    return text

translator = TranslatepyTranslator()
#translator = Translator()
def translate_to_english(text):
    try:
        if not isinstance(text, str) or text.strip() == "":
            return text
        
        # text = re.sub(r'https?://\S+', 'website', text)
        # text = re.sub(r'[A-Za-z_]+\([^)]*\)', 'code', text)
        detected_lang = detect(text)

        if detected_lang == "en":
            return text
        else:
            print(detect(text))
        
        translated = translator.translate(text, "English")
        return translated.result

        #translated = translator.translate(text, src=detected_lang, dest="en")
        # return translated.text

    except Exception as e:
        print(f"Error processing text: {text}, Error: {e}")
        return text


df = pd.read_csv('dataset/ArduinoGenAI.csv')

df["cleaned_text"] = df['d_question-submitted'].apply(preprocess_text)
df['question_translated'] = df['cleaned_text'].apply(translate_to_english)

# 保存结果
df.to_csv('dataset/Translated_Text.csv', index=False)