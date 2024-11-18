import spacy
nlp = spacy.load("en_core_web_sm")


def extract_keywords_and_entities(text):
    # 使用 SpaCy 进行解析
    doc = nlp(text)
    
    # 提取命名实体（如人名、组织、地点等）
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # 提取关键词（可以使用名词和重要的词性作为关键词）
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'PROPN']]
    
    return keywords, entities

# 测试
user_request = "Can you help me with Arduino setup and connecting it to AWS?"
keywords, entities = extract_keywords_and_entities(user_request)

print("Keywords:", keywords)
print("Entities:", entities)

