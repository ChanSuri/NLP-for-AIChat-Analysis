import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW  # 使用 PyTorch 的 AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 用于混淆矩阵的可视化


# 定义AI Chat问题分类的数据集
class ChatDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 创建数据集
def create_data_loader(texts, labels, tokenizer, max_len, batch_size):
    dataset = ChatDataset(texts, labels, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size)


# 读取CSV文件
df = pd.read_csv('dataset/question_num.csv')
df = df.reset_index(drop=True)  # 重置索引
questions = df['question']

# 标签映射为数字 0:tech 1:sport 2:business 3:politics 4:entertainment
# labels = [0, 1, 2, 2, 1]
labels = df['category_num']
label_map = {0: 'tech', 1: 'sport', 2: 'business', 3: 'politics', 4: 'entertainment'}

# 拆分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(questions, labels, test_size=0.2)
train_texts = train_texts.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
val_texts = val_texts.reset_index(drop=True)
val_labels = val_labels.reset_index(drop=True)

# 加载BERT的分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)  # 5 categories

# 设置一些超参数
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3 #avoid overfitting
LEARNING_RATE = 2e-5

# 创建数据加载器
train_loader = create_data_loader(train_texts, train_labels, tokenizer, MAX_LEN, BATCH_SIZE)
val_loader = create_data_loader(val_texts, val_labels, tokenizer, MAX_LEN, BATCH_SIZE)

# 使用 AdamW 进行优化
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

train_losses = []
train_accuracies = []
val_accuracies = []

# 训练模型
def train_epoch(model, data_loader, optimizer, device):
    model = model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0

    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / total_predictions, np.mean(losses)

# 验证模型
def eval_model(model, data_loader, device):
    model = model.eval()
    correct_predictions = 0
    total_predictions = 0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    return correct_predictions.double() / total_predictions,true_labels, pred_labels

# 训练和评估模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    # 训练并保存训练损失和准确率
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc.item())

    print(f"Train loss: {train_loss}, accuracy: {train_acc}")

    # 验证并保存验证准确率
    val_acc, true_labels, pred_labels = eval_model(model, val_loader, device)
    val_accuracies.append(val_acc.item())

    print(f"Validation accuracy: {val_acc}")

# 保存模型
model.save_pretrained('chat_classifier_model')

# 预测新问题的分类
def predict_question_class(question, model, tokenizer, max_len):
    model = model.eval()
    encoding = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted_class = torch.max(logits, dim=1)

    return predicted_class.item()

# 测试问题
new_question = "Torino: beats Juve in the derby and overtakes them in the standings. And now it's 4 wins in a row..."
predicted_class = predict_question_class(new_question, model, tokenizer, MAX_LEN)
print(f"Predicted class: {label_map[predicted_class]}")


# 打印分类报告
report = classification_report(true_labels, pred_labels, target_names=label_map.values(), output_dict=True)
# print("\nClassification Report:")
# print(report)

# 生成混淆矩阵
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_map.values(), yticklabels=label_map.values())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# 绘制训练和验证的准确率与损失图
plt.figure(figsize=(12, 6))

# 准确率图
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 损失图
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()