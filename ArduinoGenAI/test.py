import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import xgboost as xgb


# Load dataset
df = pd.read_csv('dataset/Translated_Text.csv')
print("Dataset Sample:")
print(df.head())

# Encode labels to integers
label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df['category'])  # Convert labels to numeric

# Split data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['question_translated'], df['encoded_label'], test_size=0.2, random_state=42
)

# Load the model and tokenizer
model_name =  "sentence-transformers/all-MiniLM-L6-v2" # Using a powerful model
# "roberta-base" "t5-small" "sentence-transformers/all-MiniLM-L6-v2" "sentence-transformers/all-mpnet-base-v2" "google/t5-small-lm-adapt" (高性能)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Generate embeddings using the selected model
def get_embeddings(texts):
    inputs = tokenizer(
        texts.tolist(), padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Generate embeddings for train and test sets
print("Generating embeddings for training data...")
train_embeddings = get_embeddings(train_texts)
print("Generating embeddings for testing data...")
test_embeddings = get_embeddings(test_texts)

# Optional: Dimensionality reduction using PCA or UMAP
# PCA is faster, UMAP captures non-linear patterns better
print("Performing dimensionality reduction...")
pca = PCA(n_components=50, random_state=42)
train_embeddings = pca.fit_transform(train_embeddings)
test_embeddings = pca.transform(test_embeddings)

# Optional: Use UMAP instead of PCA for better non-linear representation
# umap_reducer = umap.UMAP(n_neighbors=30, n_components=50, random_state=42)
# train_embeddings = umap_reducer.fit_transform(train_embeddings)
# test_embeddings = umap_reducer.transform(test_embeddings)

# Model Selection and Hyperparameter Tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

svc = SVC(random_state=42)
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(train_embeddings, train_labels)
best_classifier = grid_search.best_estimator_

# Cross-validation for better performance estimation
cv_scores = cross_val_score(best_classifier, train_embeddings, train_labels, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean():.2f}")

# Train and predict using the best classifier
print("Making predictions...")
predicted_labels = best_classifier.predict(test_embeddings)

# Evaluate model performance
print("Model Performance:")
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(test_labels, predicted_labels, target_names=label_encoder.classes_))

# Map predicted labels back to categories
df_test = pd.DataFrame({'text': test_texts, 'true_label': test_labels, 'predicted_label': predicted_labels})
df_test['true_category'] = label_encoder.inverse_transform(df_test['true_label'])
df_test['predicted_category'] = label_encoder.inverse_transform(df_test['predicted_label'])

# Show a few test samples
print("\nSample Predictions:")
print(df_test[['text', 'true_category', 'predicted_category']].head())

# Compute and plot confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Optional: Model ensemble using voting classifier (SVM + RandomForest + XGBoost)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_clf = xgb.XGBClassifier(random_state=42)

ensemble = VotingClassifier(estimators=[('svc', best_classifier), ('rf', rf_clf), ('xgb', xgb_clf)], voting='hard')
ensemble.fit(train_embeddings, train_labels)
predicted_labels_ensemble = ensemble.predict(test_embeddings)

# Evaluate ensemble model
ensemble_accuracy = accuracy_score(test_labels, predicted_labels_ensemble)
print(f"Ensemble Model Accuracy: {ensemble_accuracy:.2f}")
