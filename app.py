import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, roc_curve)
from wordcloud import WordCloud
import networkx as nx
import re
from collections import Counter

# Load and prepare data
data = pd.read_csv("email 1.csv")
data = data[['Category', 'Message']].dropna()
data = data[data['Category'].isin(['ham', 'spam'])].copy()
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})
data.dropna(inplace=True)

X = data['Message']
y = data['Category']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
y_proba = model.predict_proba(X_test_vec)[:,1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)

# Define prediction function
def classify_email(message):
    if not message.strip():
        return "Please enter a message.", ""
    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][pred]
    label = "Spam" if pred == 1 else "Ham"
    return f"Class: {label}", f"Probability: {proba:.2f}"

# Plot visualizations
def show_performance():
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=axs[0])
    axs[0].set_title("Confusion Matrix")
    axs[0].set_xlabel("Predicted")
    axs[0].set_ylabel("Actual")

    axs[1].plot(fpr, tpr, color='darkorange', label=f"AUC = {auc:.2f}")
    axs[1].plot([0, 1], [0, 1], linestyle='--', color='grey')
    axs[1].set_title("ROC Curve")
    axs[1].set_xlabel("False Positive Rate")
    axs[1].set_ylabel("True Positive Rate")
    axs[1].legend()

    all_text = ' '.join(data['Message'])
    wordcloud = WordCloud(width=400, height=300, background_color='white', stopwords='english').generate(all_text)
    axs[2].imshow(wordcloud, interpolation='bilinear')
    axs[2].axis('off')
    axs[2].set_title("Word Cloud")

    return fig

def show_network():
    text = ' '.join(data['Message']).lower()
    words = re.findall(r'\b\w+\b', text)
    common_words = [word for word, count in Counter(words).most_common(20)]
    edges = [(w1, w2) for i, w1 in enumerate(common_words) for w2 in common_words[i+1:]]

    G = nx.Graph()
    G.add_edges_from(edges)

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1200, edge_color='gray')
    plt.title("Word Network (Top 20 Words)")
    return plt.gcf()

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 📧 Email Spam Classifier")
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Enter Email Text", lines=6, placeholder="Type your message here...")
            classify_btn = gr.Button("Classify")
            class_output = gr.Textbox(label="Prediction")
            prob_output = gr.Textbox(label="Probability")
            classify_btn.click(classify_email, inputs=input_text, outputs=[class_output, prob_output])
        with gr.Column():
            gr.Markdown("### 📊 Model Performance")
            gr.Plot(show_performance)
            gr.Button("Show Word Network").click(fn=show_network, outputs=gr.Plot())

demo.launch()
