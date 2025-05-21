import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, roc_curve)
from wordcloud import WordCloud
import networkx as nx
import re
from collections import Counter

# تحميل البيانات ومعالجتها
@st.cache_data
def load_data():
    data = pd.read_csv("email 1.csv")
    data = data[['Category', 'Message']].dropna()
    data = data[data['Category'].isin(['ham', 'spam'])].copy()
    data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})
    data.dropna(inplace=True)
    return data

# تدريب النموذج
@st.cache_resource
def train_model(data):
    X = data['Message']
    y = data['Category']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:,1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'cm': confusion_matrix(y_test, y_pred),
        'fpr': roc_curve(y_test, y_proba)[0],
        'tpr': roc_curve(y_test, y_proba)[1]
    }
    
    return vectorizer, model, metrics

# تصنيف البريد الإلكتروني
def classify_email(message, vectorizer, model):
    if not message.strip():
        return "Please enter a message.", ""
    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][pred]
    label = "Spam" if pred == 1 else "Ham"
    return f"Class: {label}", f"Probability: {proba:.2%}"

# عرض أداء النموذج
def show_performance(metrics, data):  # تم تعديل هذه الدالة
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=axs[0])
    axs[0].set_title("Confusion Matrix")
    axs[0].set_xlabel("Predicted")
    axs[0].set_ylabel("Actual")

    axs[1].plot(metrics['fpr'], metrics['tpr'], color='darkorange', label=f"AUC = {metrics['auc']:.2f}")
    axs[1].plot([0, 1], [0, 1], linestyle='--', color='grey')
    axs[1].set_title("ROC Curve")
    axs[1].set_xlabel("False Positive Rate")
    axs[1].set_ylabel("True Positive Rate")
    axs[1].legend()

    all_text = ' '.join(data['Message'])  # تم تصحيح هذه السطر
    wordcloud = WordCloud(width=400, height=300, background_color='white', stopwords='english').generate(all_text)
    axs[2].imshow(wordcloud, interpolation='bilinear')
    axs[2].axis('off')
    axs[2].set_title("Word Cloud")

    st.pyplot(fig)

# عرض شبكة الكلمات
def show_network(data):
    text = ' '.join(data['Message']).lower()
    words = re.findall(r'\b\w+\b', text)
    common_words = [word for word, count in Counter(words).most_common(20)]
    edges = [(w1, w2) for i, w1 in enumerate(common_words) for w2 in common_words[i+1:]]

    G = nx.Graph()
    G.add_edges_from(edges)

    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1200, edge_color='gray', ax=ax)
    plt.title("Word Network (Top 20 Words)")
    st.pyplot(fig)

# واجهة Streamlit
def main():
    st.set_page_config(page_title="Email Spam Classifier", layout="wide")
    st.title("📧 Email Spam Classifier")
    
    # تحميل البيانات وتدريب النموذج
    data = load_data()
    vectorizer, model, metrics = train_model(data)
    
    # تبويبات التطبيق
    tab1, tab2 = st.tabs(["🔍 Classify Email", "📊 Model Performance"])
    
    with tab1:
        st.header("Classify Email Messages")
        
        col1, col2 = st.columns(2)
        
        with col1:
            message = st.text_area("Enter Email Text", height=200, 
                                 placeholder="Paste your email message here...",
                                 key="email_input")
            
            if st.button("Classify", type="primary"):
                if message.strip():
                    class_label, prob = classify_email(message, vectorizer, model)
                    st.session_state.class_result = (class_label, prob)
                else:
                    st.warning("Please enter a message to classify!")
        
        with col2:
            st.header("Classification Result")
            if 'class_result' in st.session_state:
                class_label, prob = st.session_state.class_result
                if "Spam" in class_label:
                    st.error(class_label)
                    st.metric("Probability", prob)
                else:
                    st.success(class_label)
                    st.metric("Probability", prob)
            else:
                st.info("Enter a message and click Classify to see results")
    
    with tab2:
        st.header("Model Performance Metrics")
        
        st.subheader("Evaluation Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Precision", f"{metrics['precision']:.2%}")
        col3.metric("Recall", f"{metrics['recall']:.2%}")
        col4.metric("F1 Score", f"{metrics['f1']:.2%}")
        
        st.subheader("Visualizations")
        show_performance(metrics, data)  # تم تعديل هذا السطر
        
        if st.button("Show Word Network"):
            show_network(data)

if __name__ == "__main__":
    main()
