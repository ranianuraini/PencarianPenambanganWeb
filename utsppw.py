import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re

@st.cache(allow_output_mutation=True)
def load_data(file):
    data = pd.read_csv(file)
    return data

file = st.file_uploader("Choose a CSV file", type=["csv"])
if file is not None:
    data = load_data(file)
    
def preprocess_text(text):
    text = re.sub(r'\W+|\d+', ' ', text)
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)
    return text

data['processed_text'] = data['text'].apply(preprocess_text)

def vectorize_text(text, vectorizer):
    return vectorizer.fit_transform(text)

vectorizer = TfidfVectorizer()
X = vectorize_text(data['processed_text'], vectorizer)

le = LabelEncoder()
y = le.fit_transform(data['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model

def evaluate_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

python
def run_model(X_train, y_train, X_test, y_test, model):
    model = train_model(X_train, y_train, model)
    accuracy, report = evaluate_model(X_test, y_test, model)
    return accuracy, report

st.title("Text Classification App")

model_selection = st.selectbox("Select a model", ["Logistic Regression", "SVM", "Decision Tree", "Random Forest", "Naive Bayes"])

if st.button("Run Model"):
    if model_selection == "Logistic Regression":
        model = LogisticRegression()
    elif model_selection == "SVM":
        model = SVC()
    elif model_selection == "Decision Tree":
        model = Decision