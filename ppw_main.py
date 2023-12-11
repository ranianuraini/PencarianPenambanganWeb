import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
from nltk.tokenize import word_tokenize

# Load the preprocessed data
data = pd.read_csv('DataOlah_Antara.csv')
data.dropna(inplace=True)

# Separate features (X) and labels (y)
X = data['artikel_tokens']
y = data['Label']

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Create a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_tfidf, y)

# Create a Sastrawi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_and_tokenize(text):
    text = re.compile('<.*?>').sub('', str(text))
    text = text.lower().strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub('nan', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if not w in stop_words]
    tokens = stemmer.stem(' '.join(tokens)).split(' ')
    return ' '.join(tokens)

def predict(text):
    cleaned_text = clean_and_tokenize(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    prediction = nb_classifier.predict(text_tfidf)[0]
    return prediction

def main():
    st.title('Naive Bayes Text Classification App')

    # Input text from user
    user_input = st.text_area('Enter text for classification:', '')

    # Make prediction when the user clicks the "Predict" button
    if st.button('Predict'):
        if user_input:
            result = predict(user_input)
            st.success(f'Prediction: {result}')
        else:
            st.warning('Please enter text for classification.')

if __name__ == '__main__':
    main()
