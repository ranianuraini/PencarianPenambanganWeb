import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Muat model dan vectorizer yang telah dilatih
# Gantilah 'your_model.pkl' dan 'your_vectorizer.pkl' dengan nama file sesungguhnya
import pickle
with open('your_model.pkl', 'rb') as model_file:
    nb_classifier = pickle.load(model_file)

with open('your_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Aplikasi Streamlit
st.title('Prediksi Kategori Berita')

# Input teks untuk pengguna memasukkan artikel berita
user_input = st.text_area("Masukkan artikel berita Anda:", "")

if st.button('Prediksi'):
    # Pembersihan dan pemrosesan teks
    user_input = cleaning(user_input)
    user_input_tokens = word_tokenize(user_input)
    user_input_tokens = [w for w in user_input_tokens if not w in stop_words]
    user_input_tokens = stemmer.stem(' '.join(user_input_tokens)).split(' ')
    user_input_tokens = ' '.join(user_input_tokens)

    # Transformasi input pengguna menggunakan vectorizer
    user_input_tfidf = vectorizer.transform([user_input_tokens])

    # Melakukan prediksi menggunakan klasifikasi Naive Bayes yang telah dilatih
    prediction = nb_classifier.predict(user_input_tfidf)

    st.subheader('Prediksi:')
    st.write(f"Kategori yang diprediksi untuk artikel yang diberikan adalah: {prediction[0]}")
