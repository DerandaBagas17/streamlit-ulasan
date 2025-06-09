import streamlit as st
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Load the saved model and vectorizer
try:
    nb_model = joblib.load('naive_bayes_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'naive_bayes_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
    st.stop()

# Initialize the stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

# Text preprocessing function (should match the training preprocessing)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return ' '.join(stemmed_tokens)

# Streamlit app title
st.title("Sentiment Analysis for Product Reviews")

# Text area for user input
user_input = st.text_area("Enter a product review:")

if user_input:
    # Preprocess the user input
    cleaned_input = preprocess_text(user_input)

    # Transform the preprocessed input using the loaded vectorizer
    input_vectorized = tfidf_vectorizer.transform([cleaned_input])

    # Convert sparse matrix to dense array for the Naive Bayes model
    input_vectorized_dense = input_vectorized.toarray()

    # Predict the sentiment
    prediction = nb_model.predict(input_vectorized_dense)

    # Display the prediction
    sentiment_label = "Positive" if prediction[0] == 1 else "Negative"
    st.write(f"Predicted Sentiment: {sentiment_label}")
