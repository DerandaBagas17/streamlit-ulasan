
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')


# Load the trained model and vectorizer
try:
    nb_model = joblib.load('naive_bayes_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please make sure 'naive_bayes_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
    st.stop()

# Initialize Sastrawi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Get Indonesian stopwords
stop_words = set(stopwords.words('indonesian'))

# Preprocessing function (should match the preprocessing used during training)
def preprocess_text(text):
    text = text.lower()  # Case folding
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabet characters
    tokens = nltk.word_tokenize(text) # Tokenization
    filtered_tokens = [word for word in tokens if word not in stop_words] # Stopword removal
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens] # Stemming
    return ' '.join(stemmed_tokens)


# Streamlit application
st.title("Sentiment Analysis Application")

st.write("Enter a review to predict its sentiment (Positive or Negative).")

# Text input for user review
user_review = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if user_review:
        # Preprocess the user input
        cleaned_review = preprocess_text(user_review)

        # Transform the cleaned review using the loaded TF-IDF vectorizer
        # Ensure the input to transform is a list containing the single review
        review_vector = tfidf_vectorizer.transform([cleaned_review])

        # Make prediction
        prediction = nb_model.predict(review_vector)

        # Display the result
        if prediction[0] == 1:
            st.success("Sentiment: Positive")
        else:
            st.error("Sentiment: Negative")
    else:
        st.warning("Please enter a review to predict the sentiment.")
