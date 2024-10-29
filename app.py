from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask application
app = Flask(__name__)

# Load model and tokenizer
model = load_model('LSTM_Model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [
        lemmatizer.lemmatize(token.replace("n't", "not").replace("$", "dollar"))
        for token in tokens if token.isalnum() and token not in english_stopwords
    ]
    return ' '.join(filtered_tokens)

# Define Flask route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    transformed_text = transform_text(text)

    # Convert text to sequence
    sequences = tokenizer.texts_to_sequences([transformed_text])
    padded_sequence = pad_sequences(sequences, maxlen=100)  # Adjust maxlen if needed

    # Predict using the model
    prediction = model.predict(padded_sequence)
    predicted_label = 'AI-Generated' if prediction[0][0] > 0.9 else 'Human-Generated'

    return render_template('index.html', prediction_text=f'Text is predicted as: {predicted_label}')

if __name__ == "__main__":
    app.run(debug=True)
