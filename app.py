from flask import Flask, request, jsonify, render_template
import pickle
import re

app = Flask(__name__, template_folder='templates')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the model and vectorizer
with open("spam_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Fetch JSON data
    message = data.get('message', '')

    if not message:
        return jsonify({'prediction': 'No message provided!'})

    processed_message = preprocess_text(message)
    message_tfidf = tfidf_vectorizer.transform([processed_message])
    prediction = model.predict(message_tfidf)[0]

    result = 'Spam' if prediction == 1 else 'Ham'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
