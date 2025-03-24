import os
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ✅ Improved dataset (Balanced Spam & Ham emails)
emails = [
    "Congratulations! You have won a free lottery. Click here to claim your prize.",  # Spam
    "Hello friend, how are you doing today?",  # Ham
    "Get a discount on our new product! Limited time offer, buy now.",  # Spam
    "Meeting at 3 PM, don't forget the documents.",  # Ham
    "Claim your free vacation now! Click the link to win.",  # Spam
    "Important update: Your account needs verification.",  # Spam
    "Hi John, let's catch up soon!",  # Ham
    "Your loan is approved! Apply now.",  # Spam
    "Reminder: Doctor appointment at 10 AM tomorrow.",  # Ham
    "URGENT: Your password has been compromised. Reset now!",  # Spam
    "See you at the conference next week.",  # Ham
    "This is not a spam message, just checking in!",  # Ham
]

labels = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0]  # 1 = Spam, 0 = Ham (Balanced dataset)

# ✅ Better train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    emails, labels, test_size=0.2, stratify=labels, random_state=42
)

# ✅ Improved preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# ✅ Improved TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    preprocessor=preprocess_text,
    stop_words="english",  # Remove common words
    max_features=500  # Limit features to avoid overfitting
)

# Transform text data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# ✅ Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ✅ Save the trained model & vectorizer
with open("spam_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("✅ Model and vectorizer saved successfully!")

# ✅ Load trained model & vectorizer
if not os.path.exists("spam_model.pkl") or not os.path.exists("tfidf_vectorizer.pkl"):
    print("❌ Error: Model or vectorizer file not found.")
else:
    with open("spam_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        tfidf_vectorizer = pickle.load(vectorizer_file)

# ✅ Function to predict if a message is spam or ham
def predict_spam(message):
    processed_message = preprocess_text(message)
    message_tfidf = tfidf_vectorizer.transform([processed_message])

    if message_tfidf.shape[1] != X_train_tfidf.shape[1]:
        return "❌ Error: Mismatch in feature dimensions. Retrain TF-IDF with consistent features."

    prediction = model.predict(message_tfidf)[0]
    probability = model.predict_proba(message_tfidf)[0][1]

    return f"🟢 Prediction: {'Spam' if prediction == 1 else 'Ham'} (Confidence: {probability:.2f})"

# ✅ Evaluate model performance
y_pred = model.predict(X_test_tfidf)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Real-time testing with user input
while True:
    user_message = input("\nEnter an email message (or type 'exit' to stop): ")
    if user_message.lower() == "exit":
        break
    print(predict_spam(user_message))
