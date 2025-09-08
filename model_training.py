import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import re

print("Starting the model training process...")

# --- Data Preparation ---
# In a real-world scenario, you would use a large, labeled dataset of news articles.
# For this example, we'll create a small, illustrative dataset.
data = {
    'text': [
        "BREAKING: Scientists discover a new planet in our solar system.",
        "The government has just passed a new bill to improve healthcare for all citizens.",
        "A recent study shows that eating vegetables daily can significantly reduce health risks.",
        "The stock market reached an all-time high today after positive economic news.",
        "Local community holds successful charity event for the homeless.",
        "SHOCKING: Aliens have landed on Earth and are secretly replacing our world leaders!",
        "The earth is flat, and all space agencies are lying to us.",
        "A miracle cure for all diseases has been found in a common household plant, doctors hate this!",
        "You won't believe what this celebrity said, it will change your life forever.",
        "Secret government memo proves elections are rigged by a hidden cabal."
    ],
    'label': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Reliable, 1 = Misinformation
}
df = pd.DataFrame(data)

# --- Text Preprocessing Function (should be the same as in app.py) ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Apply preprocessing to the text column
df['text'] = df['text'].apply(preprocess_text)
print("Text data has been preprocessed.")

# --- Feature Extraction (TF-IDF) ---
# This converts the text data into numerical features that the model can understand.
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['text'])
y = df['label']
print("Text data has been vectorized using TF-IDF.")

# --- Model Training ---
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# We use Logistic Regression, a simple and effective model for text classification.
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model training is complete.")

# --- Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# --- Saving the Model and Vectorizer ---
# We use joblib to save the trained model and the vectorizer to disk.
# These files will be loaded by our Flask app (app.py).
joblib.dump(model, 'misinformation_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("\n----------------------------------------------------")
print("Successfully trained and saved the model and vectorizer.")
print("The following files have been created:")
print("- misinformation_model.pkl (The trained classifier)")
print("- tfidf_vectorizer.pkl (The TF-IDF vectorizer)")
print("You can now upload these two files along with app.py to your server.")
print("----------------------------------------------------")
