import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

def make_tokens(f):
    return f.split('/')
# -----------------------

# 1. LOAD DATA
print("Loading dataset...")
try:
    df = pd.read_csv("malicious_phish.csv")
    print(f"Loaded {len(df)} rows of data.")
except FileNotFoundError:
    print("❌ Error: 'malicious_phish.csv' not found.")
    exit()

# 2. PREPROCESS
df['label'] = df['type'].apply(lambda x: 'safe' if x == 'benign' else 'malicious')
X = df['url']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. BUILD PIPELINE
model = make_pipeline(
    TfidfVectorizer(tokenizer=make_tokens), 
    RandomForestClassifier(n_estimators=100, n_jobs=-1)
)

# 4. TRAIN
print("\nTraining the AI...")
model.fit(X_train, y_train)

# 5. EVALUATE
print("Testing accuracy...")
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 6. SAVE
# This will now work because 'make_tokens' is a real function
joblib.dump(model, 'phish_model.pkl')
print("Model saved as 'phish_model.pkl'")

# 7. TEST LOOP
def predict_url(url):
    prediction = model.predict([url])[0]
    probability = model.predict_proba([url]).max() * 100
    emoji = "✅" if prediction == 'safe' else "⛔"
    print(f"\n{emoji} Result: {prediction.upper()} ({probability:.1f}% confidence)")

if __name__ == "__main__":
    print("\n--- PHISH-HUNTER AI READY ---")
    print("Enter a URL to scan (or type 'exit'):")
    while True:
        user_url = input(">> ")
        if user_url.lower() == 'exit':
            break
        predict_url(user_url)