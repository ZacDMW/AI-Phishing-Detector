import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib  # Used to save your model so you don't have to retrain every time

# 1. LOAD THE DATASET
# We use pandas to read the CSV file you downloaded.
print("Loading dataset...")
try:
    df = pd.read_csv("malicious_phish.csv")
    print(f"Loaded {len(df)} rows of data.")
except FileNotFoundError:
    print("Error: 'malicious_phish.csv' not found. Did you download it?")
    exit()

# 2. PREPROCESS DATA
# The dataset has a 'type' column with values like 'benign', 'defacement', etc.
# We want to simplify this into a Binary Classification: Safe vs. Unsafe.
# 'benign' = Safe (0), Everything else = Malicious (1)

df['label'] = df['type'].apply(lambda x: 'safe' if x == 'benign' else 'malicious')

print("\nData Distribution:")
print(df['label'].value_counts())

# 3. SPLIT DATA
# We split the data: 80% for training the AI, 20% for testing it.
X = df['url']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. BUILD THE PIPELINE
# TfidfVectorizer: Converts URL text into numbers (features).
# RandomForest: The decision-making brain.
model = make_pipeline(
    TfidfVectorizer(tokenizer=lambda x: x.split('/')), # Custom tokenizer splits on slashes
    RandomForestClassifier(n_estimators=100, n_jobs=-1)
)

# 5. TRAIN THE MODEL
print("\nTraining the AI (this may take a moment)...")
model.fit(X_train, y_train)

# 6. EVALUATE
print("Testing accuracy...")
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, predictions))

# 7. SAVE THE MODEL (Optional)
# This creates a file you can load later without retraining.
joblib.dump(model, 'phish_model.pkl')
print("💾 Model saved as 'phish_model.pkl'")

# 8. LIVE TEST LOOP
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
