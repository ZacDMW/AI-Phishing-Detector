import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

# 1. The Dataset (In real life, load this from a CSV file)
# format: [URL, Label] where 'bad' is phishing and 'good' is safe
data = [
    ("google.com", "good"),
    ("youtube.com", "good"),
    ("my-bank-login-secure-update.com", "bad"), # Suspicious length/words
    ("amazon.com", "good"),
    ("free-money-now.net/login", "bad"),
    ("secure-account-verify.xyz", "bad"),
    ("wikipedia.org", "good"),
    ("paypal-update-security.com", "bad"),
    ("github.com", "good"),
    ("microsoft-support-alert.info", "bad")
]

# 2. Prepare the Data
# We separate the URLs (features) from the labels (targets)
df = pd.DataFrame(data, columns=['url', 'label'])
X = df['url']
y = df['label']

# 3. Build the AI Pipeline
# 'TfidfVectorizer' turns text into numbers the AI can understand
# 'RandomForestClassifier' is the brain that learns the patterns
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())

# 4. Train the AI
print("Training the AI model on sample data...")
model.fit(X, y)
print("Training Complete!")

# 5. The Test Function
def predict_url(url):
    prediction = model.predict([url])
    probability = model.predict_proba([url])
    
    # Get the confidence score
    confidence = np.max(probability) * 100
    
    print(f"\nURL: {url}")
    print(f"Verdict: {prediction[0].upper()}")
    print(f"Confidence: {confidence:.2f}%")

# 6. Interactive Loop
if __name__ == "__main__":
    print("--- AI Phishing Detector ---")
    print("Type 'exit' to quit.")
    while True:
        user_input = input("\nEnter a URL to scan: ")
        if user_input.lower() == 'exit':
            break
        predict_url(user_input)
