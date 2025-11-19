# AI-Powered Phishing URL Detector

### Project Overview
A Machine Learning tool that detects malicious URLs (Phishing, Defacement, Malware) based on text patterns. Unlike traditional blocklists that only stop *known* threats, this AI model utilizes a Random Forest algorithm to predict *unknown* threats by analyzing the structure of the URL itself.

### Objectives
* Build a supervised learning model using **Scikit-Learn**.
* Understand **Feature Extraction** (converting text URLs into data points).
* Demonstrate how AI can augment Blue Team defense by reducing manual review time.

### Tech Stack
* **Language:** Python 3.11
* **Libraries:** `pandas`, `scikit-learn`, `numpy`, `joblib`
* **Algorithm:** Random Forest Classifier (Supervised Learning)
* **Vectorization:** TfidfVectorizer (Custom tokenizer splitting on `/` and `-`)

### Data Analysis & Performance
I utilized the **Malicious URLs Dataset** from Kaggle, which contains ~15,000 entries.

**Preprocessing Strategy:**
The original dataset contained four classes (`benign`, `defacement`, `phishing`, `malware`). I engineered a binary feature mapping these to **Safe** vs **Malicious** to optimize for a defensive "Allow/Block" decision.

**Results:**
* **Accuracy:** ~96% on the test set.
* **Why Random Forest?** I chose Random Forest over Naive Bayes because URL structures often have complex, non-linear dependencies (e.g., a safe domain like `google.com` can become malicious if followed by a specific query string `?login=failed`).

### How it Works
1. **Tokenization:** The script uses a custom function to break a URL down into semantic chunks (tokens) like "google", "com", "secure", "login", splitting by special characters like `/` and `-`.
2. **Vectorization:** It assigns mathematical weight to these words using TF-IDF (Term Frequency-Inverse Document Frequency).
3. **Prediction:** The model calculates the probability of the URL being 'good' or 'bad'.

### How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/ZacDMW/AI-Phishing-Detector.git](https://github.com/ZacDMW/AI-Phishing-Detector.git)

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn joblib

3. Download the Malicious URLs Dataset and rename it to malicious_phish.csv in the project folder. (https://www.kaggle.com/datasets/himadri07/malicious-urls-dataset-15k-rows?resource=download)

4. Run the detector:
   ```bash
   python phish_detect.py

### What I Learned
Pickling Models: I learned how to serialize (save) a trained machine learning model using joblib so it doesn't need to retrain every time it runs.

Troubleshooting: I encountered an issue where Python's pickle module cannot save lambda functions. I resolved this by refactoring the tokenizer into a standalone definition, ensuring the model could be persisted to disk.

Feature Engineering: Standard text splitters fail on URLs. I learned that custom tokenization is required to capture the specific structure of web addresses.

### Future Improvements
Connect to the Google Safe Browsing API for verification.

Implement a web interface using Flask or Streamlit.
