# AI-Powered Phishing Link Detector

### Project Overview
A Machine Learning tool that detects malicious URLs based on text patterns. Unlike traditional blocklists that only stop *known* threats, this AI model attempts to predict *unknown* threats by analyzing the structure of the URL itself.

### Objectives
* Build a supervised learning model using **Scikit-Learn**.
* Understand **Feature Extraction** (converting text URLs into data points).
* Demonstrate how AI can augment Blue Team defense.

### Tech Stack
* **Language:** Python
* **Libraries:** `pandas`, `scikit-learn`, `numpy`
* **Algorithm:** Random Forest Classifier (Supervised Learning)

### How it Works
1. **Tokenization:** The script breaks a URL down into small chunks (tokens) like "google", "com", "secure", "login".
2. **Vectorization:** It assigns mathematical weight to these words (e.g., "login" appearing with "update" might have a higher "bad" weight).
3. **Prediction:** The model calculates the probability of the URL being 'good' or 'bad'.

### Future Improvements
* Connect to the **Google Safe Browsing API** for verification.


### Data Analysis
I utilized the **Malicious URLs Dataset** from Kaggle, which contains ~15,000 entries.

**Preprocessing Strategy:**
The original dataset contained four classes: `benign`, `defacement`, `phishing`, and `malware`. To create a robust defensive tool, I engineered a new binary feature:
* **Safe:** mapped from `benign`
* **Malicious:** mapped from `defacement`, `phishing`, and `malware`

**Why Random Forest?**
I chose Random Forest over Naive Bayes because URL structures often have complex, non-linear dependencies (e.g., a safe domain like `google.com` can become malicious if followed by a specific query string `?login=failed`).
