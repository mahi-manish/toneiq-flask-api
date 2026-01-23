import pandas as pd
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    # Keep important markers for sarcasm/negation
    tokens = [w for w in tokens if w not in stop_words or w in ['not', 'no', 'never', 'but', 'however', 'sure', 'right', 'wow', 'great', 'love']]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# 1. Load the new Sarcasm Dataset
df_sarcasm = pd.read_csv("sarcasm_train_set.csv")
# Map 'negative' to 0, 'neutral' to 1, 'positive' to 2
# Since these are all sarcasm (hiding negative), we map them to 0 (Negative)
df_sarcasm['label'] = 0 

# 2. To prevent the model from becoming "Only Negative", 
# let's add some basic positive and neutral samples
synthetic_data = [
    ("This app is truly amazing and fast.", 2),
    ("I love the new UI design, very clean.", 2),
    ("Best experience so far, highly recommended.", 2),
    ("Working perfectly after the update.", 2),
    ("It is a simple app that does the job.", 1),
    ("The weather is okay today.", 1),
    ("Just a normal day at the office.", 1),
    ("I am using the app to read news.", 1),
    ("Terrible experience, would not recommend.", 0),
    ("The app is slow and full of bugs.", 0),
    ("Disaster, lost all my settings.", 0),
    ("Worst customer support ever.", 0)
]
df_synth = pd.DataFrame(synthetic_data, columns=['text', 'label'])

# Combine
df_final = pd.concat([
    df_sarcasm[['text', 'label']], 
    df_synth
]).reset_index(drop=True)

print(f"Training on {len(df_final)} samples...")

# 3. Clean and Prepare
df_final['clean_text'] = df_final['text'].apply(preprocess_text)

# 4. Create Pipeline: TF-IDF + Logistic Regression
# Balanced weights help detect the minority (sarcastic/negative) patterns
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ('clf', LogisticRegression(C=2.0, class_weight='balanced', max_iter=1000))
])

# 5. Train
pipeline.fit(df_final['clean_text'], df_final['label'])

# 6. Save
MODEL_PATH = "sentiment_pipeline.pkl"
pickle.dump(pipeline, open(MODEL_PATH, "wb"))

print("Retraining Complete! sentiment_pipeline.pkl updated.")

# Test one from the user's list
test_text = "Yeah right, this app is totally perfect."
pred = pipeline.predict([preprocess_text(test_text)])[0]
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
print(f"Test Prediction for '{test_text}': {label_map[pred]}")
