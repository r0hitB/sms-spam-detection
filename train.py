import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    """
    Preprocess the text data:
    1. Convert to lowercase
    2. Tokenize
    3. Remove non-alphanumeric characters
    4. Remove stopwords and punctuation
    5. Apply stemming
    """
    try:
        # Convert to string if not already
        text = str(text).lower()
        # Tokenize
        text = nltk.word_tokenize(text)
        
        # Remove non-alphanumeric characters
        y = [i for i in text if i.isalnum()]
        
        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        y = [i for i in y if i not in stop_words and i not in string.punctuation]
        
        # Apply stemming
        ps = PorterStemmer()
        y = [ps.stem(i) for i in y]
        
        return " ".join(y)
    except Exception as e:
        print(f"Error in transform_text: {str(e)}")
        return ""

def train_spam_model():
    try:
        # Load the dataset
        print("Loading dataset...")
        # Load the dataset with utf-8-sig encoding to handle BOM
        df = pd.read_csv('sms-spam.csv', encoding='utf-8-sig')
        
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        print(f"Dataset shape: {df.shape}")
        print(f"Number of spam messages: {len(df[df['v1'] == 'spam'])}")
        print(f"Number of ham messages: {len(df[df['v1'] == 'ham'])}")
        
        # Convert labels to numeric
        df['label'] = df['v1'].map({'spam': 1, 'ham': 0})
        
        # Preprocess the text messages
        print("\nPreprocessing text...")
        df['transformed_text'] = df['v2'].apply(transform_text)
        
        # Create TF-IDF features
        print("Creating TF-IDF features...")
        tfidf = TfidfVectorizer(max_features=3000)
        X = tfidf.fit_transform(df['transformed_text'])
        y = df['label']
        
        # Split the data
        print("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        print("Training model...")
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        # Evaluate the model
        print("\nModel Evaluation:")
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the vectorizer and model
        print("\nSaving model and vectorizer...")
        with open('model/vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf, f)
        with open('model/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print("\nTraining completed successfully!")
        print(f"Model accuracy on test set: {model.score(X_test, y_test):.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error in train_spam_model: {str(e)}")
        return False

if __name__ == "__main__":
    train_spam_model()
