import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    st.error(f"Error downloading NLTK data: {str(e)}")
    st.stop()

def transform_text(text):
    """
    Preprocess the text using the same steps as training
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
        st.error(f"Error in text transformation: {str(e)}")
        return ""

def load_models():
    """
    Load the trained model and vectorizer
    """
    try:
        model_path = os.path.join('model', 'model.pkl')
        vectorizer_path = os.path.join('model', 'vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            st.error("Model files not found. Please train the model first.")
            st.stop()
            
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

def main():
    # Page config
    st.set_page_config(
        page_title="SMS Spam Detector",
        page_icon="📱",
        layout="centered"
    )
    
    # Title and description
    st.title("📱 SMS Spam Detection")
    st.write("""
    This app uses machine learning to detect spam SMS messages.
    Enter your message below to check if it's spam or not.
    """)
    
    # Load models
    vectorizer, model = load_models()
    
    # User input
    input_sms = st.text_area("Enter the SMS message:", height=100)
    
    if st.button('Predict', key='predict'):
        if not input_sms.strip():
            st.warning("Please enter a message.")
            return
            
        try:
            # Preprocess
            transformed_sms = transform_text(input_sms)
            
            # Vectorize
            vector_input = vectorizer.transform([transformed_sms])
            
            # Predict
            prediction = model.predict(vector_input)[0]
            probability = model.predict_proba(vector_input)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("Results:")
            
            if prediction == 1:
                st.error("🚨 This message is likely SPAM!")
                st.write(f"Confidence: {probability[1]:.2%}")
            else:
                st.success("✅ This message appears to be legitimate (HAM)")
                st.write(f"Confidence: {probability[0]:.2%}")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
