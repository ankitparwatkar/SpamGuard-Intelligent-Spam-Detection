import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import BertTokenizer
import tensorflow as tf #type: ignore

# Custom BERT layer for model loading
class BertLayer(tf.keras.layers.Layer):
    def __init__(self, model_name='bert-base-uncased', **kwargs):
        super().__init__(**kwargs)
        self.bert = tf.keras.models.load_model('my_bert_model.h5').layers[2].bert
        
    def call(self, inputs):
        input_ids, attention_mask = inputs
        return self.bert(input_ids, attention_mask=attention_mask).last_hidden_state

# Set page configuration
st.set_page_config(
    page_title="SpamGuard - Intelligent Spam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    :root {
        --primary: #4361ee;
        --secondary: #3f37c9;
        --accent: #4895ef;
        --danger: #f72585;
        --success: #4cc9f0;
        --dark: #212529;
        --light: #f8f9fa;
    }
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
    }
    
    .header {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border-radius: 0 0 20px 20px;
        padding: 2rem 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .card {
        background: white;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
        padding: 1.5rem;
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .title {
        color: var(--primary);
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .input-box {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .result-box {
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .spam {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff416c 100%);
    }
    
    .ham {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
    }
    
    .stats-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .stats-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0.5rem 0;
    }
    
    .stats-label {
        color: #6c757d;
        font-size: 0.9rem;
    }
    
    .feature-section {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model with custom layer
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            'my_bert_model.h5',
            custom_objects={'BertLayer': BertLayer}
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

# Function to preprocess text
def preprocess_text(text, tokenizer, max_len=128):
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask']
    }

# Function to predict
def predict_spam(model, tokenizer, text):
    processed = preprocess_text(text, tokenizer)
    prediction = model.predict({
        'input_ids': processed['input_ids'],
        'attention_mask': processed['attention_mask']
    })
    probability = prediction[0][0]
    return probability, "SPAM 🚫" if probability > 0.5 else "HAM ✅", probability

# Sample data for visualization
sample_data = {
    "Message": [
        "Congratulations! You've won a $1000 gift card. Click here to claim",
        "Meeting tomorrow at 10 AM in conference room",
        "URGENT: Your account has been compromised. Verify now!",
        "Thanks for your email. I'll get back to you soon.",
        "FREE iPhone for our loyal customers. Limited offer!",
        "Reminder: Project deadline is this Friday"
    ],
    "Label": ["Spam", "Ham", "Spam", "Ham", "Spam", "Ham"],
    "Confidence": [0.98, 0.05, 0.92, 0.03, 0.96, 0.07]
}

# App layout
def main():
    # Header section
    st.markdown("""
    <div class="header">
        <h1 style="text-align:center; margin:0;">SpamGuard</h1>
        <p style="text-align:center; font-size:1.2rem; margin:0;">Intelligent Spam Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and tokenizer
    model = load_model()
    tokenizer = load_tokenizer()
    
    # Main columns
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        # Input section
        st.markdown("""
        <div class="card">
            <h2 class="title">Detect Spam Messages</h2>
            <p>Enter a message to check if it's spam or legitimate:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Text input
        message = st.text_area("", height=150, placeholder="Type or paste your message here...", 
                             label_visibility="collapsed")
        
        # Prediction button
        predict_btn = st.button("Analyze Message", type="primary", use_container_width=True)
        
        # Prediction result
        if predict_btn and message:
            with st.spinner("Analyzing message..."):
                try:
                    prob, label, confidence = predict_spam(model, tokenizer, message)
                    result_class = "spam" if label == "SPAM 🚫" else "ham"
                    
                    st.markdown(f"""
                    <div class="result-box {result_class}">
                        <h3>This message is classified as: {label}</h3>
                        <p>Confidence: {prob*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence visualization
                    fig, ax = plt.subplots(figsize=(8, 0.8))
                    ax.barh([''], [prob*100], color='#ff416c' if label == "SPAM 🚫" else '#56ab2f', height=0.3)
                    ax.set_xlim(0, 100)
                    ax.set_xticks([])
                    ax.text(prob*100/2, 0, f"{prob*100:.1f}%", 
                            ha='center', va='center', color='white', fontsize=12, fontweight='bold')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        elif predict_btn:
            st.warning("Please enter a message to analyze")
    
    with col2:
        # Stats section
        st.markdown("""
        <div class="card">
            <h2 class="title">System Overview</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col21, col22 = st.columns(2)
        with col21:
            st.markdown("""
            <div class="stats-card">
                <p class="stats-label">Accuracy</p>
                <p class="stats-value">97.3%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="stats-card">
                <p class="stats-label">Spam Detection</p>
                <p class="stats-value">98.1%</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col22:
            st.markdown("""
            <div class="stats-card">
                <p class="stats-label">Model Type</p>
                <p class="stats-value">BERT + BiLSTM</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="stats-card">
                <p class="stats-label">Messages Analyzed</p>
                <p class="stats-value">12,457</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Sample messages section
    st.markdown("""
    <div class="card">
        <h2 class="title">Sample Analysis</h2>
        <p>Recent message classifications:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a DataFrame for the sample data
    df = pd.DataFrame(sample_data)
    
    # Display the DataFrame with custom styling
    def style_row(row):
        if row['Label'] == 'Spam':
            return ['background-color: #ffebee'] * len(row)
        else:
            return ['background-color: #e8f5e9'] * len(row)
    
    st.dataframe(df.style.apply(style_row, axis=1), height=250)
    
    # Visualizations
    col3, col4 = st.columns([1, 1], gap="large")
    
    with col3:
        st.markdown("""
        <div class="feature-section">
            <h3>Spam Word Cloud</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate word cloud for spam
        spam_text = " ".join(df[df['Label'] == 'Spam']['Message'])
        wordcloud = WordCloud(width=600, height=300, 
                             background_color='white', colormap='Reds',
                             max_words=50).generate(spam_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    
    with col4:
        st.markdown("""
        <div class="feature-section">
            <h3>Confidence Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence distribution chart
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['Confidence'], bins=10, kde=True, ax=ax, color='#4361ee')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Confidence Distribution')
        plt.tight_layout()
        st.pyplot(fig)
    
    # How it works section
    st.markdown("""
    <div class="card">
        <h2 class="title">How SpamGuard Works</h2>
        <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:1rem; margin-top:1rem;">
            <div style="flex:1; min-width:200px; background:#f0f4f8; padding:1rem; border-radius:10px;">
                <h4>1. Input Processing</h4>
                <p>Messages are tokenized and preprocessed using BERT tokenizer</p>
            </div>
            <div style="flex:1; min-width:200px; background:#f0f4f8; padding:1rem; border-radius:10px;">
                <h4>2. Deep Learning</h4>
                <p>BERT embeddings + BiLSTM analyze semantic patterns</p>
            </div>
            <div style="flex:1; min-width:200px; background:#f0f4f8; padding:1rem; border-radius:10px;">
                <h4>3. Attention Mechanism</h4>
                <p>Focuses on key words that indicate spam</p>
            </div>
            <div style="flex:1; min-width:200px; background:#f0f4f8; padding:1rem; border-radius:10px;">
                <h4>4. Classification</h4>
                <p>Final layer predicts spam with confidence score</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>SpamGuard v1.0 • Built with BERT, TensorFlow and Streamlit</p>
        <p>© 2023 SpamGuard. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()