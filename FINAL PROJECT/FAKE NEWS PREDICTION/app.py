import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from PIL import Image

# Load images
fake_news_image = Image.open(r"C:\Users\vejen\Downloads\FAKE NEWS PREDICTION\FAKE NEWS PREDICTION\fake_news_image.jpeg")
true_news_image = Image.open(r"C:\Users\vejen\Downloads\FAKE NEWS PREDICTION\FAKE NEWS PREDICTION\true_news_image.jpg")

# Load the datasets
fake = pd.read_csv(r"C:\Users\vejen\Downloads\FAKE NEWS PREDICTION\FAKE NEWS PREDICTION\data\Fake.csv")
true = pd.read_csv(r"C:\Users\vejen\Downloads\FAKE NEWS PREDICTION\FAKE NEWS PREDICTION\data\True.csv")

# Add a target column to each dataset
fake['target'] = 'fake'
true['target'] = 'true'

# Concatenate the datasets
data = pd.concat([fake, true]).reset_index(drop=True)

# Split the dataset into features (X) and target (y)
X = data['text']  # Features
y = data['target']  # Target variable

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Function for prediction
def predict(selected_model, text):
    model = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('model', models[selected_model])])
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict using the model
    prediction = model.predict([text])
    
    return prediction[0]

# Set page title and favicon
st.set_page_config(page_title="Fake News Detection", page_icon="📰")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .sub-title {
        font-size: 24px;
        color: #424242;
        margin-bottom: 20px;
    }
    
    .prediction {
        font-size: 18px;
        color: #263238;
        margin-top: 20px;
    }
    
    .sidebar {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
    }
    
    .sidebar-title {
        font-size: 24px;
        color: #1E88E5;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar input
st.sidebar.title("Fake News Detection")
text_input = st.sidebar.text_area("Enter the text to predict:", height=200)
selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))

# Main app
if text_input:
    st.title("Fake News Detection App")
    st.markdown('<p class="sub-title">Predictions</p>', unsafe_allow_html=True)
    
    # Display prediction
    prediction = predict(selected_model, text_input)
    st.markdown(f'<p class="prediction">Prediction using {selected_model}: {prediction}</p>', unsafe_allow_html=True)
    
    if prediction == 'fake':
        st.image(fake_news_image, caption='Fake News', use_column_width=True)
    elif prediction == 'true':
        st.image(true_news_image, caption='True News', use_column_width=True)

# Display images
st.sidebar.image(fake_news_image, caption='Fake News', use_column_width=True)
st.sidebar.image(true_news_image, caption='True News', use_column_width=True)
