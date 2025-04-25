import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from dotenv import load_dotenv
import json
from fpdf import FPDF
import io

# Get the absolute path to the .env file
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')

# Load environment variables with explicit path
if not load_dotenv(env_path):
    print(f"Warning: .env file not found at {env_path}")

# Debug: Print current working directory and check if .env exists
print(f"Current working directory: {os.getcwd()}")
print(f"Looking for .env file at: {env_path}")
print(f".env file exists: {os.path.exists(env_path)}")

# Try to get API key with more detailed error handling
api_key = os.getenv('OPEN_AI_API_KEY')
if api_key is None:
    print("Error: OPEN_AI_API_KEY not found in environment variables")
    print("Please check your .env file format. It should contain:")
    print("OPEN_AI_API_KEY=sk-your-actual-api-key-here")
    print("Make sure there are no spaces around the = sign and no quotes around the key")
    
    # Try to read the .env file directly to debug
    try:
        with open(env_path, 'r') as f:
            print("\nContents of .env file:")
            print(f.read())
    except Exception as e:
        print(f"\nError reading .env file: {e}")
    
    exit(1)

print(f"OPEN_AI_API_KEY found: {api_key[:7]}...")  # Only print first 7 chars for security

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def load_model_and_data():
    """Load the trained model, vectorizer, and dataset"""
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    
    # Load the dataset with the correct path
    dataset_path = r"G:\External Work\University Work\Semester 8\FYP\Tweet Sentiment Scope\tweet_sentiment_scope\Dataset\training.1600000.processed.noemoticon.csv"
    dataset = pd.read_csv(dataset_path, encoding='ISO-8859-1')
    col_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    dataset.columns = col_names
    
    return model, vectorizer, dataset

def perform_eda(dataset):
    """Perform Exploratory Data Analysis on the dataset"""
    eda_results = {
        "dataset_shape": dataset.shape,
        "null_values": dataset.isnull().sum().to_dict(),
        "sentiment_distribution": dataset['target'].value_counts().to_dict(),
        "average_text_length": dataset['text'].str.len().mean(),
        "unique_users": dataset['user'].nunique(),
        "date_range": {
            "start": dataset['date'].min(),
            "end": dataset['date'].max()
        }
    }
    return eda_results

def generate_visualizations(dataset):
    """Generate and save visualizations"""
    # Create a directory for visualizations if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 1. Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=dataset, x='target')
    plt.title('Sentiment Distribution')
    plt.savefig('visualizations/sentiment_distribution.png')
    plt.close()
    
    # 2. Text Length Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=dataset, x=dataset['text'].str.len(), bins=50)
    plt.title('Text Length Distribution')
    plt.savefig('visualizations/text_length_distribution.png')
    plt.close()
    
    # 3. Tweets per User Distribution
    user_tweet_counts = dataset['user'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.histplot(data=user_tweet_counts, bins=50)
    plt.title('Tweets per User Distribution')
    plt.savefig('visualizations/tweets_per_user.png')
    plt.close()

def get_llm_analysis(eda_results, model_performance):
    """Get analysis and recommendations from OpenAI"""
    prompt = f"""
    As a data science expert, analyze the following sentiment analysis model and dataset information:
    
    Dataset Statistics:
    - Total samples: {eda_results['dataset_shape']}
    - Sentiment distribution: {eda_results['sentiment_distribution']}
    - Average text length: {eda_results['average_text_length']:.2f} characters
    - Number of unique users: {eda_results['unique_users']}
    - Date range: {eda_results['date_range']}
    
    Model Performance:
    - Accuracy: {model_performance['accuracy']:.4f}
    - Classification Report: {model_performance['classification_report']}
    
    Please provide:
    1. A detailed analysis of the dataset characteristics
    2. Insights about the model's performance
    3. Specific recommendations for improving the model's accuracy
    4. Potential areas for feature engineering
    5. Suggestions for handling class imbalance if present
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a data science expert specializing in NLP and sentiment analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2000
    )
    
    return response.choices[0].message.content

def create_pdf_report(eda_results, model_performance, llm_analysis):
    """Create a PDF report with all the analysis"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Sentiment Analysis Model EDA Report", ln=True, align='C')
    pdf.ln(10)
    
    # Dataset Statistics
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Dataset Statistics", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Total samples: {eda_results['dataset_shape']}")
    pdf.multi_cell(0, 10, f"Sentiment distribution: {eda_results['sentiment_distribution']}")
    pdf.multi_cell(0, 10, f"Average text length: {eda_results['average_text_length']:.2f} characters")
    pdf.multi_cell(0, 10, f"Number of unique users: {eda_results['unique_users']}")
    pdf.ln(5)
    
    # Model Performance
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Model Performance", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Accuracy: {model_performance['accuracy']:.4f}")
    pdf.multi_cell(0, 10, f"Classification Report:\n{model_performance['classification_report']}")
    pdf.ln(5)
    
    # LLM Analysis
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Expert Analysis and Recommendations", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, llm_analysis)
    
    # Add visualizations
    for viz in ['sentiment_distribution.png', 'text_length_distribution.png', 'tweets_per_user.png']:
        if os.path.exists(f'visualizations/{viz}'):
            pdf.add_page()
            pdf.image(f'visualizations/{viz}', x=10, y=10, w=190)
    
    # Save the PDF
    pdf.output('model_analysis_report.pdf')

def main():
    # Load model and data
    model, vectorizer, dataset = load_model_and_data()
    
    # Perform EDA
    eda_results = perform_eda(dataset)
    
    # Generate visualizations
    generate_visualizations(dataset)
    
    # Calculate model performance
    X = dataset['text']
    y = dataset['target']
    y_pred = model.predict(vectorizer.transform(X))
    
    model_performance = {
        'accuracy': accuracy_score(y, y_pred),
        'classification_report': classification_report(y, y_pred)
    }
    
    # Get LLM analysis
    llm_analysis = get_llm_analysis(eda_results, model_performance)
    
    # Create PDF report
    create_pdf_report(eda_results, model_performance, llm_analysis)
    
    print("Analysis complete! Check 'model_analysis_report.pdf' for the detailed report.")

if __name__ == "__main__":
    main() 