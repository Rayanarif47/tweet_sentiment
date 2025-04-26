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
from datetime import datetime

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
    As a data science expert specializing in NLP and sentiment analysis, please provide a comprehensive analysis of the following model and dataset. Structure your response in the following format:

    1. DATASET ANALYSIS
    - Dataset Characteristics: [Analyze the dataset characteristics and distribution]
    - Data Quality and Biases: [Comment on data quality and potential biases]
    - Notable Patterns: [Identify any notable patterns in the data]

    2. MODEL PERFORMANCE EVALUATION
    - Accuracy: [Evaluate the model's accuracy and overall performance]
    - Precision, Recall, F1-score: [Analyze precision, recall, and F1 scores for each class]
    - Strengths and Weaknesses: [Identify strengths and weaknesses in the model's predictions]

    3. TECHNICAL RECOMMENDATIONS
    - Model Improvements: [Suggest specific model improvements]
    - Feature Engineering: [Recommend feature engineering techniques]
    - Data Preprocessing: [Propose data preprocessing enhancements]

    4. BUSINESS IMPLICATIONS
    - Practical Implications: [Discuss the practical implications]
    - Use Cases: [Suggest potential use cases and applications]
    - Limitations/Risks: [Identify any limitations or risks]

    5. ACTION ITEMS
    1. Address Class Imbalance
       Impact: [Describe impact]
       Effort: [Describe effort level]
       Metric: [Specify metric to track]
    2. Enhance Feature Engineering
       Impact: [Describe impact]
       Effort: [Describe effort level]
       Metric: [Specify metric to track]
    3. Optimize Model Architecture
       Impact: [Describe impact]
       Effort: [Describe effort level]
       Metric: [Specify metric to track]
    4. Improve Data Preprocessing
       Impact: [Describe impact]
       Effort: [Describe effort level]
       Metric: [Specify metric to track]
    5. Explore Additional Features
       Impact: [Describe impact]
       Effort: [Describe effort level]
       Metric: [Specify metric to track]

    Dataset Statistics:
    - Total samples: {eda_results['dataset_shape']}
    - Sentiment distribution: {eda_results['sentiment_distribution']}
    - Average text length: {eda_results['average_text_length']:.2f} characters
    - Number of unique users: {eda_results['unique_users']}
    - Date range: {eda_results['date_range']}
    
    Model Performance:
    - Accuracy: {model_performance['accuracy']:.4f}
    - Classification Report: {model_performance['classification_report']}
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a data science expert specializing in NLP and sentiment analysis. Provide detailed, structured, and actionable insights. Ensure all sections are complete with no missing information."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2000
    )
    
    return response.choices[0].message.content

def create_pdf_report(eda_results, model_performance, llm_analysis):
    """Create a professional PDF report with comprehensive analysis"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title Page
    pdf.add_page()
    pdf.set_font("Arial", "B", 28)
    pdf.cell(0, 60, "", ln=True)  # Spacing
    pdf.cell(0, 20, "SENTIMENT ANALYSIS MODEL REPORT", ln=True, align='C')
    
    # Subtitle
    pdf.set_font("Arial", "", 14)
    pdf.cell(0, 10, "Comprehensive Analysis and Recommendations", ln=True, align='C')
    
    # Date
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%B %d, %Y %H:%M')}", ln=True, align='C')
    
    # Table of Contents
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 20, "Table of Contents", ln=True)
    pdf.ln(10)
    
    # TOC Entries
    pdf.set_font("Arial", "B", 12)
    sections = [
        "1. Executive Summary",
        "2. Dataset Overview",
        "3. Exploratory Data Analysis",
        "4. Model Performance Analysis",
        "5. Expert Recommendations",
        "6. Visualizations"
    ]
    
    for section in sections:
        pdf.cell(0, 10, section, ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 5, "Page " + str(pdf.page_no() + 1), ln=True)
        pdf.set_font("Arial", "B", 12)
        pdf.ln(5)
    
    # Executive Summary
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "1. Executive Summary", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, "This report presents a comprehensive analysis of our sentiment analysis model, including dataset characteristics, model performance metrics, and expert recommendations for improvement.")
    pdf.ln(10)
    
    # Dataset Overview
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "2. Dataset Overview", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "2.1 Dataset Statistics", ln=True)
    pdf.ln(5)
    
    # Create a table for dataset statistics
    pdf.set_font("Arial", "B", 12)
    col_width = 95
    row_height = 10
    
    # Table header
    pdf.cell(col_width, row_height, "Metric", 1, 0, 'C')
    pdf.cell(col_width, row_height, "Value", 1, 1, 'C')
    
    # Table content
    pdf.set_font("Arial", "", 12)
    stats = [
        ("Total samples", str(eda_results['dataset_shape'])),
        ("Number of unique users", str(eda_results['unique_users'])),
        ("Date range", f"{eda_results['date_range']['start']} to {eda_results['date_range']['end']}")
    ]
    
    for stat in stats:
        pdf.cell(col_width, row_height, stat[0], 1, 0)
        pdf.cell(col_width, row_height, stat[1], 1, 1)
    
    # Exploratory Data Analysis
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "3. Exploratory Data Analysis", ln=True)
    pdf.ln(5)
    
    # Sentiment Distribution
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "3.1 Sentiment Distribution", ln=True)
    pdf.ln(5)
    
    # Create a table for sentiment distribution
    pdf.set_font("Arial", "B", 12)
    pdf.cell(col_width, row_height, "Sentiment", 1, 0, 'C')
    pdf.cell(col_width, row_height, "Count", 1, 1, 'C')
    
    pdf.set_font("Arial", "", 12)
    for sentiment, count in eda_results['sentiment_distribution'].items():
        pdf.cell(col_width, row_height, f"Sentiment {sentiment}", 1, 0)
        pdf.cell(col_width, row_height, str(count), 1, 1)
    
    # Text Characteristics
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "3.2 Text Characteristics", ln=True)
    pdf.ln(5)
    
    # Create a table for text characteristics
    pdf.set_font("Arial", "B", 12)
    pdf.cell(col_width, row_height, "Characteristic", 1, 0, 'C')
    pdf.cell(col_width, row_height, "Value", 1, 1, 'C')
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(col_width, row_height, "Average text length", 1, 0)
    pdf.cell(col_width, row_height, f"{eda_results['average_text_length']:.2f} characters", 1, 1)
    
    # Model Performance Analysis
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "4. Model Performance Analysis", ln=True)
    pdf.ln(5)
    
    # Overall Performance
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "4.1 Overall Performance", ln=True)
    pdf.ln(5)
    
    # Create a table for model performance
    pdf.set_font("Arial", "B", 12)
    pdf.cell(col_width, row_height, "Metric", 1, 0, 'C')
    pdf.cell(col_width, row_height, "Value", 1, 1, 'C')
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(col_width, row_height, "Model Accuracy", 1, 0)
    pdf.cell(col_width, row_height, f"{model_performance['accuracy']:.4f}", 1, 1)
    
    # Classification Report
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "4.2 Detailed Classification Report", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, model_performance['classification_report'])
    
    # Expert Recommendations
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "5. Expert Recommendations", ln=True)
    pdf.ln(5)
    
    # Split the analysis into sections
    sections = llm_analysis.split("###")
    for section in sections:
        if not section.strip():
            continue
            
        # Split into title and content
        parts = section.strip().split("-", 1)
        if len(parts) < 2:
            continue
            
        title = parts[0].strip()
        content = parts[1].strip()
        
        # Print section title
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.ln(5)
        
        # Split content into bullet points
        bullet_points = content.split("-")
        for point in bullet_points:
            if not point.strip():
                continue
                
            # Clean up the point text by removing asterisks
            point = point.replace('*', '').strip()
            
            # Split into header and description
            point_parts = point.split(":", 1)
            if len(point_parts) == 2:
                header = point_parts[0].strip()
                description = point_parts[1].strip()
                
                # Print header in bold
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, header + ":", ln=True)
                
                # Print description
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(0, 8, description)
                pdf.ln(5)
            else:
                # If no colon, check if it's a numbered item
                if point.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                    # Print numbered item
                    pdf.set_font("Arial", "B", 12)
                    pdf.multi_cell(0, 8, point.strip())
                else:
                    # Print regular point
                    pdf.set_font("Arial", "", 12)
                    pdf.multi_cell(0, 8, point.strip())
                pdf.ln(5)
        
        pdf.ln(5)
    
    # Visualizations
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "6. Visualizations", ln=True)
    pdf.ln(5)
    
    viz_files = {
        'sentiment_distribution.png': 'Sentiment Distribution Analysis',
        'text_length_distribution.png': 'Text Length Distribution',
        'tweets_per_user.png': 'User Activity Distribution'
    }
    
    for viz_file, title in viz_files.items():
        if os.path.exists(f'visualizations/{viz_file}'):
            pdf.ln(5)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, title, ln=True)
            pdf.ln(5)
            pdf.image(f'visualizations/{viz_file}', x=10, y=None, w=190)
            pdf.ln(10)
    
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
    
    # Generate classification report with zero_division parameter
    class_report = classification_report(y, y_pred, zero_division=0)
    
    model_performance = {
        'accuracy': accuracy_score(y, y_pred),
        'classification_report': class_report
    }
    
    # Get LLM analysis
    llm_analysis = get_llm_analysis(eda_results, model_performance)
    
    # Create PDF report
    create_pdf_report(eda_results, model_performance, llm_analysis)
    
    print("Analysis complete! Check 'model_analysis_report.pdf' for the detailed report.")

if __name__ == "__main__":
    main() 