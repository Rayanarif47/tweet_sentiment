# TweetSentimentScope

A Twitter sentiment analysis application that analyzes the sentiment of tweets or user-provided text using machine learning.

## Overview

TweetSentimentScope is a Streamlit-based web application that performs sentiment analysis on Twitter content. It can analyze both user-provided text and tweets from specific Twitter users, providing real-time sentiment classification (Positive/Negative).

## Features

- **Text Sentiment Analysis**: Analyze sentiment of any input text
- **Twitter User Analysis**: Fetch and analyze tweets from specific Twitter users
- **Real-time Analysis**: Get instant sentiment results
- **Visual Feedback**: Color-coded sentiment display (green for positive, red for negative)
- **Pre-trained Model**: Uses a trained machine learning model for accurate sentiment prediction

## Project Structure

```
TweetSentimentScope/
├── app.py                 # Main Streamlit application
├── model.pkl             # Trained sentiment analysis model
├── vectorizer.pkl        # TF-IDF vectorizer for text processing
├── Tweet_Sentiment_Scope.ipynb  # Jupyter notebook for model development
├── Abbreviations and Slang.csv  # Dataset for text preprocessing
└── training.1600000.processed.noemoticon.csv  # Training dataset
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TweetSentimentScope.git
cd TweetSentimentScope
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Launch the application using the command above
2. Choose between two analysis modes:
   - **Input text**: Enter any text to analyze its sentiment
   - **Get tweets from user**: Enter a Twitter username to analyze their recent tweets
3. Click the respective button to perform the analysis
4. View the sentiment results displayed in color-coded cards

## Technical Details

- Built with Python and Streamlit
- Uses NLTK for text preprocessing
- Implements TF-IDF vectorization for text feature extraction
- Trained on a large dataset of 1.6 million processed tweets
- Uses Nitter for Twitter data scraping

## Requirements

- Python 3.x
- Streamlit
- scikit-learn
- NLTK
- ntscraper
- pickle

## License

[Add your license information here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

[Add your contact information here] 