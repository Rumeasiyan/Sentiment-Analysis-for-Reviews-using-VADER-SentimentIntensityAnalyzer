# Sentiment Analysis for Reviews using VADER SentimentIntensityAnalyzer

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load CSV file
input_file = "reviews.csv"
output_file = "output_predictions_vader.csv"

# Read CSV and extract reviews column
data = pd.read_csv(input_file)
reviews = data['Review'].tolist()

# Initialize SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Predict sentiments and store in a list
predictions = []
for review in reviews:
    # Get sentiment scores
    scores = sid.polarity_scores(review)

    # Determine sentiment based on compound score
    if scores['compound'] >= 0.05:
        sentiment = 1  # Positive
    elif scores['compound'] <= -0.05:
        sentiment = -1  # Negative
    else:
        sentiment = 0  # Neutral

    predictions.append(sentiment)

# Add predictions to the dataframe and save to CSV
data['label'] = predictions
data.to_csv(output_file, columns=['Review', 'label'], index=False)

print("Sentiment prediction completed and saved to", output_file)
