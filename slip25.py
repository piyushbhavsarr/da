import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

# Read the dataset
data = pd.read_csv('covid_2021_1.csv')
# Perform data cleaning operations
data = data.dropna(subset=['comment'])
data['comment'] = data['comment'].str.replace(r'[^\w\s]', '')

# Tokenize the comments into words
data['words'] = data['comment'].apply(word_tokenize)

# Perform sentiment analysis
sia = SentimentIntensityAnalyzer()
data['sentiment'] = data['comment'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Find the percentage of positive, negative, and neutral comments
positive = (data['sentiment'] > 0).sum() / len(data) * 100
negative = (data['sentiment'] < 0).sum() / len(data) * 100
neutral = (data['sentiment'] == 0).sum() / len(data) * 100

print(f'Positive: {positive:.2f}%')
print(f'Negative: {negative:.2f}%')
print(f'Neutral: {neutral:.2f}%')