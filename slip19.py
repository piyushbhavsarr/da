import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv('movie_review.csv')

sid = SentimentIntensityAnalyzer()
df['sentiment'] = df['text'].apply(lambda x: sid.polarity_scores(x)['compound'])

positive_text = ' '.join(df.loc[df['sentiment'] > 0, 'text'].values)
wordcloud = WordCloud(width=800, height=800, background_color='white').generate(positive_text)

plt.imshow(wordcloud)
plt.axis('off')
plt.show()
