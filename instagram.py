import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download the necessary NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Read the dataset
data = pd.read_csv('top_instagram_influencers.csv')

# Find the top 5 Instagram influencers from India
india_influencers = data[data['Country'] == 'India'].nlargest(5, 'Followers')
print('Top 5 Instagram influencers from India:')
print(india_influencers[['Username', 'Followers']])

# Find the Instagram account having least number of followers
least_followers = data.nsmallest(1, 'Followers')
print('\nInstagram account with the least number of followers:')
print(least_followers[['Username', 'Followers']])

# Remove stopwords and plot the wordcloud of the Category column
category_words = ' '.join(data['Category'].str.lower())
stop_words = set(stopwords.words('english'))
category_words = [word for word in word_tokenize(category_words) if word.casefold() not in stop_words]
lemmatizer = WordNetLemmatizer()
category_words = [lemmatizer.lemmatize(word) for word in category_words]
category_wordcloud = WordCloud(width=800, height=800, background_color='white').generate(' '.join(category_words))
plt.figure(figsize=(8, 8))
plt.imshow(category_wordcloud)
plt.axis('off')
plt.show()

# Group the Instagram accounts category wise
category_groups = data.groupby('Category')
for category, group in category_groups:
    print('\n{} accounts:'.format(category))
    print(group[['Username', 'Followers']])

# Plot the relationship between Followers and Authentic engagement
plt.scatter(data['Followers'], data['Authentic engagement'])
plt.xlabel('Followers')
plt.ylabel('Authentic engagement')
plt.show()
