import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

with open('file.txt', 'r') as f:
    conversation = f.read()

words = word_tokenize(conversation)

# Remove stopwords
stop_words = set(stopwords.words('english'))
words = [word for word in words if word.casefold() not in stop_words]

#lemmatization
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word, pos='v') for word in words]

wordcloud = WordCloud(width=800, height=800, background_color='white').generate(' '.join(words))

plt.figure(figsize=(8, 8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
