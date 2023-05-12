import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')

# Example text paragraph
text = "The quick brown fox jumps over the lazy dog. The dog jumps over the fence. The fox and the dog are friends. The fox is quick and brown."

# Remove stopwords
stop_words = set(stopwords.words('english'))
words = word_tokenize(text)
filtered_words = [word for word in words if not word.lower() in stop_words]
print(words)

# Tokenize the paragraph to extract sentences
sentences = sent_tokenize(text)
print(sentences)

# Calculate the word frequency distribution
word_counts = {}
for word in filtered_words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

# Plot the word frequency distribution
plt.bar(word_counts.keys(), word_counts.values())
plt.title('Word Frequency Distribution')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

# Plot the wordcloud
wordcloud = WordCloud(background_color='white').generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()