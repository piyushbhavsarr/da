import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from heapq import nlargest

# Define the text paragraph
text = "So, keep working. Keep striving. Never give up. Fall down seven times, get up eight. Ease is a greater threat to progress than hardship. Ease is a greater threat to progress than hardship. So, keep moving, keep growing, keep learning. See you at work."

# Preprocess the text to remove special characters and digits
text = re.sub(r'[^\w\s]', '', text)
text = re.sub(r'\d+', '', text)

# Tokenize the text into sentences and words
sentences = sent_tokenize(text)
words = word_tokenize(text)

# Remove stop words from the words list
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]

# Calculate the word frequency for each word
word_frequency = {}
for word in filtered_words:
    if word in word_frequency:
        word_frequency[word] += 1
    else:
        word_frequency[word] = 1

# Calculate the sentence scores based on the word frequency
sentence_scores = {}
for i, sentence in enumerate(sentences):
    for word in word_tokenize(sentence.lower()):
        if word in word_frequency:
            if i in sentence_scores:
                sentence_scores[i] += word_frequency[word]
            else:
                sentence_scores[i] = word_frequency[word]

# Get the top 2 sentences with the highest scores
summary_sentences_indices = nlargest(2, sentence_scores, key=sentence_scores.get)
summary_sentences = [sentences[i] for i in summary_sentences_indices]

# Join the summary sentences into a summary paragraph
summary = " ".join(summary_sentences)

# Print the summary
print(summary)
