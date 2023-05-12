import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from heapq import nlargest

text = "Hey there this any paragraph required for analysis, just for testing and nothing else."

# Preprocess the text to remove any special characters and digits
text = re.sub(r'[^\w\s]', '', text)
text = re.sub(r'\d+', '', text)

# Tokenize the text into sentences and words
sentences = sent_tokenize(text)
words = word_tokenize(text)

# Remove stop words
stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]

# Calculate the frequency of each word
freq = {}
for word in words:
    if word not in freq:
        freq[word] = 1
    else:
        freq[word] += 1

# Calculate the importance of each sentence
sent_importance = {}
for i, sent in enumerate(sentences):
    for word in word_tokenize(sent):
        if word in freq:
            if i not in sent_importance:
                sent_importance[i] = freq[word]
            else:
                sent_importance[i] += freq[word]

# Generate the summary by selecting the most important sentences
summary_length = 1
summary = nlargest(summary_length, sent_importance, key=sent_importance.get)
summary = ' '.join([sentences[i] for i in summary])
print(summary)