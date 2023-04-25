import re
from collections import Counter
import heapq

# Text preprocessing function to remove special characters and digits
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub('[^a-zA-Z\s]', '', text)
    text = re.sub('\d', '', text)
    # Convert to lowercase
    text = text.lower()
    # Split into words
    words = text.split()
    return words

# Extractive summarization function using the most frequent words approach
def summarize(text, num_sentences):
    # Preprocess the text
    words = preprocess_text(text)
    # Count the frequency of each word
    word_counts = Counter(words)
    # Find the most frequent words
    most_frequent_words = heapq.nlargest(num_sentences, word_counts, key=word_counts.get)
    # Find the sentences containing the most frequent words
    sentences = re.split('[.!?]', text)
    summary_sentences = []
    for sentence in sentences:
        for word in most_frequent_words:
            if word in sentence.lower():
                summary_sentences.append(sentence)
                break
        if len(summary_sentences) >= num_sentences:
            break
    # Combine the summary sentences into a summary
    summary = ' '.join(summary_sentences)
    return summary

# Example usage
text = 'The quick brown fox jumps over the lazy dog. The dog jumps over the fence. The fox and the dog are friends. The fox is quick and brown.'
num_sentences = 2

summary = summarize(text, num_sentences)
print(summary)
