import nltk
from nltk.corpus import gutenberg, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

# 下载必要资源
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

moby_dick_text = gutenberg.raw('melville-moby_dick.txt')

tokens = word_tokenize(moby_dick_text)

stop_words = set(stopwords.words('english'))
filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

pos_tags = nltk.pos_tag(filtered_tokens)

pos_freq = FreqDist(tag for word, tag in pos_tags)
common_pos = pos_freq.most_common(5)

print("Top 5 Parts of Speech and their frequencies:")
for pos, count in common_pos:
    print(f"{pos}: {count}")


lemmatizer = WordNetLemmatizer()
lemmatized_tokens = []
for word, pos in pos_tags[:20]:
    lemmatized = lemmatizer.lemmatize(word, pos=pos[0].lower() if pos[0].lower() in ['a', 'r', 'n', 'v'] else 'n')
    lemmatized_tokens.append(lemmatized)

# Plotting frequency distribution
pos_freq.plot(30, title="POS Frequency Distribution")
plt.show()
