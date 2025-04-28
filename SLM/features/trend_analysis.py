import re
import gensim
from gensim import corpora
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def perform_trend_analysis(text_corpus):
    try:
        # Ensure text_corpus is a string
        if not isinstance(text_corpus, str):
            raise TypeError("Expected text_corpus to be a string, but got a different type.")

        print("Cleaning and processing text corpus for trend analysis...")

        # Text Cleaning
        text_corpus = text_corpus.lower()
        text_corpus = re.sub(r'\W+', ' ', text_corpus)  # Remove special characters
        words = word_tokenize(text_corpus)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words and len(word) > 2]  # Remove stopwords and short words

        # Create Bag of Words representation
        word_counts = Counter(words)

        print("\nTop 10 Most Frequent Terms:")
        for word, freq in word_counts.most_common(10):
            print(f"{word}: {freq}")

        # Topic Modeling using LDA
        dictionary = corpora.Dictionary([words])
        corpus = [dictionary.doc2bow(words)]
        lda_model = gensim.models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

        print("\nIdentified Topics:")
        for idx, topic in lda_model.show_topics(num_topics=3, num_words=5, formatted=False):
            keywords = [word for word, prob in topic]
            print(f"Topic {idx + 1}: {', '.join(keywords)}")

        print("\nTrend Analysis Complete.\n")
    
    except Exception as e:
        print(f"Error during Trend Analysis: {e}")