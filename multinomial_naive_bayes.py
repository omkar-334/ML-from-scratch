import string
from collections import defaultdict

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    stop = set(stopwords.words("english"))

    tokens = [char for char in tokens if char not in string.punctuation]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop]
    return tokens


# Sources
# https://medium.com/@johnm.kovachi/implementing-a-multinomial-naive-bayes-classifier-from-scratch-with-python-e70de6a3b92e
# https://web.stanford.edu/~jurafsky/slp3/4.pdf
# https://github.com/akash18tripathi/Multinomial-Naive-Bayes-from-Scratch/blob/main/Multinomial-Naive-Bayes-from-Scratch.ipynb


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.vocab = None
        self.class_prior = {}
        self.word_prob = {}
        self.classes = []

    def build_vocabulary(self, X):
        vocab = set()
        for tokens in X:
            vocab.update(tokens)
        return list(vocab)

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.vocab = self.build_vocabulary(X)

        for cls in self.classes:
            curr_cls = [X[i] for i in range(len(y)) if y[i] == cls]
            self.class_prior[cls] = len(curr_cls) / len(y)

            word_count = defaultdict(lambda: self.alpha)
            total_count = self.alpha * len(self.vocab)

            for text in curr_cls:
                for token in text:
                    word_count[token] += 1
                    total_count += 1

            self.word_prob[cls] = {word: count / total_count for word, count in word_count.items()}

    def calculate_posterior(self, tokens):
        posteriors = []
        for cls in self.classes:
            log_prior = np.log(self.class_prior[cls])
            default_value = self.alpha / (len(self.vocab) * self.alpha)
            log_likelihood = sum(np.log(self.word_prob[cls].get(token, default_value)) for token in tokens)
            posteriors.append(log_prior + log_likelihood)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self.calculate_posterior(x) for x in X]
        return np.array(y_pred)
