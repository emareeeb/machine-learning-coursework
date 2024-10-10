# before we start with solution, whatever you have given in place of srn in the test file, this code file should also be named same.
import numpy as np
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=RuntimeWarning)


class NaiveBayesClassifier:
    """
    A simple implementation of the Naive Bayes Classifier for text classification.
    """
    
    @staticmethod
    def predict(X, class_probs, word_probs, classes):
        """
        Predicts the classes for the given test data using the trained classifier.

        Args:
            X (numpy.ndarray): The test data matrix where each row represents a document
                              and each column represents the presence (1) or absence (0) of a word.
            class_probs (dict): Prior probabilities of each class obtained from the training phase.
            word_probs (dict): Conditional probabilities of words given each class obtained from training.
            classes (numpy.ndarray): The unique classes in the dataset.

        Returns:
            list: A list of predicted class labels for the test documents.
        """
        predictions = []
        for i in range(X.shape[0]):  # For each document in test data
            doc = X[i]
            log_probs = {}
            
            # Calculate log-probabilities for each class
            for c in classes:
                log_prob_c = np.log(class_probs[c])  # Log of class prior
                log_prob_words = np.sum([np.log(word_probs[c].get(j, 1e-6)) for j in range(len(doc)) if doc[j] == 1])
                log_probs[c] = log_prob_c + log_prob_words
            
            # Predict class with the highest log-probability
            predicted_class = max(log_probs, key=log_probs.get)
            predictions.append(predicted_class)

        return predictions

    @staticmethod
    def preprocess(sentences, categories):
        """
        Preprocess the dataset to remove stop words, and missing or incorrect labels.

        Args:
            sentences (list): List of sentences to be processed.
            categories (list): List of corresponding labels.

        Returns:
            tuple: A tuple of two lists - (cleaned_sentences, cleaned_categories).
        """
        cleaned_sentences = []
        cleaned_categories = []

        for i, category in enumerate(categories):
            if category in ["technology", "food", "entertainment"]:  # Keeping only valid categories
                cleaned_sentences.append(sentences[i])
                cleaned_categories.append(category)

        return cleaned_sentences, cleaned_categories

    @staticmethod
    def fit(X, y):
        """
        Trains the Naive Bayes Classifier using the provided training data.
        
        Args:
            X (numpy.ndarray): The training data matrix where each row represents a document
                              and each column represents the presence (1) or absence (0) of a word.
            y (numpy.ndarray): The corresponding labels for the training documents.

        Returns:
            tuple: A tuple containing two dictionaries:
                - class_probs (dict): Prior probabilities of each class in the training set.
                - word_probs (dict): Conditional probabilities of words given each class.
        """
        class_probs = {}
        word_probs = {}

        # Compute class prior probabilities
        class_counts = Counter(y)
        total_docs = len(y)
        for c in class_counts:
            class_probs[c] = class_counts[c] / total_docs

        # Compute word conditional probabilities P(word|class)
        word_probs = {c: {} for c in class_counts}
        total_words_per_class = {c: 0 for c in class_counts}

        for i in range(X.shape[0]):  # For each document
            c = y[i]
            for j in range(X.shape[1]):  # For each word in the document
                if X[i][j] == 1:
                    word_probs[c][j] = word_probs[c].get(j, 0) + 1
                    total_words_per_class[c] += 1

        for c in word_probs:
            for word, count in word_probs[c].items():
                word_probs[c][word] = (count + 1) / (total_words_per_class[c] + X.shape[1])  # Laplace smoothing

        return class_probs, word_probs
