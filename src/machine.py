from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re, pandas as pd

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def clean_text(self, text):
        # Remove non-alphabetic characters and lowercase the text
        return re.sub('[^a-zA-Z]', ' ', text).lower()

    def tokenize(self, text):
        # Split the text into individual words/tokens
        return word_tokenize(text)

    def remove_stopwords_and_stem(self, tokens):
        # Remove stopwords and apply stemming to the tokens
        return [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]

    def preprocess_text(self, text):
        # Full preprocessing pipeline combining all steps
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        processed_tokens = self.remove_stopwords_and_stem(tokens)
        return ' '.join(processed_tokens)

    def load_data(self, filepath):
        # Load and preprocess data from various file types
        if filepath.endswith('.csv'):
            return self.load_and_preprocess_csv(filepath)
        elif filepath.endswith('.tsv'):
            return self.load_and_preprocess_tsv(filepath)
        elif filepath.endswith('.txt'):
            return self.load_and_preprocess_txt(filepath)
        else:
            raise ValueError("Unsupported file type") # Maybe I'll add more?

    def load_and_preprocess_csv(self, filepath):
        # Load data from a CSV file and preprocess it
        df = pd.read_csv(filepath, encoding='utf-8')
        text_data = df.iloc[:, 0]  # Assuming text is in the first column
        return [self.preprocess_text(text) for text in text_data]

    def load_and_preprocess_tsv(self, filepath):
        # Load data from a TSV file and preprocess it
        df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
        text_data = df.iloc[:, 0]  # Assuming text is in the first column
        return [self.preprocess_text(text) for text in text_data]

    def load_and_preprocess_txt(self, filepath):
        # Load data from a plain text file and preprocess it
        with open(filepath, 'r', encoding='utf-8') as file:
            text_data = file.read()
        return self.preprocess_text(text_data)

"""
preprocessor = DataPreprocessor()
preprocessed_text = preprocessor.load_data('example.txt')
print(preprocessed_text)
"""

class FeatureExtractor:
    def __init__(self, documents):
        self.documents = documents

    def extract_features(self, text):
        pass

class SentimentClassifier:
    def __init__(self, preprocessor, feature_extractor):
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.classifier = None

    def train(self, training_data):
        pass

    def predict(self, text):
        pass