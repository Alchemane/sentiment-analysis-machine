from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import re, pandas as pd, joblib, os, nltk
nltk.download('stopwords')
nltk.download('punkt')

from settings import Settings
from command_handler import CommandHandler

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
        sentiment_data = df.iloc[:, 1]  # Assuming sentiment is in the second column
        preprocessed_texts = [self.preprocess_text(text) for text in text_data]
        return preprocessed_texts, sentiment_data

class FeatureExtractor:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def fit_transform(self, documents):
        # Fit the model and transform documents into feature vectors
        return self.vectorizer.fit_transform(documents)

    def transform(self, text):
        # Transform new documents using the fitted model
        return self.vectorizer.transform(text)

class SentimentClassifier:
    def __init__(self, preprocessor, feature_extractor):
        settings = Settings()
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.classifier = RandomForestClassifier(**settings.model_params)

    def train_on_default_data(self):
        filepath = 'path/to/Restaurant_Reviews.tsv'
        preprocessed_texts, sentiments = self.preprocessor.load_and_preprocess_tsv(filepath)
        features = self.feature_extractor.fit_transform(preprocessed_texts)
        X_train, X_test, y_train, y_test = train_test_split(features, sentiments, test_size=0.2, random_state=0)
        self.classifier.fit(X_train, y_train)

    def train(self, X, y):
        # Convert X to features
        features = self.feature_extractor.fit_transform(X)  # Use fit_transform here
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=0)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))

    def save_model(self, model_path):
        joblib.dump(self.classifier, model_path)
        print(f"Model saved to {model_path}.")

    def load_model(self, model_path):
        self.classifier = joblib.load(model_path)
        print(f"Model loaded from {model_path}.")

    def predict(self, text):
        preprocessed_text = self.preprocessor.preprocess_text(text)
        features = self.feature_extractor.vectorizer.transform([preprocessed_text])
        return self.classifier.predict(features)
    
def analyze_sentiment(text, model_path=None, train_new=False, user_data_path=None):
    """
    Entry point for analyzing sentiment of the given text.
    
    Parameters:
    - text: The text to analyze.
    - model_path: Path to a saved model to load. If None, use default training or train_new.
    - train_new: If True, train a new model using user_data_path or default data if user_data_path is None.
    - user_data_path: Path to user-provided training data. Used if train_new is True.
    """
    # Correcting the path for pretrained models
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)
    default_data_path = os.path.join(project_root, '..', 'resources', 'Restaurant_Reviews.tsv')
    pretrained_models_path = os.path.join(project_root, '..', 'resources', 'Pre-trained Models', 'pretrained_model.joblib')

    preprocessor = DataPreprocessor()
    feature_extractor = FeatureExtractor()
    classifier = SentimentClassifier(preprocessor, feature_extractor)
    
    # Decide on the action based on the provided parameters
    if model_path and os.path.exists(model_path):
        # Load the specified pre-trained model
        classifier.load_model(model_path)
    elif os.path.exists(pretrained_models_path) and not train_new:
        # Load the default pre-trained model if it exists and training a new model is not requested
        classifier.load_model(pretrained_models_path)
    else:
        # Train a new model
        if user_data_path:
            # Load and preprocess user-provided data, then train a new model
            X, y = preprocessor.load_data(user_data_path)
        else:
            # Use default training data
            X, y = preprocessor.load_data(default_data_path)
        classifier.train(X, y)
        # Optionally, save the newly trained model
        classifier.save_model(pretrained_models_path)
    
    # Make a prediction
    prediction = classifier.predict(text)
    return prediction[0]  # Assuming binary classification (0 or 1)