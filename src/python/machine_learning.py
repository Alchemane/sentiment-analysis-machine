from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVR
import re, pandas as pd, nltk, pickle
from settings import Settings
nltk.download('stopwords')
nltk.download('punkt')
settings = Settings()

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
        if filepath.endswith('.txt'):
            return self.load_and_preprocess_txt(filepath)
        elif filepath.endswith('.csv'):
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
    
    def load_and_preprocess_txt(self, filepath):
        preprocessed_texts = []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                # Strip whitespace and preprocess each line
                clean_line = line.strip()
                if clean_line:  # Check if the line is not empty
                    preprocessed_text = self.preprocess_text(clean_line)
                    preprocessed_texts.append(preprocessed_text)
        return preprocessed_texts

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
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        model_constructors = {
            'rfr': RandomForestClassifier,
            'svr': SVR
        }
        model_params = settings.model_specific_params[settings.current_model]
        model_constructor = model_constructors.get(settings.current_model)
        # Initialization flags
        self.model = None  # Initialize model attribute to None
        self.vectorizer = None  # Initialize vectorizer attribute to None
        self.load_model()

        if model_constructor:
            self.classifier = model_constructor(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {settings.current_model}")

    def train(self, X, y):
        # Convert X to features using the feature extractor
        features = self.feature_extractor.fit_transform(X)
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=settings.training_params['test_size'], random_state=settings.training_params['random_state'])
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)

        # Will change this to log to Delphi console
        print(confusion_matrix(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))

        self.save_model(settings.model_save_path)

    def save_model(self, model_path=None, vectorizer_path=None):
        # Provide default paths if none are provided
        model_path = model_path if model_path else settings.model_save_path
        vectorizer_path = vectorizer_path if vectorizer_path else settings.vectorizer_save_path

        # Save classifier with pickle
        if self.classifier:
            with open(model_path, 'wb') as f:
                pickle.dump(self.classifier, f)
            print(f"Model saved to {model_path}.")
        
        # Save vectorizer with pickle
        if hasattr(self.feature_extractor, 'vectorizer') and self.feature_extractor.vectorizer:
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.feature_extractor.vectorizer, f)
            print(f"Vectorizer saved to {vectorizer_path}.")

    def load_model(self):
        try:
            with open(settings.model_save_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(settings.vectorizer_save_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            if self.model and self.vectorizer:  # Check if both model and vectorizer are loaded
                print("Model and vectorizer loaded successfully.")
            else:
                print("Failed to load model and vectorizer. Consider training the model.")
        except Exception as e:
            print(f"Loading failed: {e}")
            self.model = None
            self.vectorizer = None
    
    def is_ready(self):
        """Check if the model and vectorizer are ready for predictions."""
        return self.model is not None and self.vectorizer is not None

    def predict(self, text):
        if not self.is_ready():
            raise Exception("Model and vectorizer must be loaded and ready before prediction.")
        # Check explicitly if model has been fitted by looking for a fitted attribute
        if not hasattr(self.model, "estimators_"):
            raise Exception("The model has not been fitted. Please fit the model before prediction.")

        preprocessed_text = self.preprocessor.preprocess_text(text)
        features = self.vectorizer.transform([preprocessed_text])  # Use self.vectorizer directly
        prediction = self.model.predict(features)  # Use self.model directly
        return prediction
    
def analyze_sentiment(text=None, training_path=None, context_path=None):
    if training_path == None:
        training_path = settings.training_path

    if context_path == None and text == None:
        context_path = settings.context_path

    preprocessor = DataPreprocessor()
    feature_extractor = FeatureExtractor()
    classifier = SentimentClassifier(preprocessor, feature_extractor)
    if not classifier.is_ready():
        if training_path:
            print("Training model as it is not ready for predictions.")
            X, y = preprocessor.load_data(training_path)
            classifier.train(X, y)
            if not classifier.is_ready():
                raise Exception("Failed to train and load model.")
        else:
            raise Exception("Model is not ready and no training path provided.")

    if text is None and context_path:
        with open(context_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()
    
    if not text:
        return "No text provided for sentiment analysis."
    
    prediction = classifier.predict(text)
    return 'Positive' if prediction[0] == 1 else 'Negative'
    
if __name__ == "__main__":
    sample_text = None

    sentiment = analyze_sentiment(text="this is a no from me", training_path=None, context_path=None)

    print("overall sentiment: ", sentiment)