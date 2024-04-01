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

    def load_training(self, filepath):
        if filepath.endswith('.csv'):
            return self.preprocess_csv(filepath)
        elif filepath.endswith('.tsv'):
            return self.preprocess_tsv(filepath)
        else:
            return {"Unsupported file type for training data. Please use .csv or .tsv files."}

    def preprocess_csv(self, filepath):
        # Load data from a CSV file and preprocess it
        df = pd.read_csv(filepath, encoding='utf-8')
        text_data = df.iloc[:, 0]  # Assuming text is in the first column
        return [self.preprocess_text(text) for text in text_data]

    def preprocess_tsv(self, filepath):
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
            return {"status": "error", "message": "Unsupported model type: {settings.current_model}"}

    def train(self, X, y):
        features = self.feature_extractor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=settings.training_params['test_size'], random_state=settings.training_params['random_state'])
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        self.save_model(settings.model_save_path)
        return {"status": "success", "confusion_matrix": cm.tolist(), "accuracy": acc}

    def save_model(self, model_path=None, vectorizer_path=None):
        response = {"status": "", "message": ""}
        try:
            model_path = model_path if model_path else settings.model_save_path
            vectorizer_path = vectorizer_path if vectorizer_path else settings.vectorizer_save_path

            with open(model_path, 'wb') as model_file:
                pickle.dump(self.classifier, model_file)
            with open(vectorizer_path, 'wb') as vectorizer_file:
                pickle.dump(self.feature_extractor.vectorizer, vectorizer_file)

            response["status"] = "success"
            response["message"] = "Model and vectorizer saved successfully."
        except Exception as e:
            response["status"] = "error"
            response["message"] = f"Failed to save model and vectorizer: {e}"
        return response

    def load_model(self):
        response = {"status": "", "message": ""}
        try:
            with open(settings.model_save_path, 'rb') as model_file:
                self.model = pickle.load(model_file)
            with open(settings.vectorizer_save_path, 'rb') as vectorizer_file:
                self.vectorizer = pickle.load(vectorizer_file)

            if self.model and self.vectorizer:  # Check if both model and vectorizer are loaded
                response["status"] = "success"
                response["message"] = "Model and vectorizer loaded successfully."
            else:
                response["status"] = "error"
                response["message"] = "Failed to load model and vectorizer. Consider training the model."
        except Exception as e:
            response["status"] = "error"
            response["message"] = f"Loading failed: {e}"
            self.model = None
            self.vectorizer = None
        return response
    
    def is_ready(self):
        """Check if the model and vectorizer are ready for predictions."""
        return self.model is not None and self.vectorizer is not None

    def predict(self, text):
        if not self.is_ready():
            return {"status": "error", "message": "Model and vectorizer must be loaded and ready before prediction."}

        try:
            preprocessed_text = self.preprocessor.preprocess_text(text)
            features = self.vectorizer.transform([preprocessed_text])
            prediction = self.model.predict(features)
            sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
            return {"status": "success", "sentiment": sentiment}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
def analyze_sentiment(text=None, training_path=None, context_path=None):
    response = {"status": "", "message": "", "sentiment": ""}
    try:
        training_path = training_path or settings.training_path
        context_path = context_path or settings.context_path
        preprocessor = DataPreprocessor()
        feature_extractor = FeatureExtractor()
        classifier = SentimentClassifier(preprocessor, feature_extractor)
        if not classifier.is_ready():
            if training_path:
                X, y = preprocessor.load_training(training_path)
                train_response = classifier.train(X, y)
                if not train_response["status"] == "success":
                    return train_response
            else:
                return {"status": "error", "message": "Model is not ready and no training path provided."}

        if text is None and context_path:
            with open(context_path, 'r', encoding='utf-8') as file:
                text = file.read().strip()
        
        if not text:
            return {"status": "error", "message": "No text provided for sentiment analysis."}
        
        prediction_response = classifier.predict(text)
        if prediction_response["status"] == "success":
            response.update({"status": "success", "sentiment": prediction_response["sentiment"]})
        else:
            return prediction_response
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return response