from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
import re, pandas as pd, nltk, pickle
from settings import Settings
nltk.download('stopwords')
nltk.download('punkt')

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
    def __init__(self, preprocessor, feature_extraction_params=None):
        self.preprocessor = preprocessor
        if feature_extraction_params is None:
            feature_extraction_params = {}
        # Convert list to tuple for 'ngram_range'
        if 'ngram_range' in feature_extraction_params:
            ngram_range = feature_extraction_params['ngram_range']
            if isinstance(ngram_range, list):
                feature_extraction_params['ngram_range'] = tuple(ngram_range)
        self.vectorizer = CountVectorizer(**feature_extraction_params)

    def fit_transform(self, documents):
        # Fit the model and transform documents into feature vectors
        return self.vectorizer.fit_transform(documents)

    def transform(self, text):
        # Transform new documents using the fitted model
        processed_text = self.preprocessor.preprocess_text(text)
        return self.vectorizer.transform([processed_text])

class SentimentClassifier:
    def __init__(self, preprocessor, feature_extractor, model_params=None):
        self.settings = Settings()
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        model_type = self.settings.current_model
        if model_params is None:
            model_params = {}
        model_constructors = {
            'rfr': RandomForestClassifier,
            'svc': SVC
        }
        
        # Initialization flags
        self.model = None
        self.vectorizer = None
        self.load_model()

        if model_type in model_constructors:
            self.classifier = model_constructors[model_type](**self.settings.model_specific_params.get(model_type, {}))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, X, y):
        features = self.feature_extractor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=self.settings.training_params['test_size'], random_state=self.settings.training_params['random_state'])
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        self.save_model(self.settings.model_save_path)
        return {"status": "success", "confusion_matrix": cm.tolist(), "accuracy": acc}

    def save_model(self, model_path=None, vectorizer_path=None):
        response = {"status": "", "message": ""}
        try:
            model_path = model_path if model_path else self.settings.model_save_path
            vectorizer_path = vectorizer_path if vectorizer_path else self.settings.vectorizer_save_path

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

    def load_model(self, model_path=None, vectorizer_path=None):
        response = {"status": "", "message": ""}
        model_path = model_path if model_path else self.settings.model_save_path
        vectorizer_path = vectorizer_path if vectorizer_path else self.settings.vectorizer_save_path

        try:
            with open(model_path, 'rb') as model_file:
                self.model = pickle.load(model_file)
            with open(vectorizer_path, 'rb') as vectorizer_file:
                self.vectorizer = pickle.load(vectorizer_file)

            response["status"] = "success"
            response["message"] = "Model and vectorizer loaded successfully."
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
            raw_prediction = self.model.predict(features)
            # Example of converting raw_prediction to sentiment label
            sentiment = 'Positive' if raw_prediction[0] == 1 else 'Negative'
            return {"status": "success", "sentiment": sentiment}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
def analyze_sentiment(text=None, training_path=None, context_path=None):
    response = {"status": "", "message": "", "sentiment": ""}
    try:
        settings = Settings()
        training_path = training_path or settings.training_path
        context_path = context_path or settings.context_path
        preprocessor = DataPreprocessor()
        feature_extractor = FeatureExtractor(preprocessor, settings.feature_extraction_params)
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