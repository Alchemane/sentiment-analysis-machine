from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import re, pandas as pd, joblib, os, nltk
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
        self.classifier = RandomForestClassifier(**settings.model_params)

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

    def predict(self, text):
        preprocessed_text = self.preprocessor.preprocess_text(text)
        features = self.feature_extractor.vectorizer.transform([preprocessed_text])
        return self.classifier.predict(features)
    
def analyze_sentiment(text=None, training_path=None, context_path=None):
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)
    default_data_path = os.path.join(project_root, '..', 'resources', 'Context Data', 'context.txt')

    preprocessor = DataPreprocessor()
    feature_extractor = FeatureExtractor()
    classifier = SentimentClassifier(preprocessor, feature_extractor)
    
    # If text is not provided but a context_path is, load and use the text from the file
    if text is None and context_path:
        with open(context_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()

    if not text:
        return "No text provided for sentiment analysis."

    elif training_path:
        if context_path:
            # Use provided context path if available
            X, y = preprocessor.load_data(context_path)
        else:
            # Fallback to default training data
            X, y = preprocessor.load_data(default_data_path)
        classifier.train(X, y)

    # Process and predict sentiment for the provided text
    preprocessed_text = preprocessor.preprocess_text(text)
    features = feature_extractor.vectorizer.transform([preprocessed_text])
    prediction = classifier.classifier.predict(features)
    overall_sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    
    return overall_sentiment

if __name__ == "__main__":
    sample_text = None
    context_path = r'C:\Users\Kevin\Desktop\Sentiment Analysis Machine\resources\Context Data\context.txt'
    training_path = r'C:\Users\Kevin\Desktop\Sentiment Analysis Machine\resources\Training Data\Restaurant_Reviews.tsv'

    sentiment = analyze_sentiment(sample_text, training_path=training_path, context_path=context_path)

    print(f"Sentiment prediction: {'Positive' if sentiment == 1 else 'Negative'}")