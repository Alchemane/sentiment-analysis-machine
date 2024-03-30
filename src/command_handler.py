from machine_learning import DataPreprocessor, FeatureExtractor, SentimentClassifier
from settings import Settings
import json, os
settings = Settings()

class CommandHandler:
    def __init__(self, settings_file='settings.json'):
        self.preprocessor = DataPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.classifier = SentimentClassifier(self.preprocessor, self.feature_extractor)
        self.settings_file = settings_file
        self.load_settings()
        # Updated command mappings
        self.commands = {
            "list_cmd": self.list_commands,
            "load_training": self.load_training_data,
            "load_context": self.load_context_data,
            "train_model": self.train_model,
            "predict_sentiment": self.predict_sentiment,
            "load_model": self.load_model,
            "change_settings": self.change_settings,
        }
        self.training_data = None
        self.context_data = None
        self.model = None

    def handle_command(self, command, *args):
        if command in self.commands:
            try:
                return self.commands[command](*args)
            except TypeError as e:
                return f"Invalid arguments for command '{command}': {e}"
        else:
            return "Unknown command"

    def list_commands(self):
        return "\n".join(self.commands.keys())

    def load_training_data(self, path):
        try:
            self.training_data, self.training_labels = self.preprocessor.load_data(path)
            return f"Training data loaded from {path}"
        except Exception as e:
            return f"Failed to load training data: {e}"

    def load_context_data(self, path):
        try:
            self.context_data = self.preprocessor.load_data(path, context=True)
            return f"Context data loaded from {path}"
        except Exception as e:
            return f"Failed to load context data: {e}"

    def train_model(self):
        if self.training_data and self.training_labels:
            self.classifier.train(self.training_data, self.training_labels)
            self.classifier.save_model(self.settings.model_path)
            return "Model trained and saved successfully."
        else:
            return "Training data not loaded."

    def predict_sentiment(self, text=None):
        if not self.model:
            return "Model not loaded or trained."
        if not text and self.context_data:
            # Load context data from the file
            with open(self.context_data, 'r', encoding='utf-8') as file:
                text = file.read()
        if not text:
            return "No text provided for sentiment analysis."
        # Proceed with sentiment analysis using the loaded or provided text
        prediction = self.classifier.predict(text)
        return f"Predicted sentiment: {'Positive' if prediction == 1 else 'Negative'}"

    def load_model(self, path):
        try:
            self.classifier.load_model(path)
            self.model = self.classifier.classifier
            return f"Model loaded from {path}"
        except Exception as e:
            return f"Failed to load model: {e}"

    def change_settings(self, setting, value):
        if setting in self.settings.model_params:
            try:
                self.settings.model_params[setting] = int(value) if setting == 'n_estimators' else value
                return f"Setting '{setting}' changed to {value}"
            except ValueError:
                return f"Invalid argument for {setting}. Must be an integer."
        else:
            return "Unknown setting."
        
    def save_settings(self):
        self.settings.save_settings('settings.json')
        return "Settings saved successfully."

    def load_settings(self):
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as file:
                self.settings = json.load(file)
        else:
            self.settings = {}