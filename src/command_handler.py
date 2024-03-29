from machine_learning import DataPreprocessor, FeatureExtractor, SentimentClassifier
from settings import Settings
import json, os

class CommandHandler:
    def __init__(self, settings_file='settings.json'):
        self.settings = Settings()
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
        # Placeholder for loaded data and model
        self.training_data = None
        self.context_data = None
        self.model = None

    def handle_command(self, command, *args):
        """Execute the command if it exists."""
        if command in self.commands:
            try:
                return self.commands[command](*args)
            except TypeError:
                return f"Invalid arguments for command '{command}'"
        else:
            return "Unknown command"

    def list_commands(self):
        """List all available commands."""
        return "\n".join(self.commands.keys())

    def load_training_data(self, path):
        # Actual logic to load training data
        try:
            self.training_data, self.training_labels = self.preprocessor.load_data(path)
            return f"Training data loaded from {path}"
        except Exception as e:
            return f"Failed to load training data: {e}"

    def load_context_data(self, path):
        # Actual logic to load context data
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
        
        if text:
            prediction = self.classifier.predict(text)
            return f"Predicted sentiment for '{text}': {prediction}"
        elif self.context_data:
            predictions = [self.classifier.predict(data) for data in self.context_data]
            return f"Predicted sentiments for context data: {predictions}"
        else:
            return "No text or context data provided."

    def load_model(self, path):
        try:
            self.classifier.load_model(path)
            self.model = self.classifier.classifier
            return f"Model loaded from {path}"
        except Exception as e:
            return f"Failed to load model: {e}"

    def change_settings(self, setting, setting_arg):
        # Example: Changing the number of estimators in the RandomForestClassifier
        if setting == "n_estimators":
            try:
                n_estimators = int(setting_arg)
                self.settings.model_params['n_estimators'] = n_estimators
                return f"Setting '{setting}' changed to {setting_arg}"
            except ValueError:
                return "Invalid argument for n_estimators. Must be an integer."
        else:
            return "Unknown setting."
        
    def save_settings(self):
        try:
            with open(self.settings_file, 'w') as file:
                json.dump(self.settings, file)
            return "Settings saved successfully."
        except Exception as e:
            return f"Failed to save settings: {e}"

    def load_settings(self):
        """Load settings from a JSON file."""
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as file:
                self.settings = json.load(file)
        else:
            self.settings = {}