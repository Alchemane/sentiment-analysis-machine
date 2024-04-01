from machine_learning import DataPreprocessor, FeatureExtractor, SentimentClassifier
from settings import Settings
import json, os
settings = Settings()

class CommandHandler:
    def __init__(self):
        self.settings = Settings()
        self.preprocessor = DataPreprocessor()
        self.feature_extractor = FeatureExtractor(self.settings.feature_extraction_params)
        self.classifier = SentimentClassifier(self.preprocessor, self.feature_extractor, self.settings.model_params)
        # Updated command mappings
        self.commands = {
            "list_cmd": self.list_commands,
            "load_training": self.load_training_data,
            "load_context": self.load_context_data,
            "train_model": self.train_model,
            "predict_sentiment": self.predict_sentiment,
            "load_model": self.load_model,
            "change_settings": self.change_settings,
            "reset_settings": self.reset_settings,
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

    def change_settings(self, setting_name, new_value):
        # Attempt to dynamically update a setting based on provided name and value
        if hasattr(self.settings, setting_name):
            # Convert new_value to the correct type based on the current setting's type
            current_value = getattr(self.settings, setting_name)
            try:
                if isinstance(current_value, int):
                    new_value = int(new_value)
                elif isinstance(current_value, float):
                    new_value = float(new_value)
                elif isinstance(current_value, bool):
                    new_value = new_value.lower() in ('true', '1', 't')
                # Add more type conversions as necessary

                # Update the setting
                setattr(self.settings, setting_name, new_value)
                self.settings.save_settings()  # Save changes to the JSON file
                return f"Setting '{setting_name}' updated to {new_value}."
            except ValueError as e:
                return f"Error: Could not convert {new_value} to the correct type for '{setting_name}': {e}"
        else:
            return f"Unknown setting '{setting_name}'."

    def reset_settings(self):
        # Reset settings to their defaults
        self.settings.reset_settings()
        return "Settings have been reset to defaults."