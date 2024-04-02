from machine_learning import DataPreprocessor, FeatureExtractor, SentimentClassifier, analyze_sentiment
from settings import Settings
import textwrap

class CommandHandler:
    def __init__(self):
        self.settings = Settings()
        self.preprocessor = DataPreprocessor()
        self.feature_extractor = FeatureExtractor(self.preprocessor, self.settings.feature_extraction_params)
        self.classifier = SentimentClassifier(self.preprocessor, self.feature_extractor, self.settings.model_params)
        self.commands = {
            "list_cmd": self.list_commands,
            "load_training": self.load_training_data,
            "load_context": self.load_context_data,
            "train_model": self.train_model,
            "predict_sentiment": self.predict_sentiment,
            "load_model": self.load_model,
            "change_setting": self.change_setting,
            "reset_settings": self.reset_settings,
            "analyze_sentiment": self.analyze_sentiment,
            "clear_context": self.clear_context,
            "clear_training": self.clear_training,
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
        commands = [
            ("list_cmd", "Lists all available commands for sentiment analysis."),
            ("load_training {filepath}", "Loads training data from a specified .csv or .tsv file."),
            ("load_context {filepath}", "Loads context data for analysis from a specified .txt file."),
            ("train_model", "Trains the sentiment analysis model using the loaded training data."),
            ("predict_sentiment {text}", "Predicts the sentiment of the provided text directly or loaded context. Parameter is optional."),
            ("load_model {model_path}", "Loads a pre-trained model from the specified path. Parameter is optional."),
            ("change_setting {setting} {value}", "Updates a specific setting to the new specified value."),
            ("reset_settings", "Resets all settings to their default values."),
            ("analyze_sentiment {text} {training_path} {context_path}", "Comprehensive command for analyzing context quickly. All parameters are optional."),
            ("clear_context", "Clears the value of the context path in main memory."),
            ("reset_settings", "Clears the value of the training path in main memory."),
        ]
        first_column_width = 25
        max_description_width = 75
        formatted_commands = []
        for command, description in commands:
            wrapped_description_lines = textwrap.wrap(description, width=max_description_width)

            first_line = f"{command.ljust(first_column_width)}{wrapped_description_lines[0]}"
            formatted_commands.append(first_line)

            for additional_line in wrapped_description_lines[1:]:
                formatted_commands.append(' ' * first_column_width + additional_line)

        # This directly prints the result, but you can return it instead if needed
        formatted_command_list = "\n".join(formatted_commands)
        print(formatted_command_list)
        return formatted_command_list

    def load_training_data(self, path):
        # Check the file extension to decide how to process the file
        if path.endswith('.csv') or path.endswith('.tsv'):
            try:
                if path.endswith('.csv'):
                    self.training_data, self.training_labels = self.preprocessor.preprocess_csv(path)
                else:  # For '.tsv'
                    self.training_data, self.training_labels = self.preprocessor.preprocess_tsv(path)
                    
                return "Training data loaded and preprocessed successfully."
            except Exception as e:
                return f"Failed to load training data: {e}"
        else:
            return "Unsupported file type. Please use .csv or .tsv files."

    def load_context_data(self, filepath):
        try:
            self.context_data = filepath
            return f"Context data loaded successfully from {filepath}."
        except Exception as e:
            return f"Failed to load context data: {e}"

    def train_model(self):
        # Ensure training_data and training_labels are not None and not empty
        if self.training_data is not None and self.training_labels is not None and len(self.training_data) > 0 and len(self.training_labels) > 0:
            self.classifier.train(self.training_data, self.training_labels)
            self.classifier.save_model(self.settings.model_save_path)
            self.model = True
            return "Model trained and saved successfully."
        else:
            return "Training data not loaded."

    def predict_sentiment(self, text=None):
        if not self.model:
            return "Model not loaded or trained."

        context_path = self.context_data if self.context_data else self.settings.context_path
        if not text and context_path:
            try:
                with open(context_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            except FileNotFoundError:
                return f"Context data file not found at {context_path}."
            except Exception as e:
                return f"Failed to load context data from {context_path}: {e}"

        # Proceed with sentiment analysis using the loaded or provided text
        prediction_response = self.classifier.predict(text)
        if prediction_response["status"] == "success":
            return f"Predicted sentiment: {prediction_response['sentiment']}"
        else:
            return f"Error predicting sentiment: {prediction_response['message']}"

    def load_model(self, path=None):
        try:
            load_response = self.classifier.load_model(model_path=path, vectorizer_path=path)
            if load_response["status"] == "success":
                self.model = self.classifier.model
                return f"Model loaded successfully from {path if path is not None else '/saved models.'}"
            else:
                return load_response["message"]
        except Exception as e:
            return f"Failed to load model: {e}"

    def change_setting(self, setting_name, new_value):
        # Define valid options for the 'current_model' setting
        valid_model_types = ['rfr', 'svc']

        # Attempt to dynamically update a setting based on provided name and value
        if hasattr(self.settings, setting_name):
            # Special validation for 'current_model' to ensure it's a supported type
            if setting_name == "current_model" and new_value not in valid_model_types:
                return f"Error: '{new_value}' is not a supported model type. Please choose from {valid_model_types}."

            # Convert new_value to the correct type based on the current setting's type
            current_value = getattr(self.settings, setting_name)
            try:
                if isinstance(current_value, int):
                    new_value = int(new_value)
                elif isinstance(current_value, float):
                    new_value = float(new_value)
                elif isinstance(current_value, bool):
                    new_value = new_value.lower() in ('true', '1', 't')
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
    
    def analyze_sentiment(self, text=None, training_path=None, context_path=None):
        # Call the analyze_sentiment function
        result = analyze_sentiment(text, training_path, context_path or self.context_data)
        
        # Format and return the analysis result
        if result["status"] == "success":
            sentiment = result.get("sentiment", "Unknown")
            return f"Sentiment analysis result: {sentiment}"
        else:
            return f"Failed to analyze sentiment: {result.get('message', 'Unknown error')}"
        
    def clear_context(self):
        self.context_data = None
        return f"Cleared context data from main memory."
    
    def clear_training(self):
        self.training_data = None
        return f"Cleared training data from main memory."