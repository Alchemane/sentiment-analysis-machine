import json, os

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Settings(metaclass=SingletonMeta):
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)

        # Define default settings directly
        self.model_params = {
            'n_estimators': 100,
            'criterion': 'entropy',
            'max_depth': None
        }
        self.feature_extraction_params = {
            'max_features': None,
            'ngram_range': (1, 1),
            'stop_words': 'english',
        }
        self.training_params = {
            'test_size': 0.2,
            'random_state': 42,
        }
        self.preprocessing_params = {
            'lower_case': True,
            'remove_punctuation': True,
            'stemming': True,
        }
        self.model_specific_params = {
            'svr': {'kernel': 'rbf', 'C': 100, 'gamma': 'auto'},
            'rfr': {'n_estimators': 100, 'max_depth': None},
        }
        self.current_model = 'rfr' # Optionally set to 'svr' for support vector regression
        # Default paths definition
        self.model_save_path = os.path.join(parent_dir, 'saved models', 'pretrained_model.pkl')
        self.vectorizer_save_path = os.path.join(parent_dir, 'saved models', 'vectorizer.pkl')
        self.training_path = os.path.join(parent_dir, 'training data', 'Restaurant_Reviews.tsv')
        self.context_path = os.path.join(parent_dir, 'context', 'context.txt')
        self.settings_dir = os.path.join(parent_dir, 'settings')
        self.settings_file = os.path.join(self.settings_dir, 'settings.json') # JSON

        # Load settings from a JSON file if available
        self.load_settings()

    def reset_settings(self):
        # Reinitialize default settings
        self.__init__()
        self.save_settings()
        print("Settings have been reset to defaults.")

    def load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as file:
                    settings_dict = json.load(file)
                for key, value in settings_dict.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            except Exception as e:
                print(f"Error loading settings from {self.settings_file}: {e}")
        else:
            print(f"{self.settings_file} not found. Using default settings.")
            self.save_settings()  # Create the settings file with default settings


    def save_settings(self):
        # Ensure the settings directory exists
        os.makedirs(self.settings_dir, exist_ok=True)
        
        # Save current settings to the JSON file
        settings_dict = {attr: getattr(self, attr) for attr in dir(self) if not attr.startswith('_') and not callable(getattr(self, attr))}
        try:
            with open(self.settings_file, 'w') as file:
                json.dump(settings_dict, file, indent=4)
            print("Settings saved successfully.")
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def update_settings(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a recognized setting")