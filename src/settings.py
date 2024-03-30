import json, os

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Settings(metaclass=SingletonMeta):
    def __init__(self, settings_file=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_params = {
            'n_estimators': 100,
            'criterion': 'entropy',
            'max_depth': None
        }
        self.training_path = os.path.join(current_dir, '..', 'resources', 'training_data', 'Restaurant_Reviews.tsv')

        if settings_file:
            self.load_settings(settings_file)

    def load_settings(self, settings_file):
        try:
            with open(settings_file, 'r') as file:
                settings = json.load(file)
            self.model_params.update(settings.get('model_settings', {}))
            if 'training_path' in settings.get('data_paths', {}):
                self.training_path = settings['data_paths']['training_path']
        except FileNotFoundError:
            print(f"Settings file {settings_file} not found. Using default settings.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {settings_file}. Using default settings.")

    def save_settings(self, settings_file):
        settings = {
            'model_settings': self.model_params,
            'data_paths': {
                'training_data': self.training_path
            }
        }
        try:
            with open(settings_file, 'w') as file:
                json.dump(settings, file, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {e}")