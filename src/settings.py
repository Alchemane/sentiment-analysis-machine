import json

class Settings:
    def __init__(self, settings_file=None):
        # Initialize default settings
        self.estimators = 100
        self.criterion = 'entropy'
        self.max_depth = None
        self.training_path = ''
        self.pretrained_model_path = ''
        
        # Load settings from file if provided
        if settings_file:
            self.load_settings(settings_file)

    def load_settings(self, settings_file):
        try:
            with open(settings_file, 'r') as file:
                settings = json.load(file)
            # Update class attributes based on file contents
            self.estimators = settings.get('model_settings', {}).get('n_estimators', 100)
            self.criterion = settings.get('model_settings', {}).get('criterion', 'entropy')
            self.max_depth = settings.get('model_settings', {}).get('max_depth', None)
            self.training_path = settings.get('data_paths', {}).get('training_data', '')
            self.pretrained_model_path = settings.get('data_paths', {}).get('pretrained_model', '')
        except FileNotFoundError:
            print(f"Settings file {settings_file} not found. Using default settings.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {settings_file}. Using default settings.")

    def save_settings(self, settings_file):
        settings = {
            'model_settings': {
                'n_estimators': self.estimators,
                'criterion': self.criterion,
                'max_depth': self.max_depth
            },
            'data_paths': {
                'training_data': self.training_path,
                'pretrained_model': self.pretrained_model_path
            }
        }
        with open(settings_file, 'w') as file:
            json.dump(settings, file, indent=4)

# Example usage
if __name__ == "__main__":
    settings = Settings('path/to/settings.json')
    print(settings.estimators, settings.criterion)
    # Optionally, save updated settings
    # settings.save_settings('path/to/updated_settings.json')