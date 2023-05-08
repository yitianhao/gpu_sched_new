from transformers import pipeline

class TransformerModel:
    def __init__(self, config, device_id) -> None:
        self.config = config

        batch_size = self.config['batch_size'] if 'batch_size' in self.config else 1
        self.generator = pipeline(
            model=self.config['model_name'], device=device_id,
            batch_size=batch_size)

    def __call__(self):
        return self.generator("hello world")
