from transformers import pipeline

def data():
    for i in range(1000):
        yield f"My example {i}"

class TransformerModel:
    def __init__(self, config, device_id) -> None:
        self.config = config

        batch_size = self.config.get('batch_size', 1)
        # self.generator = pipeline(
        #     model=self.config['model_name'], device=device_id,
        #     batch_size=batch_size)
        self.generator = pipeline(
            task='text-generation', model='../transformer_cache/gpt2',
            device=device_id, batch_size=batch_size)
        self.generator.tokenizer.pad_token_id = self.generator.model.config.eos_token_id
        self.pip_iter = self.generator(data())

    def __call__(self):
        # return self.generator("hello world")
        next(iter(self.pip_iter))
