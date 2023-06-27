import time
import torch
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer


def is_transformer(model_name, model_weight):
    return True

def data():
    for i in range(1000):
        yield f"My example {i}"

class TransformerModel:
    def __init__(self, config, device_id) -> None:
        self.config = config

        # batch_size = self.config.get('batch_size', 1)
        # # self.generator = pipeline(
        # #     model=self.config['model_name'], device=device_id,
        # #     batch_size=batch_size)
        # self.generator = pipeline(
        #     task='text-generation', model='../transformer_cache/gpt2',
        #     device=device_id, batch_size=batch_size, max_length=3)
        # self.generator.tokenizer.pad_token_id = self.generator.model.config.eos_token_id
        # self.generator.tokenizer.padding_size = 'left'
        # self.pip_iter = self.generator(data())


        model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer
        device = "cuda:" + str(device_id)
        tokenizer = tokenizer_class.from_pretrained("../transformer_cache/gpt2")
        self.model = model_class.from_pretrained("../transformer_cache/gpt2").to(device)
        prefix = "hello"
        prompt_text = "hello"
        self.encoded_prompt = tokenizer.encode(
            prefix + prompt_text, add_special_tokens=False,
            return_tensors="pt").cuda()
        if self.encoded_prompt.size()[-1] == 0:
            self.input_ids = None
        else:
            self.input_ids = self.encoded_prompt
        input_ids_len = 8
        batch_size = config['batch_size']
        self.input_ids = (torch.rand((batch_size, input_ids_len)) * 10000).long().to(device)

    def __call__(self):
        # return self.generator("hello world")
        start = time.time()
        # print(next(iter(self.pip_iter)))
        # self.model.generate()
        print(self.input_ids)
        output_sequences = self.model.generate(
                        input_ids=self.input_ids,
                        max_length=20 + 512,
                        temperature=1.0,
                        top_k=0,
                        top_p=0.9,
                        repetition_penalty=1.0,
                        do_sample=True,
                        num_return_sequences=1,
                    )
        print(time.time() - start)

# data_iter = data()
# for out in data_iter:
#     print(out)
# import pdb
# pdb.set_trace()
