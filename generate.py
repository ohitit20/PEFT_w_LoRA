import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import datasets
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score


class ModelSampler:
    def __init__(self, out_dir, init_from="resume", device="cuda", max_new_tokens=5, temperature=1.0, top_k=200):
        self.out_dir = out_dir
        self.init_from = init_from
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k

        # Initialize sampling as part of __init__
        self._initialize_sampling()

    def _initialize_sampling(self):
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        dtype = 'bfloat16'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.float16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        self.test_dataset = datasets.load_dataset("rotten_tomatoes", split='test').shuffle(seed=42).select(range(100))

        if self.init_from == 'resume':
            ckpt_path = os.path.join(self.out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            gptconf = GPTConfig(**checkpoint['model_args'])
            self.model = GPT(gptconf)
            self.model.load_state_dict(checkpoint['model'])
        elif self.init_from.startswith('gpt2'):
            self.model = GPT.from_pretrained(self.init_from, dict(dropout=0.0))
        else:
            print("Warning: Invalid Resume paramater!")
            return

        self.model.eval()
        self.model.to(self.device)

        enc = tiktoken.get_encoding("gpt2")
        self.encode = lambda s: enc.encode(s, allowed_special={""})
        self.decode = lambda l: enc.decode(l)

    def get_generation(self, prompt):
        prompt_ids = self.encode(prompt)
        x = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)[None, ...]

        with torch.no_grad():
            with self.ctx:
                y = self.model.generate(x, self.max_new_tokens, temperature=self.temperature, top_k=self.top_k)
                return self.decode(y[0].tolist())

    def get_accuracy(self):
        predictions = []
        targets = []
        positive_predictions = 0
        negative_predictions = 0
        counter = 0

        instruction = 'Label the sentiment of following sentence'
        INSTRUCTION_TEMPLATE = "{}:\nSentence:{}\nLabel:"
        for example in self.test_dataset:
            prompt_text = INSTRUCTION_TEMPLATE.format(instruction, example['text'])
            target_label = 'positive' if example['label'] == 1 else 'negative'
            targets.append(target_label)

            generated_text = self.get_generation(prompt_text)
            if('<|endoftext|>' in generated_text):
                generated_text = generated_text.split('<|endoftext|>')[0]
            positive = 'positive' in generated_text
            negative = 'negative' in generated_text

            if positive and negative:
                predicted_label = 'null'
            elif positive:
                predicted_label = 'positive'
                positive_predictions += 1
            elif negative:
                predicted_label = 'negative'
                negative_predictions += 1
            else:
                predicted_label = 'null'
            predictions.append(predicted_label)
            if(counter % 5 == 2):
                print('Prompt text:' + prompt_text)
                print('Generated text:' + generated_text)
                print('Target label:' + target_label)
                print('Predicted label:' + predicted_label)
            counter += 1

        accuracy = sum([1 for pred, target in zip(predictions, targets) if pred == target]) / len(targets)
        precision = precision_score(targets, predictions, labels=["positive", "negative"], average='micro')
        recall = recall_score(targets, predictions, labels=["positive", "negative"], average='micro')
        f1 = f1_score(targets, predictions, labels=["positive", "negative"], average='micro')
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('F1: ', f1)
        return accuracy, positive_predictions, negative_predictions
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get model accuracy from command line parameters.')
    parser.add_argument('--out_dir', type=str, help='Output directory for model checkpoints')
    parser.add_argument('--init_from', type=str, default='resume', help='Initialization method',
                        choices=['resume', 'gpt2', "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation')
    parser.add_argument('--max_new_tokens', type=int, default=5, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation')
    parser.add_argument('--top_k', type=int, default=200, help='Top K tokens to sample from')
    args = parser.parse_args()

    sampler = ModelSampler(args.out_dir, args.init_from, args.device, args.max_new_tokens, args.temperature, args.top_k)
    accuracy, pos_counter, neg_counter = sampler.get_accuracy()
    print(f"Accuracy: {accuracy}, Positive Predictions: {pos_counter}, Negative Predictions: {neg_counter}")
