from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence


class CustomDataLoader:
    def __init__(self, dataset, tokenizer, batch_size=8):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt",
                                                     padding="max_length", max_length=1024)

        # Format dataset with prompts and answers
        self.formatted_dataset = dataset.map(self._add_instruction_finetuning, remove_columns=dataset.column_names)
        self.formatted_dataset.set_format(type='torch', columns=['instr_tuned_text'])

    def _add_instruction_finetuning(self, rec):
        instruction = 'Label the sentiment of following sentence'
        INSTRUCTION_TEMPLATE = "{}:\nSentence:{}\nLabel:{}"
        label = "positive" if (rec["label"] == 1 or rec["label"] == "1") else "negative"
        rec["instr_tuned_text"] = INSTRUCTION_TEMPLATE.format(instruction, rec['text'], label)
        return rec

    def _tokenize(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding=True,
                              max_length=1024)  # Dynamic padding will be applied later

    def collate_fn(self, batch):
        # Extract texts from the batch
        texts = [item['instr_tuned_text'] for item in batch]

        # Tokenize all texts in the batch
        tokenized_batch = self.tokenizer(texts, truncation=True, padding=True,
                                         max_length=1024, return_tensors='pt')

        # Prepare labels: shift right, pad and append EOS token ID
        input_ids = tokenized_batch['input_ids']
        labels = input_ids[:, 1:].clone()  # Shift right
        labels = torch.cat([labels, torch.full((labels.size(0), 1), self.tokenizer.pad_token_id, dtype=torch.long)],
                           dim=1)  # Append EOS
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return input_ids_padded, labels_padded

    def get_loader(self, shuffle=True):
        return DataLoader(self.formatted_dataset, batch_size=self.batch_size, shuffle=shuffle,
                          collate_fn=self.collate_fn)
