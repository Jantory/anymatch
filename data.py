from torch.utils.data import Dataset
import torch


class BaseT5Dataset(Dataset):
    def __init__(self, tokenizer, descriptions, targets, max_len=350):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_descriptions, self.text_targets = descriptions, targets
        self.descriptions, self.targets = self.__filter__(self.tokenizer, self.text_descriptions, self.text_targets)

    def __filter__(self, tokenizer, text_descriptions, text_targets):
        'filter out the data that is too long'
        descriptions = []
        targets = []
        filtered_num = 0
        indices = []
        for i, text_description in enumerate(text_descriptions):
            description = tokenizer.encode(text_description)
            target = tokenizer.encode(text_targets[i])
            if len(description) <= self.max_len:
                descriptions.append(description)
                targets.append(target)
            else:
                filtered_num += 1
                indices.append(i)
        print(f'{filtered_num} out of {len(text_targets)} data samples are filtered.')
        return descriptions, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slicing
            return [(self.descriptions[i], self.targets[i]) for i in range(*idx.indices(len(self)))]
        else:
            # Handle single index
            return self.descriptions[idx], self.targets[idx]

    def collate_fn(self, batch):
        max_len_data = 0
        max_len_label = 0
        for description, target in batch:
            if len(description) > max_len_data: max_len_data = len(description)
            if len(target) > max_len_label: max_len_label = len(target)

        attn_masks = []
        targets = []
        descriptions = []
        # llama2 is not using pad_token_id
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
        for description, target in batch:
            description.extend([pad_token_id] * (max_len_data - len(description)))
            descriptions.append(description)

            attn_mask = [int(e != pad_token_id) for e in description]
            attn_masks.append(attn_mask)

            target.extend([0] * (max_len_label - len(target)))
            targets.append(target)
        model_inputs = {'input_ids': torch.LongTensor(descriptions), 'attention_mask': torch.LongTensor(attn_masks),
                        'labels': torch.LongTensor(targets)}
        return model_inputs


class T5Dataset(BaseT5Dataset):
    def __init__(self, tokenizer, data_df, max_len=350):
        data_df['label'] = data_df['label'].map({0: 'False', 1: 'True'})
        descriptions = data_df['text'].apply(lambda x: x.strip())
        targets = data_df['label'].apply(lambda x: x.strip())

        super().__init__(tokenizer, descriptions, targets, max_len=max_len)


class BaseGPTDataset(Dataset):
    def __init__(self, tokenizer, descriptions, targets, max_len=350):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_descriptions, self.text_targets = descriptions, targets
        self.descriptions, self.targets = self.__filter__(self.tokenizer, self.text_descriptions, self.text_targets)

    def __filter__(self, tokenizer, text_descriptions, text_targets):
        'filter out the data that is too long'
        descriptions = []
        targets = []
        filtered_num = 0
        indices = []
        for i, text_description in enumerate(text_descriptions):
            description = tokenizer.encode(text_description)
            target = 1 if text_targets[i] == 'True' or text_targets[i] == 'Yes' else 0
            if len(description) <= self.max_len:
                descriptions.append(description)
                targets.append(target)
            else:
                filtered_num += 1
                indices.append(i)
        print(f'{filtered_num} out of {len(text_targets)} data samples are filtered.')
        return descriptions, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slicing
            return [(self.descriptions[i], self.targets[i]) for i in range(*idx.indices(len(self)))]
        else:
            # Handle single index
            return self.descriptions[idx], self.targets[idx]

    def collate_fn(self, batch):
        max_len_data = 0
        for description, target in batch:
            if len(description) > max_len_data: max_len_data = len(description)

        attn_masks = []
        targets = []
        descriptions = []
        if self.tokenizer.pad_token_id:
            pad_token_id = self.tokenizer.pad_token_id
        elif self.tokenizer.eos_token_id:
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = 0
        for description, target in batch:
            description.extend([pad_token_id] * (max_len_data - len(description)))
            descriptions.append(description)

            attn_mask = [int(e != pad_token_id) for e in description]
            attn_masks.append(attn_mask)

            targets.append(target)
        model_inputs = {'input_ids': torch.LongTensor(descriptions), 'attention_mask': torch.LongTensor(attn_masks),
                        'labels': torch.LongTensor(targets)}
        return model_inputs


class GPTDataset(BaseGPTDataset):
    def __init__(self, tokenizer, data_df, max_len=350):
        data_df['label'] = data_df['label'].map({0: 'No', 1: 'Yes'})
        descriptions = data_df['text'].apply(lambda x: x.strip())
        targets = data_df['label'].apply(lambda x: x.strip())

        super().__init__(tokenizer, descriptions, targets, max_len=max_len)


class BertDataset(BaseGPTDataset):
    def __init__(self, tokenizer, data_df, max_len=350):
        data_df['label'] = data_df['label'].map({0: 'No', 1: 'Yes'})
        descriptions = data_df['text'].apply(lambda x: x.strip())
        targets = data_df['label'].apply(lambda x: x.strip())

        super().__init__(tokenizer, descriptions, targets, max_len=max_len)
