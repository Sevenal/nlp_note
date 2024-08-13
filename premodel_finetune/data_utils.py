from torch.utils.data import Dataset,IterableDataset,DataLoader
import json
from transformers  import  AutoTokenizer
import torch

class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt',encoding='utf-8') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 数据集过大时
class IterableAFQMC(IterableDataset):
    def __init__(self, data_file):
        self.data_file = data_file

    def __iter__(self):
        with open(self.data_file, 'rt') as f:
            for line in f:
                sample = json.loads(line.strip())
                yield sample


def collate_fn(batch_samples,tokenizer):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1, 
        batch_sentence_2, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)

    return X, y

# 由于collate_fn中只接收batch,使用包装器传递tokenizer参数
def collate_fn_wrapper(batch):
    return collate_fn(batch, tokenizer)

from args import args
train_dataset = AFQMC(args.train_file)
valid_dataset = AFQMC(args.valid_file)
tokenizer = AutoTokenizer.from_pretrained(args.pre_model)
train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn_wrapper)
valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn_wrapper)

if __name__ == "__main__":
    train_dataset = AFQMC("data/afqmc_public/train.json")
    checkpoint = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train = DataLoader(train_dataset, batch_size = 4, shuffle = True, collate_fn = collate_fn_wrapper)

    batch_x,batch_y = next(iter(train))
    print(batch_x,batch_y)
    print(batch_x['input_ids'].shape)