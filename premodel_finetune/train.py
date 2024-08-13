from tqdm.auto import tqdm
from args import args
import torch
from data_utils import AFQMC, IterableAFQMC, collate_fn, collate_fn_wrapper

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = (epoch-1)*len(dataloader)
    
    model.train()
    for step, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(args.device), y.to(args.device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(args.device), y.to(args.device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size

    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
    return correct

from transformers import AdamW
from model import  BertForPairwiseCLS
from transformers import AutoConfig
from torch import nn
from transformers import get_scheduler
from data_utils import train_dataloader,valid_dataloader

def run(train_dataloader, valid_dataloader):
    config = AutoConfig.from_pretrained(args.pre_model)
    model = BertForPairwiseCLS.from_pretrained(args.pre_model, config = config).to(args.device)
    optimizer = AdamW(model.parameters(), lr = args.lr)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.epoch_num*len(train_dataloader),
    )

    # total_loss = 0.
    # for t in range(args.epoch_num):
    #     print(f"Epoch {t+1}/{args.epoch_num}\n-------------------------------")
    #     total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    #     test_loop(valid_dataloader, model, mode='Valid')
    
    total_loss = 0.
    best_acc = 0.
    for t in range(args.epoch_num):
        print(f"Epoch {t+1}/{args.epoch_num}\n-------------------------------")
        total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
        valid_acc = test_loop(valid_dataloader, model, mode='Valid')
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('saving new weights...\n')
            torch.save(model.state_dict(), f'epoch_{t+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.bin')
    print("Done!")


if __name__ == '__main__':
    train_dataset = AFQMC("data/afqmc_public/train.json")
    valid_dataset = AFQMC("data/afqmc_public/dev.json")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pre_model)
    run(train_dataloader,valid_dataloader)
