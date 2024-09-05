import copy
import csv
import json
import time
import numpy as np
import random

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup


import warnings
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.")


def set_seed(seed):
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(preds: list, golds: list):
    if isinstance(preds[0], str):
        preds = ['yes' in pred.lower() or 'true' in pred.lower() for pred in preds]
    if isinstance(golds[0], str):
        golds = ['yes' in gold.lower() or 'true' in gold.lower() for gold in golds]
    f1 = f1_score(golds, preds)
    acc = accuracy_score(golds, preds)
    return f1, acc


def get_random_indices(dataset_len, num_samples, seed):
    df = pd.DataFrame({'index': list(range(dataset_len))})
    df = df.sample(n=num_samples, random_state=seed)
    return df['index'].tolist()


def train(tokenizer, model, train_dataset, valid_dataset, seed=42, patient=True, save_model=False,
          patience_start=0, **kwargs):
    set_seed(seed)
    lr = kwargs['lr']
    epochs = kwargs['epochs']
    base_model = kwargs['base_model']
    train_batch_size = kwargs['train_batch_size']
    valid_batch_size = kwargs['valid_batch_size']
    save_freq = kwargs['save_freq']
    patience = kwargs['patience']
    model_path = kwargs['save_model_path']
    result_prefix = kwargs['save_result_prefix']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"The training will use {torch.cuda.device_count()} GPUs!", flush=True)
        model = torch.nn.DataParallel(model)
    model.to(device)

    train_dl = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                          collate_fn=train_dataset.collate_fn, num_workers=2 * torch.cuda.device_count())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dl) * epochs)
    best_f1 = 0.0
    best_model = None
    train_stat_path = f'{result_prefix}_train_stat.csv'
    train_stat = ['train_loss, train_acc, train_f1, valid_loss, valid_acc, valid_f1, train_time']
    no_improvement = 0
    for epoch in range(epochs):
        # since the distribution of test set is unknown, we sample a subset of validation set during training
        if len(valid_dataset) > 2000:
            sampled_indices = get_random_indices(len(valid_dataset), 2000, 42+epoch)
            sampled_valid_dataset = Subset(valid_dataset, sampled_indices)
        else:
            sampled_valid_dataset = valid_dataset
        valid_dl = DataLoader(sampled_valid_dataset, batch_size=valid_batch_size, shuffle=False,
                              collate_fn=valid_dataset.collate_fn, num_workers=2 * torch.cuda.device_count())

        model.train()
        train_loss = 0.0
        train_preds, train_gts = [], []
        start_time = time.time()
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)

            if 'gpt' in base_model or 'bert' in base_model:
                loss, logits = output[:2]
                train_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                train_preds += logits.argmax(axis=-1).flatten().tolist()
                train_gts += batch['labels'].detach().cpu().numpy().flatten().tolist()
            elif 't5' in base_model:
                loss = output.loss.mean()
                train_loss += loss.detach().cpu().float().item()
                train_preds += tokenizer.batch_decode(torch.argmax(output.logits, dim=-1).detach().cpu().numpy(),
                                                      skip_special_tokens=True)
                train_gts += tokenizer.batch_decode(batch['labels'].detach().cpu().numpy(),
                                                    skip_special_tokens=True)
            else:
                raise ValueError('Invalid model')

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradient
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        end_time = time.time()

        valid_loss, valid_preds, valid_gts = evaluate(tokenizer, model, device, valid_dl, base_model)
        train_f1, train_acc = compute_metrics(train_preds, train_gts)
        valid_f1, valid_acc = compute_metrics(valid_preds, valid_gts)
        train_loss, valid_loss = train_loss / len(train_dl), valid_loss / len(valid_dl)

        print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | "
              f"Train acc: {train_acc * 100:.2f} | Valid acc: {valid_acc * 100:.2f} | "
              f"Train f1: {train_f1 * 100:.2f} | Valid f1: {valid_f1 * 100:.2f} | "
              f"Train Time: {end_time - start_time} secs", flush=True)
        stat_string = f"{train_loss}, {train_acc}, {train_f1}, {valid_loss}, {valid_acc}, {valid_f1}, " \
                      f"{end_time - start_time}"
        train_stat.append(stat_string)

        if epoch % save_freq == 0 and epoch > 0:
            save_data = {'prediction': valid_preds, 'ground_truth': valid_gts}
            save_path = f'{result_prefix}_valid_epoch_{epoch}.json'
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=4)

        if patient:
            if valid_f1 > best_f1:
                best_f1 = valid_f1
                best_model = copy.deepcopy(model)
                print(f"The best model is updated at epoch: {epoch+1} with f1 score {best_f1}", flush=True)
                no_improvement = 0
            else:
                if best_f1 > 1e-6:
                    no_improvement += 1
                if no_improvement >= patience and epoch > patience_start:
                    print(f"Early stopping at epoch: {epoch+1}", flush=True)
                    break
        else:
            best_model = model

    if save_model and patient:
        best_model.module.save_pretrained(model_path) if num_gpus > 1 else best_model.save_pretrained(model_path)

    with open(train_stat_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(train_stat)

    return best_model


@torch.no_grad()
def evaluate(tokenizer, model, device, eval_dataloader, base_model):
    model.eval()
    eval_loss = 0.0
    eval_preds, eval_gts = [], []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        if 'gpt' in base_model or 'bert' in base_model:
            loss, logits = output[:2]
            eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            eval_preds += logits.argmax(axis=-1).flatten().tolist()
            eval_gts += batch['labels'].detach().cpu().numpy().flatten().tolist()
        elif 't5' in base_model:
            eval_loss += output.loss.mean().detach().cpu().float().item()
            eval_preds += tokenizer.batch_decode(torch.argmax(output.logits, dim=-1).detach().cpu().numpy(),
                                                 skip_special_tokens=True)
            eval_gts += tokenizer.batch_decode(batch['labels'].detach().cpu().numpy(), skip_special_tokens=True)
        else:
            raise ValueError('Invalid model')
    return eval_loss, eval_preds, eval_gts


def inference(tokenizer, model, test_dataset, batch_size, base_model):
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    _, preds, gts = evaluate(tokenizer, model, device, test_dl, base_model)
    print('The predictions and ground truth are:', flush=True)
    print(preds, flush=True)
    print(gts, flush=True)
    test_f1, test_acc = compute_metrics(preds, gts)
    return test_f1, test_acc
