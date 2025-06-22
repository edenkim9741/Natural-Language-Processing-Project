import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import wandb
from datetime import datetime

from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn import functional as F
from jiwer import cer
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from tqdm import tqdm

from torch.optim import AdamW
from sklearn.model_selection import KFold

from datasets import load_dataset
import ast

import os

import argparse

def tokenize_dataset(train_path, validation_path, dataset_path):
    # 토크나이저 및 모델 로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")
    model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")


    dataset = load_dataset("csv", data_files={"train": train_path, "validation": validation_path})

    def preprocess_function(examples):
        inputs = tokenizer(examples["input"], max_length=256, truncation=True, padding="max_length")
        targets = tokenizer(examples["translated_text"], max_length=256, truncation=True, padding="max_length")

        inputs["input_ids"] = inputs["input_ids"]
        inputs["labels"] = targets["input_ids"]
        inputs["input_wav"] = [ast.literal_eval(x) for x in examples["input_wav"]]
        # inputs["target_wav"] = [ast.literal_eval(x) for x in examples["target_wav"]]

        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["input", "original_text", "translated_text"], num_proc=16)

    # tokenized_dataset을 저장
    tokenized_dataset.save_to_disk(dataset_path)
    return tokenized_dataset




class CustomBartWithFusion(nn.Module):
    def __init__(self, bart_model_name="gogamza/kobart-base-v2"):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(bart_model_name)

        # Fusion Layer: [text_feat; phonetic_feat] → hidden_dim
        self.fusion = nn.Linear(768 + 768, 768)  # 고정 크기 가정 (BART hidden size = 768)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, phonetic_feat):
        # 1. BART Encoder
        encoder_outputs = self.model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )  # encoder_outputs.last_hidden_state: [B, T, 768]

        # text_feat: [B, T, 768]
        text_feat = encoder_outputs.last_hidden_state
        
        # phonetic_feat: [B, 768] → sequence 형태로 확장
        phonetic_feat_expanded = phonetic_feat.unsqueeze(1).expand(-1, text_feat.size(1), -1)  # [B, T, 768]

        # Fusion: [text_feat, phonetic_feat] → [B, T, 768]
        fused_encoder_hidden = self.fusion(torch.cat([text_feat, phonetic_feat_expanded], dim=-1))  # [B, T, 768]

        # 5. Decoder
        decoder_outputs = self.model.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=fused_encoder_hidden,
            encoder_attention_mask=attention_mask  # 그대로 사용 가능
        )

        # 6. Language modeling head
        lm_logits = self.model.lm_head(decoder_outputs.last_hidden_state)

        return lm_logits
    
def create_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id
    
    def collate_fn(batch, tokenizer=None):
        input_ids = [torch.tensor(example['input_ids']) for example in batch]
        attention_mask = [torch.tensor(example['attention_mask']) for example in batch]
        labels = [torch.tensor(example['labels']) for example in batch]
        phonetic_features = [torch.tensor(example['input_wav']) for example in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # BART의 labels는 -100으로 마스킹해야 loss에서 무시됨
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            'input_ids': input_ids,  # [batch_size, seq_length]
            'attention_mask': attention_mask,
            'labels': labels,
            'phonetic_features': torch.stack(phonetic_features, dim=0)  # [batch_size, phonetic_dim]
        }
    
    
    return collate_fn
    

def train_one_epoch(model, tokenizer, train_dataloader, optimizer, device, args):
    model.train()
    bar = tqdm(train_dataloader, desc="Training")
    for batch in bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        phonetic_features = batch["phonetic_features"].to(device)  # Assuming embedding is already a tensor
        tgt_input = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            decoder_input_ids=tgt_input, 
            decoder_attention_mask=(tgt_input != tokenizer.pad_token_id).long(),
            phonetic_feat=phonetic_features)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), tgt_input.view(-1), ignore_index=-100)
        bar.set_postfix(loss=loss.item())
        if args.log:
            wandb.log({"loss": loss.item()})
        loss.backward()
        optimizer.step()

def evaluate_model(model, tokenizer, valid_dataloader, device):
    model.eval()
    bar2 = tqdm(valid_dataloader, desc="Evaluating")
    epoch_loss = []
    
    preds = []
    refs = []
    
    with torch.no_grad():
        for batch in bar2:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            phonetic_features = batch["phonetic_features"].to(device)
            tgt_input = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                decoder_input_ids=tgt_input, 
                decoder_attention_mask=(tgt_input != tokenizer.pad_token_id).long(),
                phonetic_feat=phonetic_features)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), tgt_input.view(-1), ignore_index=-100)
            bar2.set_postfix(eval_loss=loss.item())
            epoch_loss.append(loss.item())
            
            pred_ids = outputs.argmax(dim=-1)
            pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_texts = tokenizer.batch_decode(tgt_input, skip_special_tokens=True)
            
            preds.extend(pred_texts)
            refs.extend(label_texts)
    return preds, refs, epoch_loss


def experiment(args):
    # 저장된 tokenized_dataset을 불러오기
    if os.path.exists(args.dataset_path):
        print("Loading tokenized dataset from disk...")
        dataset = load_from_disk(args.dataset_path)
    else:
        print("Tokenized dataset not found. Creating a new one...")
        dataset = tokenize_dataset(
            all_path=args.full_path,
            dataset_path=args.dataset_path
        )


    tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")
    device = torch.device(args.device)

        
    model = CustomBartWithFusion().to(device)
    model.load_state_dict(torch.load("translation_model_epoch_5.pt", map_location=device))
    print(len(dataset))
    # random_data_indices = random.sample(range(len(dataset)), 620489)  # Randomly select 100,000 indices
    # random_data
    # eval_dataset = dataset.select(random_data_indices)  # For evaluation, select a subset of the dataset
    eval_dataset = dataset  # Use the entire dataset for evaluation
    
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=create_collate_fn(tokenizer))

    preds, refs, epoch_loss = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        valid_dataloader=dataloader,
        device=device
    )
    # preds, refs = zip(preds, refs)

    wrong_predictions = [(p, r) for p, r in zip(preds, refs) if p != r]
    print(f"Number of wrong predictions: {len(wrong_predictions)}")
    print(f"Example wrong predictions: {wrong_predictions}")
    
    #     # KFold split
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # all_indices = list(range(len(dataset)))
    
    # for fold, (train_indices, val_indices) in enumerate(kf.split(all_indices)):
    #     if fold != 0: 
    #         continue
    #     if args.log:
    #         wandb.login()  # wandb 로그인
    #         wandb.init(
    #             project="Natural-Language_Processing",  # 원하는 프로젝트 이름으로 변경
    #             name=datetime.now().strftime(f"translation_%Y%m%d_%H%M%S_fold{fold}"),
    #             config={
    #                 "epochs": args.epochs,
    #                 "batch_size": args.batch_size,
    #                 "learning_rate": 5e-5,
    #                 "model": "PhoneticFusionBART_Translation",
    #                 "tokenizer": "gogamza/kobart-base-v2"
    #             }
    #         )


    #     model = CustomBartWithFusion().to(device)
    #     optimizer = AdamW(model.parameters(), lr=5e-5)

    #     train_dataset = dataset.select(train_indices)
    #     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=create_collate_fn(tokenizer))

    #     valid_dataset = dataset.select(val_indices)
    #     valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=create_collate_fn(tokenizer))

    #     from tqdm import tqdm
    #     for epoch in range(args.epochs):
    #         train_one_epoch(model, tokenizer, train_dataloader, optimizer, device, args)

    #         # gpu 메모리 해제
    #         torch.cuda.empty_cache()
            
    #         preds, refs, epoch_loss = evaluate_model(model, tokenizer, valid_dataloader, device)
                    
    #         # 지표 계산
    #         exact_match = sum([p == r for p, r in zip(preds, refs)]) / len(preds)
    #         cer_score = cer(refs, preds)
    #         bleu_scores = [sentence_bleu([ref], pred) for pred, ref in zip(preds, refs)]
    #         avg_bleu = sum(bleu_scores) / len(bleu_scores)

    #         _, _, bert_f1 = bert_score(preds, refs, lang="en")
    #         avg_bert_f1 = bert_f1.mean().item()
            
    #         metrics = {
    #             "exact_match": exact_match,
    #             "cer": cer_score,
    #             "avg_bleu": avg_bleu,
    #             "avg_bert_f1": avg_bert_f1,
    #             "epoch_eval_loss": sum(epoch_loss) / len(epoch_loss)
    #         }
    #         if args.log:
    #             wandb.log(metrics)
    #         print(f"Epoch {epoch + 1}, epoch_eval_loss: {sum(epoch_loss) / len(epoch_loss)}")

    #         # 모델 저장
    #         if args.save:
    #             torch.save(model.state_dict(), f"translation_model_epoch_{epoch + 1}.pt")
    #     if args.log:
    #         wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom BART model with phonetic fusion")
    parser.add_argument("--train_path", type=str, default="Dataset/train.csv", help="Path to the training dataset")
    parser.add_argument("--validation_path", type=str, default="Dataset/validation.csv", help="Path to the validation dataset")
    parser.add_argument("--test_path", type=str, default="Dataset/test.csv", help="Path to the test dataset")
    parser.add_argument("--dataset_path", type=str, default="Dataset/full_tokenized_dataset", help="Path to the tokenized dataset")
    parser.add_argument("--full_path", type=str, default="Dataset/temp_full_dataset.csv", help="Path to the full dataset for tokenization")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--log", action="store_true", help="Enable logging with wandb")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use for training (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument("--save", action="store_true", help="Save the model after training")
    args = parser.parse_args()
    
    experiment(args)

