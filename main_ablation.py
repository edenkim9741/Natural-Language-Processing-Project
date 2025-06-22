import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import wandb
from datetime import datetime
from jiwer import cer
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from tqdm import tqdm

wandb.init(
    project="Natural-Language_Processing",  # 원하는 프로젝트 이름으로 변경
    name=datetime.now().strftime("ablation-%Y%m%d%H%M"),
    config={
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "model": "BartForConditionalGeneration(ablation)",
        "tokenizer": "gogamza/kobart-base-v2"
    }
)
# 저장된 tokenized_dataset을 불러오기
from datasets import load_from_disk
tokenized_dataset = load_from_disk("tokenized_dataset")

class CustomBartWithFusion(nn.Module):
    def __init__(self, bart_model_name="gogamza/kobart-base-v2"):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(bart_model_name)

        # Fusion Layer: [text_feat; phonetic_feat] → hidden_dim
        # self.fusion = nn.Linear(768 + 768, 768)  # 고정 크기 가정 (BART hidden size = 768)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, phonetic_feat):
        # 1. BART Encoder
        encoder_outputs = self.model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )  # encoder_outputs.last_hidden_state: [B, T, 768]

        # text_feat: [B, T, 768]
        text_feat = encoder_outputs.last_hidden_state

        # # 3. Fusion: [text_feat; phonetic_feat] → [B, 768]
        # fused_feat = self.fusion(torch.cat([text_feat, phonetic_feat], dim=-1))  # [B, 768]

        # # 4. BART decoder는 encoder_hidden_states가 [B, T, 768]이길 기대하므로 repeat
        # batch_size, seq_len = encoder_outputs.last_hidden_state.size(0), encoder_outputs.last_hidden_state.size(1)
        # fused_encoder_hidden = fused_feat.unsqueeze(1).repeat(1, seq_len, 1)  # [B, T, 768]

        # 5. Decoder
        decoder_outputs = self.model.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=text_feat,  # [B, T, 768]
            encoder_attention_mask=attention_mask  # 그대로 사용 가능
        )

        # 6. Language modeling head
        lm_logits = self.model.lm_head(decoder_outputs.last_hidden_state)

        return lm_logits

from torch.nn.utils.rnn import pad_sequence
import torch

tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")

def collate_fn(batch):
    input_ids = [torch.tensor(example['input_ids']) for example in batch]
    attention_mask = [torch.tensor(example['attention_mask']) for example in batch]
    labels = [torch.tensor(example['labels']) for example in batch]
    phonetic_features = [torch.tensor(example['input_wav']) for example in batch]


    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # BART의 labels는 -100으로 마스킹해야 loss에서 무시됨
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids,  # [batch_size, seq_length]
        'attention_mask': attention_mask,
        'labels': labels,
        'phonetic_features': torch.stack(phonetic_features, dim=0)  # [batch_size, phonetic_dim]
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
phonetic_model = CustomBartWithFusion().to(device)
from torch.optim import AdamW
optimizer = AdamW(phonetic_model.parameters(), lr=5e-5)
from torch.utils.data import DataLoader
from torch.nn import functional as F

train_dataset = tokenized_dataset["train"]
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

valid_dataset = tokenized_dataset["validation"]
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

from tqdm import tqdm
for epoch in range(5):
    phonetic_model.train()
    bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
    for batch in bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        phonetic_features = batch["phonetic_features"].to(device)  # Assuming embedding is already a tensor
        tgt_input = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = phonetic_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            decoder_input_ids=tgt_input, 
            decoder_attention_mask=(tgt_input != tokenizer.pad_token_id).long(),
            phonetic_feat=phonetic_features)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), tgt_input.view(-1), ignore_index=-100)
        bar.set_postfix(loss=loss.item())
        wandb.log({"loss": loss.item()})
        loss.backward()
        optimizer.step()
    torch.cuda.empty_cache()
    phonetic_model.eval()
    bar2 = tqdm(valid_dataloader, desc=f"Validation Epoch {epoch + 1}")
    epoch_loss = []
    
    preds = []
    refs = []
    
    with torch.no_grad():
        for batch in bar2:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            phonetic_features = batch["phonetic_features"].to(device)
            tgt_input = batch["labels"].to(device)

            outputs = phonetic_model(
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
            
    # 지표 계산
        # 지표 계산
    exact_match = sum([p == r for p, r in zip(preds, refs)]) / len(preds)
    cer_score = cer(refs, preds)
    bleu_scores = [sentence_bleu([ref], pred) for pred, ref in zip(preds, refs)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    _, _, bert_f1 = bert_score(preds, refs, lang="ko")
    avg_bert_f1 = bert_f1.mean().item()
    
    metrics = {
        "exact_match": exact_match,
        "cer": cer_score,
        "avg_bleu": avg_bleu,
        "avg_bert_f1": avg_bert_f1
    }
    wandb.log(metrics)
    wandb.log({"epoch_eval_loss": sum(epoch_loss) / len(epoch_loss)})
    print(f"Epoch {epoch + 1}, epoch_eval_loss: {sum(epoch_loss) / len(epoch_loss)}")

    # 모델 저장
    torch.save(phonetic_model.state_dict(), f"ablation_model_epoch_{epoch + 1}.pt")

wandb.finish()
