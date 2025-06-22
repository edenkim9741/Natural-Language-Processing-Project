import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, VitsModel, AutoTokenizer, PreTrainedTokenizerFast, BartForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2Model
import wandb
from datetime import datetime

from datasets import load_from_disk

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn import functional as F
from jiwer import cer
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from tqdm import tqdm

from torch.optim import AdamW

from sklearn.model_selection import KFold

import torchaudio
from datasets import load_dataset
import ast

import os

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_model='facebook/wav2vec2-base-960h'
wav_processor = Wav2Vec2Processor.from_pretrained(audio_model)
wav_model = Wav2Vec2Model.from_pretrained(audio_model).to(device).eval()

tts_model = VitsModel.from_pretrained("facebook/mms-tts-kor").to(device).eval()
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")


tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")

@torch.no_grad()
def make_tts_outputs(text):
    tts_inputs = tts_tokenizer(text, max_length=256, truncation=True, padding="max_length", return_tensors="pt").to(device)  # [batch_size, seq_length]
    output = tts_model(**tts_inputs).waveform

    return output

@torch.no_grad()
def extract_features(waveforms):
    print(waveforms.shape)
    max_len = max(waveform.shape[0] for waveform in waveforms)

    padded_waveforms = [F.pad(waveform, (0, max_len - waveform.shape[0]), value=0).numpy() for waveform in waveforms.cpu()]
    # padded_waveforms = torch.stack(padded_waveforms).to("cpu")  # [batch_size, T]
    # print(f"Padded waveforms shape: {padded_waveforms.shape}")

    # 가장 긴 waveform 기준으로 padding 처리
    inputs = wav_processor(padded_waveforms, sampling_rate=16000, return_tensors='pt', padding=True)['input_values'].to(device)  # [batch_size, T]
    
    # inputs = inputs.squeeze(0)
    # print(f"Inputs shape: {inputs.shape}")
    outputs = wav_model(inputs)

    return outputs.last_hidden_state.mean(dim=1)  # [batch_size, 768]

def tokenize(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

    inputs = inputs
    return inputs

class CustomBartWithFusion(nn.Module):
    def __init__(self, bart_model_name="gogamza/kobart-base-v2"):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(bart_model_name)

        # Fusion Layer: [text_feat; phonetic_feat] → hidden_dim
        self.fusion = nn.Linear(768 + 768, 768)  # 고정 크기 가정 (BART hidden size = 768)

    def forward(self, input_ids, attention_mask, phonetic_feat, decoder_input_ids=None, decoder_attention_mask=None):
        max_length = 256  # 최대 생성 길이
        eos_token_id = self.model.config.eos_token_id  # EOS 토큰 ID
        # 1. BART Encoder
        encoder_outputs = self.model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )  # encoder_outputs.last_hidden_state: [B, T, 768]

        # text_feat: [B, T, 768]
        text_feat = encoder_outputs.last_hidden_state
        
        # phonetic_feat: [B, 768] → sequence 형태로 확장
        phonetic_feat_expanded = phonetic_feat.expand(-1, text_feat.size(1), -1)  # [B, T, 768]

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

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, phonetic_feat, max_length=64, num_beams=5, eos_token_id=None):
        """
        Beam Search 기반 문장 생성 함수
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        eos_token_id = eos_token_id if eos_token_id is not None else self.model.config.eos_token_id
        decoder_start_token_id = self.model.config.decoder_start_token_id

        # 1. Encoder + Fusion
        encoder_outputs = self.model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = encoder_outputs.last_hidden_state  # [B, T, 768]
        phonetic_feat_expanded = phonetic_feat.unsqueeze(1).expand(-1, text_feat.size(1), -1)
        fused_encoder_hidden = self.fusion(torch.cat([text_feat, phonetic_feat_expanded], dim=-1))  # [B, T, 768]

        # 2. Beam을 위해 각 배치 확장
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
        fused_encoder_hidden = fused_encoder_hidden.repeat_interleave(num_beams, dim=0)

        # 3. 초기 설정
        generated = torch.full((batch_size * num_beams, 1), decoder_start_token_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9  # 첫 토큰에서는 첫 beam만 살아 있도록
        beam_scores = beam_scores.view(-1)  # [B * num_beams]

        finished = torch.zeros_like(beam_scores, dtype=torch.bool)

        for step in range(max_length):
            decoder_outputs = self.model.model.decoder(
                input_ids=generated,
                encoder_hidden_states=fused_encoder_hidden,
                encoder_attention_mask=attention_mask
            )
            next_token_logits = self.model.lm_head(decoder_outputs.last_hidden_state[:, -1, :])  # [B*num_beams, V]
            next_token_logprobs = F.log_softmax(next_token_logits, dim=-1)

            vocab_size = next_token_logprobs.size(-1)
            scores = beam_scores.unsqueeze(1) + next_token_logprobs  # [B*num_beams, V]
            scores = scores.view(batch_size, num_beams * vocab_size)

            topk_scores, topk_indices = torch.topk(scores, num_beams, dim=-1)  # [B, num_beams]
            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size

            # 업데이트
            generated = generated.view(batch_size, num_beams, -1)
            new_generated = []
            for i in range(batch_size):
                beams = []
                for b in range(num_beams):
                    beams.append(torch.cat([generated[i, beam_indices[i, b]], token_indices[i, b].unsqueeze(0)], dim=0))
                new_generated.append(torch.stack(beams))
            generated = torch.stack(new_generated).view(batch_size * num_beams, -1)
            beam_scores = topk_scores.view(-1)

            # 종료 조건 처리
            is_eos = token_indices == eos_token_id
            if is_eos.any():
                finished |= is_eos.view(-1)

            if finished.all():
                break

        # 4. 결과 정리 (최고 점수 beam만 추출)
        generated = generated.view(batch_size, num_beams, -1)
        beam_scores = beam_scores.view(batch_size, num_beams)
        best_indices = torch.argmax(beam_scores, dim=-1)

        best_sequences = []
        for i in range(batch_size):
            best_seq = generated[i, best_indices[i]]
            best_sequences.append(best_seq)

        return torch.stack(best_sequences, dim=0)  # [B, L]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BART model with phonetic features")
    parser.add_argument("--input_text", type=str, default="많은 도시드른 나무를 시믈 쑤 인는 지여글 늘리기 위해 열심히 노력하고 있습니다.", help="input text for the model")

    model = CustomBartWithFusion().to(device)
    model.load_state_dict(torch.load("model_epoch_5.pt", map_location=device))

    input_text = parser.parse_args().input_text

    tts_out = make_tts_outputs(input_text)
    phonetic_features = extract_features(tts_out).to(device)

    input_ids = tokenize(input_text)["input_ids"].squeeze(0).to(device)  # [seq_length]
    attention_mask = tokenize(input_text)["attention_mask"].squeeze(0).to(device)  # [seq_length]


    outputs = model.generate(
        input_ids=input_ids.unsqueeze(0),  # [1, seq_length]
        attention_mask=attention_mask.unsqueeze(0),  # [1, seq_length]
        phonetic_feat=phonetic_features
    )


    # pred_ids = outputs.argmax(dim=-1)
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input Text: {input_text}")
    print(f"Predicted Text: {pred_text}")