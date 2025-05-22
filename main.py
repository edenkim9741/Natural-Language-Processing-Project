import torch
import torch.nn as nn
from bert_feature.bert_encoder import BertFeatureExtractor
from phonetic_feature.phonetic_encoder import PhoneticEncoder


class PhoneticCorrectionModel(nn.Module):
    def __init__(self, phonetic_dim, hidden_dim, vocab_size):
        super().__init__()
        self.text_encoder = BertFeatureExtractor()
        self.phonetic_encoder = PhoneticEncoder(phonetic_dim, hidden_dim)       ## phonetic_dim: 입력 벡터 차원, hidden_dim: 모델 내부부의 hidden_state 차원원

        # 각 모델의 pretraining weight load
        
        self.fusion = FeatureFusion(text_dim=768, phonetic_dim=hidden_dim * 2)
        self.decoder = TransformerDecoder(d_model=768, vocab_size=vocab_size)

    def forward(self, input_ids, attention_mask, phonetic_features, tgt_input):
        text_feat = self.text_encoder(input_ids, attention_mask)
        phonetic_feat = self.phonetic_encoder(phonetic_features)
        fused = self.fusion(text_feat, phonetic_feat)
        output = self.decoder(tgt_input, memory=fused)
        return output
