import torch
import torch.nn as nn
from .embedding_generator import PhoneticEmbeddingGenerator

class PhoneticEncoder(nn.Module):
    """Phonetic Feature 학습용 인코더 제시
    학습 시 입력되는 phonetic feature tensor [B, T, 768]을 정제된 발음 임베딩 [B, D]로 변경!! """
    def __init__(self, input_dim=768, hidden_dim=256, num_layers = 2, bidirectional=True):
        super().__init__()

        ## 타 유닛으로 대체 가능!!
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim       ## 출력 차원의 명시적 저장!!

    def forward(self, x):
        """
        x: [B, T, input_dim] → 예: wav2vec2에서 추출된 특징 시퀀스
        return: [B, output_dim] → 문장 단위 발음 임베딩
        """
        _, h_n = self.gru(x)        ## h_n: [num_layers * num_directions, B, H]

        if self.gru.bidirectional:
            # 양방향일 경우 마지막 layer의 forward, backward 각각 추출하기
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=-1)
        else:
            # 단방향일 경우, 마지막 layer의 마지막 state를 사용한다다
            h_last = h_n[-1]    # [B, H]
        
        return h_last

if __name__ == "__main__":
    # Wav2Vec2 특징 벡터의 추출
    generator = PhoneticEmbeddingGenerator(
        wav_dir="./tts_outputs_test/wav",
        mp3_dir="./tts_outputs_test/mp3"
    )

    test = "이것은 테스트 문장입니다."
    feature, wav_path, mp3_path = generator.encode(test, filename_hint='encoder_test')

    # [T, 768]로 변경하기
    # -> extract_features()에서 mean(dim=1) 대신에 squeeze(0) 사용하도록 수정!!
    print(f"wav2vec2 raw feature shape: {feature.shape}")

    # 이건 수정 버전이라서 가정하고 진행한다
    feature = feature.squeeze(0).unsqueeze(0)

    # 인코더 통과하기
    encoder = PhoneticEncoder(input_dim=768, hidden_dim=256)
    output = encoder(feature)

    # 결과 확인
    print(f"Final Pronouncy Embedding shape: {output.shape}")