import torchaudio
import torch
import os
import uuid
import subprocess

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from gtts import gTTS

class PhoneticEmbeddingGenerator:
    """음성 기반의 임베딩 생성을 위한 TTS + Wav2Vec2 추론 파이프라인
    쉽게 말하면, '문장 하나 -> 발음 .wav 생성 -> wav2vec2 특징 벡터 추출[1, 768]' """
    def __init__(self, tts_engine='gtts', 
                 audio_model='facebook/wav2vec2-base-960h', 
                 wav_dir='/home/jovyan/Data/tts_datasets/wav', 
                 mp3_dir='/home/jovyan/Data/tts_datasets/mp3'):
        self.tts_engine = tts_engine.lower()
        self.wav_dir = wav_dir
        self.mp3_dir = mp3_dir
        os.makedirs(wav_dir, exist_ok=True)
        os.makedirs(mp3_dir, exist_ok=True)

        ## 허깅페이스 Transformer로부터 모델 로드하기
        self.processor = Wav2Vec2Processor.from_pretrained(audio_model)
        self.model = Wav2Vec2Model.from_pretrained(audio_model)
        self.model.eval()   ## 추론 전용으로 일단 돌림

    def synthesize_tts(self, text: str, base_filename: str):
        """mp3 파일에서 wav 파일로의 변환"""
        # 경로 생성하기
        mp3_path = os.path.join(self.mp3_dir, f"{base_filename}.mp3")
        wav_path = os.path.join(self.wav_dir, f"{base_filename}.wav")

        ## tts 변환하기
        tts = gTTS(text=text, lang='ko')
        tts.save(mp3_path)

        ## mp3 -> wav 변경하기기
        subprocess.call(['ffmpeg', '-y', '-i', mp3_path, wav_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

    
    def extract_features(self, wav_path):
        waveform, sample_rate = torchaudio.load(wav_path)       ## 일단 특징 벡터 추출을 위해 wav 파일을 로드함

        ## 모든 wav 파일은 16kHz로 리샘플링
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        inputs = self.processor(waveform.squeeze(), sampling_rate=16000, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state.mean(dim=1)        # [1, 768] 크기 고정 벡터로 변환환
    
    
    def encode(self, text: str, filename_hint=None):
        """전체 파이프라인 요약"""
        base_name = filename_hint or str(uuid.uuid4())[:8]

        # TTS + wav 생성성
        self.synthesize_tts(text, base_name)

        # 경로 설정
        mp3_path = os.path.join(self.mp3_dir, f"{base_name}.mp3")
        wav_path = os.path.join(self.wav_dir, f"{base_name}.wav")

        feature = self.extract_features(wav_path)   # 특징 추출하기기

        return feature, wav_path, mp3_path
    

## 제너레이터 테스트용 
if __name__ == "__main__":
    ## 테스트용 문장
    test_sents = [
        '안녕하세요 테스트 입니다.',
        '저는 자연어 처리가 너무 좋아요.'
    ]

    ## 테스트용 출력 경로를 설정한다 (실제 task에서는 기본 값 그대로 호출출)
    generator = PhoneticEmbeddingGenerator(
        wav_dir = './tts_outputs_test/wav',
        mp3_dir = './tts_outputs_test/mp3'
    )

    for i, sentece in enumerate(test_sents):
        print(f"문장 {i+1}: \"{sentece}")

        filename_hint = f'test_{i}'
        feature, wav_path, mp3_path = generator.encode(sentece, filename_hint)

        print(f" MP3 저장 위치: {mp3_path}")
        print(f" WAV 저장 위치: {wav_path}")
        print(f" Feature shape: {feature.shape}")

