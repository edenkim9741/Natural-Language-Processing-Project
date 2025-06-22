import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch, torchaudio
import torch.nn.functional as F
import os

audio_model='facebook/wav2vec2-base-960h'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wav_processor = Wav2Vec2Processor.from_pretrained(audio_model)
wav_model = Wav2Vec2Model.from_pretrained(audio_model).to(device)

def extract_features(waveforms):
    # waveforms = []
    # for wav_path in wav_paths:
    #     waveform, sample_rate = torchaudio.load(wav_path)

    #     if sample_rate != 16000:
    #         resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    #         waveform = resampler(waveform)

    #     waveforms.append(waveform.squeeze(0))  # [1, T] → [T]

    max_len = max(waveform.shape[0] for waveform in waveforms)

    padded_waveforms = [F.pad(waveform, (0, max_len - waveform.shape[0]), value=0) for waveform in waveforms]
    padded_waveforms = torch.stack(padded_waveforms)  # [batch_size, T]

    # 가장 긴 waveform 기준으로 padding 처리
    inputs = wav_processor(padded_waveforms, sampling_rate=16000, return_tensors='pt', padding=True).to(wav_model.device)['input_values']
    inputs = inputs.squeeze(0).squeeze(0)
    # print(inputs.shape)  # [batch_size, T]

    # print(inputs)
    # raise

    with torch.no_grad():
        outputs = wav_model(inputs)

    return outputs.last_hidden_state.mean(dim=1)  # [batch_size, 768]

from transformers import VitsModel, AutoTokenizer
import uroman as ur

model = VitsModel.from_pretrained("facebook/mms-tts-kor").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
@torch.no_grad()
def make_tts_outputs(texts:str):
    uroman = ur.Uroman()

    romanized_texts = [uroman.romanize_string(t, lang="ko", to="latin") for t in texts]
    inputs = tokenizer(romanized_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        output = model(**inputs).waveform

    return output

import json
import pandas as pd
from tqdm import tqdm



json_file_list = os.listdir("/home/chaewon215/chbf/PatternSVG/temp/Natural-Language-Processing-Project/data/12.한영말뭉치")
json_dir_list = [os.path.join("/home/chaewon215/chbf/PatternSVG/temp/Natural-Language-Processing-Project/data/12.한영말뭉치", file) for file in json_file_list if file.endswith("preprocessed.json")]

# # JSON 데이터 로드
# with open("/home/chaewon215/chbf/PatternSVG/temp/Natural-Language-Processing-Project/data/001.문서요약/1.Training_1216_add/Train_법률_data/train_original_preprocessed.json", "r", encoding="utf-8") as f:
#     raw_data = json.load(f)["data"]

raw_data = []
for json_dir in json_dir_list:
    with open(json_dir, "r", encoding="utf-8") as f:
        data = json.load(f).get("data", [])
        raw_data.extend(data)

# 데이터의 5%만 사용
raw_data = raw_data[:(int(0.1 * len(raw_data)))]
# g2p_result 중 하나와 sentence를 매칭해서 여러 샘플 생성

half = int(len(raw_data) / 2)
raw_data = raw_data[half:]  # 데이터의 절반만 사용

pairs = []

batch_size = 16
batch_raw_data = [raw_data[i:i+batch_size] for i in range(0, len(raw_data), batch_size)]

for batch in tqdm(batch_raw_data):
    batch_wav_texts = []
    for item in batch:
        correct = item["sentence"][0]
        # batch_wav_texts.append(correct)  # 정답 문장 추가
        for g2p in item["g2p_result"]:
            batch_wav_texts.append(g2p)

    tts_outputs = make_tts_outputs(batch_wav_texts)
    features = extract_features(tts_outputs).to("cpu").tolist()

    feature_idx = 0

    for item in batch:
        correct = item["sentence"][0]
        g2p_len = len(item["g2p_result"])

        for g2p, phonetic_feature in zip(item["g2p_result"], features[feature_idx:g2p_len]):  # 첫 번째는 정답 문장
            pairs.append({"input": g2p, "target": correct, "input_wav": phonetic_feature})
            
        feature_idx += g2p_len  # +1은 정답 문장에 해당하는 feature
    


# train, validation, test 데이터로 나누기
import random
random.seed(42)
random.shuffle(pairs)
train_size = int(0.8 * len(pairs))
validation_size = int(0.1 * len(pairs))
test_size = len(pairs) - train_size - validation_size
train_pairs = pairs[:train_size]
validation_pairs = pairs[train_size:train_size + validation_size]
test_pairs = pairs[train_size + validation_size:]


# CSV로 저장
train_df = pd.DataFrame(train_pairs)
validation_df = pd.DataFrame(validation_pairs)
test_df = pd.DataFrame(test_pairs)

train_df.to_csv("train2.csv", index=False)
validation_df.to_csv("validation2.csv", index=False)
test_df.to_csv("test2.csv", index=False)

