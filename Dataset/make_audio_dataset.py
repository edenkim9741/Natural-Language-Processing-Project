import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch, torchaudio
import torch.nn.functional as F
import os

audio_model='facebook/wav2vec2-base-960h'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

wav_processor = Wav2Vec2Processor.from_pretrained(audio_model)
wav_model = Wav2Vec2Model.from_pretrained(audio_model).to(device0).eval()

# from transformers import WhisperProcessor, WhisperModel

# wav_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
# wav_model = WhisperModel.from_pretrained("openai/whisper-small").to(device).eval()

# # Load model directly
# from transformers import AutoProcessor, AutoModel

# wav_processor = AutoProcessor.from_pretrained("ntu-spml/distilhubert")
# wav_model = AutoModel.from_pretrained("ntu-spml/distilhubert")

@torch.no_grad()
def extract_features(waveforms):

    max_len = max(waveform.shape[0] for waveform in waveforms)

    padded_waveforms = [F.pad(waveform, (0, max_len - waveform.shape[0]), value=0).numpy() for waveform in waveforms.cpu()]
    # padded_waveforms = torch.stack(padded_waveforms).to("cpu")  # [batch_size, T]
    # print(f"Padded waveforms shape: {padded_waveforms.shape}")

    # 가장 긴 waveform 기준으로 padding 처리
    inputs = wav_processor(padded_waveforms, sampling_rate=16000, return_tensors='pt', padding=True)['input_values'].to(device0)  # [batch_size, T]
    
    inputs = inputs.squeeze(0).squeeze(0)
    outputs = wav_model(inputs)

    return outputs.last_hidden_state.mean(dim=1)  # [batch_size, 768]

from transformers import VitsModel, AutoTokenizer
import uroman as ur

model = VitsModel.from_pretrained("facebook/mms-tts-kor").to(device1).eval()
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")

@torch.no_grad()
def make_tts_outputs(inputs):
    
    output = model(**inputs).waveform

    return output

import json
import pandas as pd
from tqdm import tqdm



# json_file_list = os.listdir("/home/chaewon215/chbf/PatternSVG/temp/Natural-Language-Processing-Project/data/12.한영말뭉치")
# json_dir_list = [os.path.join("/home/chaewon215/chbf/PatternSVG/temp/Natural-Language-Processing-Project/data/12.한영말뭉치", file) for file in json_file_list if file.endswith("preprocessed.json")]

# # JSON 데이터 로드
# with open("/home/chaewon215/chbf/PatternSVG/temp/Natural-Language-Processing-Project/data/001.문서요약/1.Training_1216_add/Train_법률_data/train_original_preprocessed.json", "r", encoding="utf-8") as f:
#     raw_data = json.load(f)["data"]

# raw_data = []
# for json_dir in json_dir_list:
with open("/home/chaewon215/chbf/PatternSVG/temp/Natural-Language-Processing-Project/data/12.한영말뭉치/translate_data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f).get("data", [])
        # raw_data.extend(data)


# 데이터의 5%만 사용
# raw_data = raw_data[:(int(0.1 * len(raw_data)))]
# 데이터의 10%만 랜덤으로 추출
import random
random.seed(42)
raw_data = random.sample(raw_data, int(0.1 * len(raw_data)))  # 전체 데이터의 5%만 사용


# g2p_result 중 하나와 sentence를 매칭해서 여러 샘플 생성
pairs = []

batch_size = 32
# batch_raw_data = [raw_data[i:i+batch_size] for i in range(0, len(raw_data), batch_size)]
# uroman = ur.Uroman()


from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial

# 1. 평탄화된 g2p 전체 리스트 구성
all_g2ps = [g2p for entry in raw_data for g2p in entry["g2p_result"]]

# def init_uroman():
#     global _uroman
#     _uroman = ur.Uroman()

# def romanize_truncated(g2p, max_char_length=512):
#     global _uroman
#     return _uroman.romanize_string(g2p[:max_char_length], lang="ko", to="latin")


# num_workers = cpu_count()  # or manually set like 4, 8, etc.
# with Pool(processes=num_workers, initializer=init_uroman) as pool:
#     g2p_texts = list(tqdm(pool.imap(romanize_truncated, all_g2ps), total=len(all_g2ps)))


# # print(g2p_texts[:10])  # 확인용 출력

# from datasets import Dataset

# # 1. 리스트 → Dataset 객체로 변환
# g2p_dataset = Dataset.from_dict({"text": g2p_texts})


# print(f"Dataset size: {len(g2p_dataset)}")

# def preprocess_function(examples):
#     inputs = tokenizer(examples["text"], max_length=256, truncation=True, padding="max_length", return_tensors="pt")
#     return inputs

# batch_encoding = g2p_dataset.map(preprocess_function, batched=True, num_proc=cpu_count(), batch_size=1024)

# # batch_encoding 저장
# batch_encoding.save_to_disk("g2p_dataset")

# batch_encoding 불러오기
from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
batch_encoding = load_from_disk("g2p_dataset")

batch_encoding = batch_encoding.remove_columns(["text"])  # 'text' 컬럼 제거

print(f"Batch encoding size: {len(batch_encoding)}")

dataloader = DataLoader(batch_encoding, batch_size=batch_size, shuffle=False)


# # input_ids 추출
# g2ps = batch_encoding.remove_columns(["text"])  # 'text' 컬럼 제거

# # g2ps = [raw_data[i]["g2p_result"] for i in range(len(raw_data))]

# batch_g2ps = [g2ps[i:i+batch_size] for i in range(0, len(g2ps), batch_size)]

wav_feature = []

import numpy as np

for batch in tqdm(dataloader, desc="Processing batches"):
    # batch_wav_texts = []
    # for item in batch:
    #     correct = item["sentence"][0]
    #     # batch_wav_texts.append(correct)  # 정답 문장 추가
    #     for g2p in item["g2p_result"]:
    #         batch_wav_texts.append(g2p)
    inputs = {
        "input_ids": torch.from_numpy(np.array(batch["input_ids"])).to(device1),
        "attention_mask": torch.from_numpy(np.array(batch["attention_mask"])).to(device1)
    }

    tts_outputs = make_tts_outputs(inputs).to(device1)
    features = extract_features(tts_outputs).to("cpu").tolist()
    
    wav_feature.extend(features)


for item, feature in zip(raw_data, wav_feature):
    translated_text = item["translated_text"][0]
    original_text = item["orignal_text"]
    for g2p in item["g2p_result"]:
        pairs.append({"input": g2p, "translated_text": translated_text, "original_text": original_text, "input_wav": feature})
    


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

train_df.to_csv("train.csv", index=False)
validation_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

