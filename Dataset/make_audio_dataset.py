import os, sys
import json
import torch
from tqdm import tqdm
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
my_modlue_dir = os.path.dirname(os.path.abspath(__file__)) + "/../phonetic_feature"
sys.path.insert(0, my_modlue_dir)

from embedding_generator import PhoneticEmbeddingGenerator


json_files = []
root_dir = "/home/eden/Documents/JNU/2025-1/Natural-Language-Processing/Natural-Language-Processing-Project/data/001.문서요약"

for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".json") and 'preprocessed' in filename:
            json_files.append(os.path.join(dirpath, filename))

generator = PhoneticEmbeddingGenerator(
    wav_dir = '/home/eden/Documents/JNU/2025-1/Natural-Language-Processing/Natural-Language-Processing-Project/data/tts_outputs_test/wav',
    mp3_dir = '/home/eden/Documents/JNU/2025-1/Natural-Language-Processing/Natural-Language-Processing-Project/data/tts_outputs_test/mp3'
)

feature_dir = '/home/eden/Documents/JNU/2025-1/Natural-Language-Processing/Natural-Language-Processing-Project/data/tts_outputs_test/feature'

os.makedirs(generator.wav_dir, exist_ok=True)

for json_file in json_files:
    # if "신문" not in json_file:
    #     continue
    print(f"Processing {json_file}...")
    # 예시: json 파일을 불러올 때
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    

    for entry in tqdm(data['data']):
        id_ = entry['id']
        for idx, sentence in enumerate(entry['sentence']+entry['g2p_result']):
            filename_hint = f"{id_}_{idx}"
            feature, wav_path, mp3_path = generator.encode(sentence, filename_hint)
            torch.save(feature, os.path.join(feature_dir, f"{filename_hint}.pt"))

    

    
# ## 테스트용 문장
# test_sents = [
#     '안녕하세요 테스트 입니다.',
#     '저는 자연어 처리가 너무 좋아요.'
# ]

# ## 테스트용 출력 경로를 설정한다 (실제 task에서는 기본 값 그대로 호출출)
# generator = PhoneticEmbeddingGenerator(
#     wav_dir = './tts_outputs_test/wav',
#     mp3_dir = './tts_outputs_test/mp3'
# )

# for i, sentece in enumerate(test_sents):
#     print(f"문장 {i+1}: \"{sentece}")

#     filename_hint = f'test_{i}'
#     feature, wav_path, mp3_path = generator.encode(sentece, filename_hint)

#     print(f" MP3 저장 위치: {mp3_path}")
#     print(f" WAV 저장 위치: {wav_path}")
#     print(f" Feature shape: {feature.shape}")
