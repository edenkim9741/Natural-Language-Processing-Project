import json

import os, sys

from copy import deepcopy
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
my_modlue_dir = os.path.dirname(os.path.abspath(__file__)) + "/../g2pK"
sys.path.insert(0, my_modlue_dir)

from g2pk import G2p

from concurrent.futures import ProcessPoolExecutor, as_completed
# from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

g2p = G2p()  # 각 프로세스에서 개별로 생성

def process_sentence(args):
    sentences_w_idx = args
    result = g2p(sentences_w_idx, sampling_num=5)
    temp_result = deepcopy(result)
    for idx, g2p_result in temp_result.items():
        result[idx] = {'g2p_result': g2p_result, 'sentence': list(sentences_w_idx[idx])}
    return result

def chunkify(lst, n):
    return_list = []
    for i in range(0, len(lst), n):
        temp_dict = {}
        for dict_i in lst[i:i+n]:
            for key, value in dict_i.items():
                if key not in temp_dict:
                    temp_dict[key] = set()
                temp_dict[key].update(value)
        return_list.append(temp_dict)

    return return_list

json_files = []
root_dir = "/home/eden/Documents/JNU/2025-1/Natural-Language-Processing/Natural-Language-Processing-Project/data/001.문서요약"

for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".json") and 'preprocessed' not in filename:
            json_files.append(os.path.join(dirpath, filename))

for json_file in json_files:
    # if "신문" not in json_file:
    #     continue
    print(f"Processing {json_file}...")
    # 예시: json 파일을 불러올 때
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 모든 문장을 담을 리스트
    all_sentences = []

    # 각 document에 대해 순회
    for doc in data["documents"]:
        for paragraph in doc["text"]:  # text는 리스트 안에 리스트 구조
            for sentence_obj in paragraph:
                all_sentences.append(sentence_obj["sentence"])

    args_list = [{idx: {sentence}} for idx, sentence in enumerate(all_sentences)]
    batched_args = chunkify(args_list, 256)

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_sentence, args) for args in batched_args]

        results_nested = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing sentences"):
            results_nested.append(future.result())

    results = []
    for batch_result in results_nested:
        for sent_id, sentence_and_g2pk in batch_result.items():
            results.append({
                "id": sent_id,
                "sentence": sentence_and_g2pk["sentence"],
                "g2p_result": sentence_and_g2pk["g2p_result"]
            })


    result = {"data": results}


    with open(json_file.replace(".json", "_preprocessed.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        