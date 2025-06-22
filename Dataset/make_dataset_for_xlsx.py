import json

import os, sys
import pandas as pd

from copy import deepcopy
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
my_modlue_dir = os.path.dirname(os.path.abspath(__file__)) + "/../g2pK"
sys.path.insert(0, my_modlue_dir)

from g2pk import G2p

from concurrent.futures import ProcessPoolExecutor, as_completed
# from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

g2p = G2p()  # 각 프로세스에서 개별로 생성

def process_sentence(args, translated_args):
    sentences_w_idx = args
    result = g2p(sentences_w_idx, sampling_num=5)
    temp_result = deepcopy(result)
    for idx, g2p_result in temp_result.items():
        result[idx] = {'g2p_result': g2p_result, 'sentence': list(translated_args[idx])}
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

xlsx_files = []
root_dir = "/home/chaewon215/chbf/PatternSVG/temp/Natural-Language-Processing-Project/data/12.한영말뭉치"

for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".xlsx") and 'preprocessed' not in filename:
            xlsx_files.append(os.path.join(dirpath, filename))

results = []

for xlsx_file in xlsx_files:
    # if "신문" not in json_file:
    #     continue
    print(f"Processing {xlsx_file}...")
    # 예시: json 파일을 불러올 때

    df = pd.read_excel(xlsx_file)

    # 모든 문장을 담을 리스트
    all_sentences = df["원문"]
    # 첫번째 열을 인덱스로 사용
    sent_idx = df.iloc[:, 0]
    all_translated_sentences = df["번역문"]

    args_list = [{idx: {sentence}} for idx, sentence in zip(sent_idx, all_sentences)]
    translated_list = [{idx: {sentence}} for idx, sentence in zip(sent_idx, all_translated_sentences)]

    translated_args = chunkify(translated_list, 256)
    batched_args = chunkify(args_list, 256)
    

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_sentence, args, t_args) for args, t_args in zip(batched_args, translated_args)]

        results_nested = []
        for future in tqdm(futures, total=len(futures), desc="Processing sentences"):
            results_nested.append(future.result())
            
    print("Processing complete.")

    for batch_result, batch_arg in zip(results_nested, batched_args):
        for (sent_id, sentence_and_g2pk), o_sent_id in zip(batch_result.items(), batch_arg.keys()):
            assert o_sent_id == sent_id, f"Mismatch in sentence ID: {o_sent_id} != {sent_id}"
            results.append({
                "id": sent_id,
                "orignal_text" : batch_arg[sent_id].pop(),
                "translated_text": sentence_and_g2pk["sentence"],
                "g2p_result": sentence_and_g2pk["g2p_result"]
            })


result = {"data": results}

with open(os.path.join(root_dir, "translate_data.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
        