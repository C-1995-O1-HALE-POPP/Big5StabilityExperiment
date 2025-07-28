import json
from pathlib import Path
from typing import List, Dict
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
from tqdm import tqdm
import os
from datetime import datetime


from ocean_classifier.inference import big5_classifier
SEMAPHORE = Semaphore(32)
LOCK = Lock()

classifier = big5_classifier()



def inference(uuid, value):
    return {
        "uuid": uuid, 
        "scores": classifier.inference(texts=value["responses"]), 
        "config": value["config"]
    }

def save_incrementally(path, data):
    with LOCK:
        if os.path.exists(path):
            os.remove(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"runs/{exp_id}"
    ensure_dir(out_dir)

    output_path = os.path.join(out_dir, "scores.json")
    # 读取对话数据
    file_path = Path("dialogue.json")
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取所有助手回复内容
    n = 0
    dialogues = {}
    for entry in data:
        if "data" in entry:
            for item in entry["data"]:
                config = {
                    "assistant": item["assistant"],
                    "starter": item["starter"],
                    "background": item["background"],
                    "description": item["description"],
                }
                responses = []
                for i in item["content"]:
                    if i["role"] == "assistant":
                        responses.append(i["content"])
                        n += 1
                dialogues[item["uuid"]] = {"responses": responses, "config": config}
    logger.success(f"Loaded {len(dialogues.keys())} dialogues for scoring, total {n} responses.")

    output = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for uuid, value in tqdm(dialogues.items(), desc="Dialogues"):
            futures.append(executor.submit(inference, uuid, value))
        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                result = future.result()
                output[result["uuid"]] = {"scores": result["scores"], "config": result["config"]}
                save_incrementally(output_path, output)
                pbar.update(1)
    logger.success(f"Scores saved to {output_path}")

if __name__ == "__main__":
    main()