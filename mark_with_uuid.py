import json
import uuid
from pathlib import Path

# 读取 summary.json 文件
file_path = Path("dialogue.json")
with file_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

# 为每个 data 元素添加 uuid
for entry in data:
    if "data" in entry:
        for item in entry["data"]:
            item["uuid"] = str(uuid.uuid4())

# 保存新文件
output_path = Path("summary_with_uuid.json")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)