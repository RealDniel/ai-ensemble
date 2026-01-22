import requests
from pathlib import Path
import json

import json

with open("text_data.json", "r", encoding="utf-8") as f:
    sources = json.load(f)


Path("raw_texts").mkdir(exist_ok=True)

for title, meta in sources.items():
    text = requests.get(meta["url"]).text
    with open(f"raw_texts/{title}.txt", "w", encoding="utf-8") as f:
        f.write(text)