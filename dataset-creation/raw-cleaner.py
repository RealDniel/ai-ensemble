import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from typing import List
import json
import os

#Open source .json file
with open("text_data.json", "r", encoding="utf-8") as f:
    SOURCES = json.load(f)



#Initial cleaning to get rid of gutenberg project formatting
def get_rid_of_gutenberg_stuff(text: str) -> str:
    start = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*", text)
    end = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*", text)

    if start and end:
        content = text[start.end():end.start()]
        content = re.sub("\n{3,}", "\n\n", content)
    else:
        content = text

    return content.strip()


#Chunk the book into sentences, then make a list of chunks of 2-5 sentences
def chunker(text: str, min_sents=2, max_sents=3) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    
    i = 0
    while(i < len(sentences)):
        chunk_size = min(max_sents, len(sentences) - i)
        if chunk_size < min_sents:
            break

        chunk = " ".join(sentences[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size

    return chunks


#Label the chunks per text/author, and return a json file containing those labelled chunks
def text_labeller(chunks: List[str], source: str, author: str, tradition: str):
    
    return [
        {
            "source" : source,
            "author" : author,
            "tradition" : tradition,
            "text" : c
        }
        for c in chunks
    ]


#Final formatting the json into proper format for finetuning
def final_format(labelled_text: List[dict]):
    formatted = []
    for t in labelled_text:
        formatted.append({
            "messages" : [
                {"role" : "assistant", "content" : t["text"]}
            ],
            "metadata" : {
                "source" : t["source"],
                "author" : t["author"],
                "tradition" : t["tradition"]
            }
        })

    return formatted


#Save into .json
def json_save(data: List[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():

    final_data = []
    raw_directory = "raw_texts"

    for title, meta in SOURCES.items():
        filename = f"{title}.txt"
        file_location = os.path.join(raw_directory, filename)


        with open(file_location, "r", encoding="utf-8") as f:
            text = f.read()
        
            text = get_rid_of_gutenberg_stuff(text)
            chunks = chunker(text)
            labelled_chunks = text_labeller(chunks, title, meta["author"], meta["tradition"])
            final_data.extend(final_format(labelled_chunks))

        
    json_save(final_data, "ancient_texts.jsonl")
    return 1


if __name__ == '__main__':
    main()