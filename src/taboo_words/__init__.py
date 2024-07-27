import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
import json
import tqdm

SYSTEM_MESSAGE = """
You are a good Taboo game word generator designed to output JSON. You speak Persian.
"""

USER_MESSAGE = """
Generate a list of 4 taboo words for the target word: "{}".
All the generated words should be in Persian.
Present the output in a JSON format with the structure {{"words": []}}.
Make it as challenging as possible for the player who is trying
to convey the target word.
Avoid using the target word in the generated list.
"""


def generate_taboo_words(client: OpenAI, word: str) -> List[str]:
    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE,
                },
                {
                    "role": "user",
                    "content": USER_MESSAGE.format(word),
                },
            ],
        )
        try:
            words = json.loads(response.choices[0].message.content)["words"]
            if len(words) >= 4:
                return words[:4]
        except Exception as e:
            print(e)
            continue


def generate_all(client: OpenAI, raw_words: pd.DataFrame) -> pd.DataFrame:
    result = []
    for i, row in tqdm.tqdm(raw_words.iterrows(), total=raw_words.shape[0]):
        taboo_words = generate_taboo_words(client, row["text"])
        (
            result.append(
                {
                    "text": row["text"],
                    "complexity": row["complexity"],
                    "taboo_word1": taboo_words[0],
                    "taboo_word2": taboo_words[1],
                    "taboo_word3": taboo_words[2],
                    "taboo_word4": taboo_words[3],
                }
            ),
        )
    return pd.DataFrame(result)


def main() -> int:
    raw_words = pd.read_csv("words.csv")
    raw_words = raw_words.sample(frac=1).reset_index(drop=True)
    raw_words = raw_words.head(100)

    load_dotenv()
    client = OpenAI()

    taboo_words = generate_all(client, raw_words)
    taboo_words.to_csv("taboo_words.csv", index=False)

    return 0
