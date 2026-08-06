import re
import json
import pandas as pd
from datasets import load_dataset

# Sentence Tokenizer
def sentence_tokenize(text):
    """Splits paragraph into sentences using Hindi danda and English punctuation."""
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r'(?<=[।.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


# Word Tokenizer
TOKEN_PATTERN = re.compile(
    r'''
    https?://\S+                      |  # URL
    www\.\S+                          |  # www URL
    [\w\.-]+@[\w\.-]+\.\w+            |  # Email
    \d{1,2}[/-]\d{1,2}[/-]\d{2,4}     |  # Date (dd/mm/yyyy)
    \d{4}[/-]\d{1,2}[/-]\d{1,2}       |  # Date (yyyy/mm/dd)
    \d+\.\d+                          |  # Decimal number
    \d+                               |  # Integer
    [^\W\d_]+                         |  # Hindi/English words
    [.!?।]+                           |  # Repeated sentence punctuation (..., !!)
    [^\s]                                # Any other single punctuation
    ''',
    re.VERBOSE | re.UNICODE
)

def word_tokenize(sentence):
    return TOKEN_PATTERN.findall(sentence)


# Main
MAX_PARAGRAPHS = 10000  # set to None to process the full corpus

dataset = load_dataset(
    "ai4bharat/IndicCorpV2",
    name="indiccorp_v2",
    split="hin_Deva",
    streaming=True
)

tokenized_sentences = []
total_words = 0
total_characters = 0
unique_tokens = set()

for idx, example in enumerate(dataset):
    if MAX_PARAGRAPHS is not None and idx >= MAX_PARAGRAPHS:
        break

    text = example.get("text", "")
    if not text.strip():
        continue

    for sentence in sentence_tokenize(text):
        tokens = word_tokenize(sentence)
        if not tokens:
            continue

        tokenized_sentences.append(" ".join(tokens))
        total_words += len(tokens)
        for token in tokens:
            total_characters += len(token)
            unique_tokens.add(token)

total_sentences = len(tokenized_sentences)

# Save tokenized data (Parquet, compressed)
df = pd.DataFrame({"tokenized_sentence": tokenized_sentences})
df.to_parquet("tokenized_hindi.parquet", compression="snappy", index=False)


# Compute and save statistics
avg_sentence_length = total_words / total_sentences if total_sentences else 0
avg_word_length = total_characters / total_words if total_words else 0
ttr = len(unique_tokens) / total_words if total_words else 0

stats = {
    "total_sentences": total_sentences,
    "total_words": total_words,
    "total_characters": total_characters,
    "average_sentence_length": round(avg_sentence_length, 2),
    "average_word_length": round(avg_word_length, 2),
    "unique_tokens": len(unique_tokens),
    "type_token_ratio": round(ttr, 4),
}

with open("stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("=" * 40)
print("Corpus Statistics")
print("=" * 40)
for key, value in stats.items():
    print(f"{key:25}: {value}")
print("=" * 40)
print("Saved tokenized data -> tokenized_hindi.parquet")
print("Saved statistics     -> stats.json")