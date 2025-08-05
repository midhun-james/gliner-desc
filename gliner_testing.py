from gliner import GLiNER
import spacy
import time
import json
import re
import gzip
import sys
import pandas as pd
from path import GLINER_PATH,SPACY_PATH

model_path = GLINER_PATH
nlp = spacy.load(SPACY_PATH)

model_start = time.time()
model = GLiNER.from_pretrained(model_path, local_files_only=True)
model_end = time.time()
print(f"Model loaded in {model_end - model_start} seconds.")

patterns = {
    'email': r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
    'phone': r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b',
    'id': r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'
}

labels = ["name of a person", "organization","IT service"]
with gzip.open("faker_dataset_v3.json.gz", "rt", encoding="utf-8") as f:
    fake_list= json.load(f)
fake_data={}
for d in fake_list:
    fake_data.update(d)

entity_to_fake={}
used_fakes={"names": set(), "company": set()}
def keep_nouns_with_positions(text):
    doc = nlp(text)
    noun_tokens = []
    original_positions = []
    
    for token in doc:
        if token.pos_ == 'PROPN':
            noun_tokens.append(token.text)
            original_positions.append((token.idx, token.idx + len(token.text)))
            print(f'Token inside Noun: {token.text}')
            # f.write(f"{noun_tokens[-1]} : {original_positions[-1]}\n")
    noun_text = ' '.join(noun_tokens)
    print(f"Noun text: {noun_text}")
    return noun_text, original_positions

def chunk_text_no_split(text, max_len=390, overlap_words=3):
    """
    Chunk text into pieces of max_len, ensuring no word is split between chunks.
    Overlap is handled by number of words, not characters.
    Returns a list of (chunk, start_char_index) tuples.
    """
    words = re.findall(r'\S+', text)
    chunks = []
    start = 0
    while start < len(words):
        end = start
        char_count = 0
        chunk_words = []
        while end < len(words) and char_count + len(words[end]) + (1 if chunk_words else 0) <= max_len:
            chunk_words.append(words[end])
            char_count += len(words[end]) + (1 if chunk_words else 0)
            end += 1
        chunk_text = ' '.join(chunk_words)
        # Find the character index in the original text
        if chunk_words:
            first_word = chunk_words[0]
            start_char = text.find(first_word, 0 if not chunks else chunks[-1][1] + 1)
        else:
            start_char = 0
        chunks.append((chunk_text, start_char))
        if end == len(words):
            break
        # Overlap by words
        start = end - overlap_words if end - overlap_words > 0 else end
    return chunks

def map_positions(ents, noun_positions, chunk_offset):
    mapped_ents = []
    for ent in ents:
        ent_start = ent['start']
        ent_end = ent['end']
        start_char = None
        for orig_start, orig_end in noun_positions:
            if orig_start <= ent_start < orig_end:
                start_char = orig_start + (ent_start - orig_start)
                break
        if start_char is None:
            start_char = ent_start
        end_char = None
        for orig_start, orig_end in noun_positions:
            if orig_start < ent_end <= orig_end:
                end_char = orig_start + (ent_end - orig_start)
                break
        if end_char is None:
            end_char = ent_end

        mapped_ents.append({
            'text': ent['text'],
            'start': start_char + chunk_offset,
            'end': end_char + chunk_offset,
            'label': ent['label']
        })
    return mapped_ents

def extract_entities(text):
    # print(f'text is : {text[:100]}')
    regex_entities = []
    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            regex_entities.append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'label': label
            })
    noun_text, noun_positions = keep_nouns_with_positions(text)
    print(f"Noun text: {noun_text[:100]}")
    chunks = chunk_text_no_split(text, max_len=384, overlap_words=3)
    all_entities = []
    for chunk_text_part, offset in chunks:
        # print(f"chunk is :   {len(chunk_text_part)}")
        ents = model.predict_entities(chunk_text_part, labels, threshold=0.5)
        mapped = map_positions(ents, noun_positions, offset)
        all_entities.extend(mapped)
    all_entities.sort(key=lambda x: x['start'])
    return all_entities

# --- Direct entity extraction version (no spacy noun filtering) ---
def extract_entities_no_spacy(text):
    regex_entities = []
    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            regex_entities.append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'label': label
            })
    # Directly use the full text for GLiNER
    chunks = chunk_text_no_split(text, max_len=384, overlap_words=3)
    all_entities = []
    for chunk_text_part, offset in chunks:
        ents = model.predict_entities(chunk_text_part, labels, threshold=0.5)
        # If you want to map positions, you can do so here, but since we're not filtering, just offset
        for ent in ents:
            all_entities.append({
                'text': ent['text'],
                'start': ent['start'] + offset,
                'end': ent['end'] + offset,
                'label': ent['label']
            })
    all_entities.sort(key=lambda x: x['start'])
    return all_entities

def descriptive_finder(path):   
        df =pd.read_excel(path,nrows=20)
        des=[]
        for col in df.columns:
            series = df[col].dropna().astype(str)
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
                continue
            if len(series)==0:
                continue
            else:
                
                unique_ratio = series.nunique() / len(series)
                avg_length = series.apply(len).mean()
                avg_word_count = series.apply(lambda x: len(x.split())).mean()
                has_punctuation = series.str.contains(r'[.,;:?!]').mean()
                if (
                    unique_ratio > 0.5 and
                    avg_length > 10 and
                    avg_word_count > 3 and
                    has_punctuation > 0.3
                ):
                    des.append(col)
        return des

def get_fake_value(label,real_value):
    # key="names" if label.lower()=="name of a person" else "company"
    if label.lower() == "name of a person":
        key = "names"
    elif label.lower() == "organization":
        key = "company"
    pool = fake_data.get(key, [])
    if real_value in entity_to_fake:
        return entity_to_fake[real_value]
    for fake in pool:
        if fake not in used_fakes[key]:
            used_fakes[key].add(fake)
            entity_to_fake[real_value] = fake
            return fake
if __name__ == "__main__":
    

    df = pd.read_excel("SO(real).xlsx")
    st= time.time()
    descriptive_columns= descriptive_finder("SO(real).xlsx")
    # descriptive_columns=["Comments","Reasons"]
    et= time.time()
    print(f"Descriptive columns: {descriptive_columns}")
    descriptive_data = df[descriptive_columns].fillna('').astype(str).values.ravel()[:20]
    string = ' '.join(descriptive_data)
    # print(string[:100])
    run_start = time.time()
    # entities = extract_entities(string)  # <-- original, with spacy noun filtering
    entities = extract_entities(string)  # <-- new, direct to GLiNER
    print(f"Extracted {len(entities)} entities (no spacy filtering).")
    run_end = time.time()
    s=time.time()
    for entity in entities:
        label= entity['label'].lower()
        if label in ["name of a person", "organization"]:
            get_fake_value(label, entity['text'])
    with open("descriptive_mapping_spacy_3.json","w", encoding="utf-8") as f:
        json.dump(entity_to_fake, f, ensure_ascii=False, indent=4)
    e=time.time()
    for entity in entities:
        print(f"{entity['text']} => {entity['label']} [{entity['start']}:{entity['end']}]")

    print(f"Mapping time: {e-s:.2f} seconds")
    print(f"Descriptive columns extraction time: {et-st:.2f} seconds")
    print(f'Model loading time: {model_end - model_start:.2f} seconds')
    print(f'Entity extraction run time: {run_end - run_start:.2f} seconds')


