from gliner import GLiNER
import spacy
import time
import json
import re
import gzip
import sys
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
from joblib import Memory
import os
from path import GLINER_PATH, SPACY_PATH

# Cache for repeated computations
cache_dir = './cache'
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, verbose=0)

model_path = GLINER_PATH

# Optimized spaCy pipeline - disable unnecessary components
nlp = spacy.load(SPACY_PATH, 
                 disable=["ner", "parser", "lemmatizer", "attribute_ruler", "senter"])

model_start = time.time()
model = GLiNER.from_pretrained(model_path, local_files_only=True)
model_end = time.time()
print(f"Model loaded in {model_end - model_start:.2f} seconds.")

patterns = {
    'email': r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
    'phone': r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b',
    'id': r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'
}

labels = ["name of a person", "organization", "IT service"]

# Load fake data once
with gzip.open("faker_dataset_v3.json.gz", "rt", encoding="utf-8") as f:
    fake_list = json.load(f)

fake_data = {}
for d in fake_list:
    fake_data.update(d)

# Convert to sets for faster lookup
fake_data_sets = {
    "names": set(fake_data.get("names", [])),
    "company": set(fake_data.get("company", []))
}

entity_to_fake = {}
used_fakes = {"names": set(), "company": set()}

@memory.cache
def keep_nouns_with_positions_cached(text):
    """Cached version of keep_nouns_with_positions for repeated text"""
    return keep_nouns_with_positions(text)

def keep_nouns_with_positions(text):
    """Optimized to only extract proper nouns"""
    doc = nlp(text)
    noun_tokens = []
    original_positions = []

    for token in doc:
        if token.pos_ == 'PROPN':
            noun_tokens.append(token.text)
            original_positions.append((token.idx, token.idx + len(token.text)))
    
    noun_text = ' '.join(noun_tokens)
    return noun_text, original_positions

def chunk_text_batch(text, max_len=384, overlap_words=3):
    """
    Optimized chunking that prepares for batch processing
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
        
        if chunk_words:
            first_word = chunk_words[0]
            start_char = text.find(first_word, 0 if not chunks else chunks[-1][1] + 1)
        else:
            start_char = 0
            
        chunks.append((chunk_text, start_char))
        
        if end == len(words):
            break
            
        start = end - overlap_words if end - overlap_words > 0 else end
    
    return chunks

def map_positions_batch(ents_batch, noun_positions, offsets):
    """Batch version of position mapping"""
    all_mapped = []
    
    for ents, offset in zip(ents_batch, offsets):
        mapped_ents = []
        for ent in ents:
            ent_start = ent['start']
            ent_end = ent['end']
            
            # Find start position
            start_char = ent_start
            for orig_start, orig_end in noun_positions:
                if orig_start <= ent_start < orig_end:
                    start_char = orig_start + (ent_start - orig_start)
                    break
            
            # Find end position
            end_char = ent_end
            for orig_start, orig_end in noun_positions:
                if orig_start < ent_end <= orig_end:
                    end_char = orig_start + (ent_end - orig_start)
                    break
            
            mapped_ents.append({
                'text': ent['text'],
                'start': start_char + offset,
                'end': end_char + offset,
                'label': ent['label']
            })
        all_mapped.extend(mapped_ents)
    
    return all_mapped

def extract_entities_optimized(text):
    """Optimized entity extraction with batch processing"""
    # Extract regex entities
    regex_entities = []
    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            regex_entities.append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'label': label
            })
    
    # Use cached noun extraction if possible
    try:
        noun_text, noun_positions = keep_nouns_with_positions_cached(text)
    except:
        noun_text, noun_positions = keep_nouns_with_positions(text)
    
    # Prepare chunks for batch processing
    chunks = chunk_text_batch(noun_text, max_len=384, overlap_words=3)
    
    if not chunks:
        return regex_entities
    
    # Batch processing - process all chunks at once
    chunk_texts = [chunk_text for chunk_text, _ in chunks]
    offsets = [offset for _, offset in chunks]
    
    # Single batch prediction call
    try:
        ents_batch = model.predict_entities(chunk_texts, labels, threshold=0.5)
    except:
        # Fallback to individual processing if batch fails
        ents_batch = []
        for chunk_text in chunk_texts:
            ents = model.predict_entities(chunk_text, labels, threshold=0.5)
            ents_batch.append(ents)
    
    # Map positions for all chunks
    all_entities = map_positions_batch(ents_batch, noun_positions, offsets)
    
    # Combine with regex entities
    all_entities.extend(regex_entities)
    all_entities.sort(key=lambda x: x['start'])
    
    return all_entities

def descriptive_finder_optimized(path, sample_size=50):
    """Optimized descriptive finder with sampling"""
    df = pd.read_excel(path, nrows=sample_size)
    descriptive_cols = []
    
    for col in df.columns:
        series = df[col].dropna().astype(str)
        
        # Skip numeric and datetime columns
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
            
        if len(series) == 0:
            continue
        
        # Vectorized operations for better performance
        unique_ratio = series.nunique() / len(series)
        lengths = series.str.len()
        avg_length = lengths.mean()
        word_counts = series.str.split().str.len()
        avg_word_count = word_counts.mean()
        has_punctuation = series.str.contains(r'[.,;:?!]', na=False).mean()
        
        if (unique_ratio > 0.5 and avg_length > 10 and 
            avg_word_count > 3 and has_punctuation > 0.3):
            descriptive_cols.append(col)
    
    return descriptive_cols

def get_fake_value_optimized(label, real_value):
    """Optimized fake value assignment with set-based lookup"""
    if real_value in entity_to_fake:
        return entity_to_fake[real_value]
    
    if label.lower() == "name of a person":
        key = "names"
    elif label.lower() == "organization":
        key = "company"
    else:
        return None
    
    # Use set difference for faster unused fake finding
    available_fakes = fake_data_sets[key] - used_fakes[key]
    
    if available_fakes:
        fake = next(iter(available_fakes))  # Get first available
        used_fakes[key].add(fake)
        entity_to_fake[real_value] = fake
        return fake
    
    return None

def process_text_chunk(text_chunk):
    """Function for parallel processing of text chunks"""
    return extract_entities_optimized(text_chunk)

def parallel_entity_extraction(texts, n_jobs=None):
    """Parallel processing of multiple texts"""
    if n_jobs is None:
        n_jobs = min(cpu_count(), len(texts))
    
    if n_jobs == 1 or len(texts) == 1:
        return [extract_entities_optimized(text) for text in texts]
    
    with Pool(n_jobs) as pool:
        results = pool.map(process_text_chunk, texts)
    
    return results

def main():
    print("Starting optimized GLiNER processing...")
    
    # Load data
    df_1 = pd.read_excel("SO(real).xlsx")
    # result = df_1.to_dict(orient='records')
    # print(f"Loaded {len(result)} rows")
    
    # df = pd.DataFrame(result)
    
    # Find descriptive columns (optimized)
    st = time.time()
    descriptive_columns = descriptive_finder_optimized("SO(real).xlsx")
    et = time.time()
    print(f"Descriptive columns: {descriptive_columns}")
    print(f"Descriptive columns extraction time: {et-st:.2f} seconds")
    
    # Prepare text data
    descriptive_data = df_1[descriptive_columns].fillna('').astype(str).values.ravel()[:100]
    string = ' '.join(descriptive_data)
    
    # Process entities (optimized)
    run_start = time.time()
    entities = extract_entities_optimized(string)
    run_end = time.time()
    
    # Generate fake mappings (optimized)
    mapping_start = time.time()
    for entity in entities:
        label = entity['label'].lower()
        if label in ["name of a person", "organization"]:
            get_fake_value_optimized(label, entity['text'])
    
    # Save mappings
    with open("descriptive_mapping_optimized.json", "w", encoding="utf-8") as f:
        json.dump(entity_to_fake, f, ensure_ascii=False, indent=4)
    
    mapping_end = time.time()
    
    # Print results
    for entity in entities:
        fake_val = entity_to_fake.get(entity['text'], 'N/A')
        print(f"{entity['text']} => {entity['label']} => {fake_val} [{entity['start']}:{entity['end']}]")
    
    print(f"\n=== Performance Report ===")
    print(f"Model loading time: {model_end - model_start:.2f} seconds")
    print(f"Entity extraction time: {run_end - run_start:.2f} seconds")
    print(f"Mapping generation time: {mapping_end - mapping_start:.2f} seconds")
    print(f"Total entities found: {len(entities)}")
    print(f"Unique mappings created: {len(entity_to_fake)}")

if __name__ == "__main__":
    main()
