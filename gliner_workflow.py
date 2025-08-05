import threading
import pandas as pd
import json
import time
from gliner_testing import extract_entities_no_spacy,extract_entities, descriptive_finder, get_fake_value, entity_to_fake
import os
import time
import sys
sys.path.append("..")
from path import GLINER_PATH,FILE_PATH

@staticmethod
def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'\n⏳ Execution time {func.__name__}: {end-start:.6f} seconds')
        return result
    return wrapper
@time_it
def process_descriptive_data(file_path, output_path):
    print("[BG] Background descriptive data processing started.")
    df = pd.read_excel(file_path)
    descriptive_columns = descriptive_finder(file_path)
    descriptive_data = df[descriptive_columns].fillna('').astype(str).values.ravel()[:10]
    string = ' '.join(descriptive_data)
    # Simulate long processing

    entities = extract_entities(string)
    for entity in entities:
        label = entity['label'].lower()
        if label in ["name of a person", "organization"]:
            get_fake_value(label, entity['text'])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entity_to_fake, f, ensure_ascii=False, indent=4)
    print(f"[BG] Descriptive data mapping complete. Output: {output_path}")

# function print n number
def print_n_numbers(n):
    for i in range(n):
        print(i + 1, end=' ')
        time.sleep(1)  # Simulate some processing delay
    print()
    print("completed")  # New line after printing numbers

def descriptive_finder_df(df):
    des = []
    for col in df.columns:
        series = df[col].dropna().astype(str)
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        if len(series) == 0:
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
@time_it
def process_sql_dataframe(df, output_path):
    descriptive_columns = descriptive_finder_df(df)
    print(f"Descriptive columns found: {descriptive_columns}")
    descriptive_data = df[descriptive_columns].fillna('').astype(str).values.ravel()[:20]
    string = ' '.join(descriptive_data)
    entities = extract_entities(string)
    for entity in entities:
        label = entity['label'].lower()
        if label in ["name of a person", "organization"]:
            get_fake_value(label, entity['text'])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entity_to_fake, f, ensure_ascii=False, indent=4)
    print(f"SQL DataFrame mapping complete. Output: {output_path}")
    # time.sleep(35)  # Simulate processing time

def start_background_descriptive(file_path, output_path):
    thread = threading.Thread(target=process_descriptive_data, args=(file_path, output_path), daemon=False)
    thread.start()
    print("Started background descriptive data processing.")

    return thread


# Call this at the start of your workflow or before model loading

    # ...existing main code...

# Example usage:
if __name__ == "__main__":

    print("[MAIN] Starting background thread for descriptive data...")
    bg_thread = start_background_descriptive(FILE_PATH, "descriptive_mapping_background.json")

    print("[MAIN] Main thread continues. Simulating SQL DataFrame processing...")
    sql_df = pd.read_excel(FILE_PATH, nrows=5)
    print("[MAIN] Processing SQL DataFrame...")
    process_sql_dataframe(sql_df, "sql_query_mapping.json")
    print("[MAIN] SQL DataFrame processing done. Main thread can do other work.")
    # Uncomment to wait for background thread to finish:
    bg_thread.join()
    print("[MAIN] Main thread finished.")
