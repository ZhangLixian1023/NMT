import json
def load_pairs(file_path):
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                pairs.append((record["src"], record["tgt"]))
    print("Done: loading pairs: " + file_path)
    return pairs