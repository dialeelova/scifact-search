import zipfile
from pathlib import Path

import requests

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DATA_DIR = Path("data")

def main():
    target = DATA_DIR / "scifact"
    if (target / "corpus.jsonl").exists():
        print("Already downloaded")
        return
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "scifact.zip"
    print("Downloading...")
    r = requests.get(URL, stream=True, timeout=60)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    zip_path.unlink()
    print("Done:", target)

if __name__ == "__main__":
    main()
