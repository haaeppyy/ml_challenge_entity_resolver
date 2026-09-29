import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "dataset")
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
OUTPUT_DIR = os.path.join(ROOT, "output")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def source_path(split: str, source: int) -> str:
    return os.path.join(DATASET_DIR, split, f"{split}_source{source}.tsv")


def ground_truth_path(split: str) -> str:
    return os.path.join(DATASET_DIR, split, f"{split}_ground_truth.tsv")


def train_source_path(source: int) -> str:
    return source_path("train", source)


def test_source_path(source: int) -> str:
    return source_path("test", source)