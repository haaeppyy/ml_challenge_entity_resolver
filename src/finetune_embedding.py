import os
import sys
import time
import math

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import polars as pl
from config import DATA_DIR

def load_training_data():
    """Load contrastive training pairs from parquet."""
    print("Loading training data...")
    df = pl.read_parquet(os.path.join(DATA_DIR, "contrastive_train_pairs.parquet"))
    print(f"  Total pairs: {df.height:,}")
    print(f"  Positive: {df.filter(pl.col('label') == 1).height:,}")
    print(f"  Negative: {df.filter(pl.col('label') == 0).height:,}")
    # Filter out null/empty texts
    df = df.filter(
        pl.col("text_l").is_not_null() & (pl.col("text_l").str.len_chars() > 0) &
        pl.col("text_r").is_not_null() & (pl.col("text_r").str.len_chars() > 0)
    )
    print(f"  After filtering null/empty: {df.height:,}")
    return df


def create_examples(df, max_examples=None):
    """Convert to sentence-transformers InputExamples."""
    print("Creating InputExamples...")
    texts_l = df["text_l"].to_list()
    texts_r = df["text_r"].to_list()
    labels = df["label"].to_list()
    
    if max_examples:
        texts_l = texts_l[:max_examples]
        texts_r = texts_r[:max_examples]
        labels = labels[:max_examples]
    
    examples = []
    for t1, t2, label in zip(texts_l, texts_r, labels):
        examples.append(InputExample(texts=[t1, t2], label=float(label)))
    
    print(f"Created {len(examples):,} examples")
    return examples


def main():
    print("=== Fine-tuning Sentence Transformer ===")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU available, training will be slow on CPU")
    
    # Load data
    df = load_training_data()
    
    # Use a subset for initial training (too much data for CPU)
    # We'll use 1M examples (balanced)
    pos_df = df.filter(pl.col("label") == 1)
    neg_df = df.filter(pl.col("label") == 0)
    
    # Sample equal positives and negatives
    sample_size = min(1_000_000, len(pos_df))
    print(f"Sampling {sample_size:,} positives and {sample_size:,} negatives...")
    pos_sample = pos_df.sample(sample_size, seed=42)
    neg_sample = neg_df.sample(sample_size, seed=42)
    train_df = pl.concat([pos_sample, neg_sample])
    print(f"Training pairs: {train_df.height:,}")
    
    # Create examples
    examples = create_examples(train_df)
    
    # Load base model
    print("Loading base model: all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Model max_seq_length: {model.max_seq_length}")
    
    # Create DataLoader
    batch_size = 64
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    
    # Loss function: ContrastiveLoss for pair classification
    train_loss = losses.ContrastiveLoss(model)
    
    # Training parameters
    epochs = 3
    warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)
    output_path = os.path.join(os.path.dirname(DATA_DIR), "models", "finetuned_minilm_v1")
    
    print(f"Training for {epochs} epochs, batch_size={batch_size}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Output: {output_path}")
    
    # Train
    t0 = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        checkpoint_path=os.path.join(output_path, "checkpoints"),
        checkpoint_save_steps=5000,
    )
    
    print(f"Training completed in {time.time()-t0:.1f}s")
    print(f"Model saved to {output_path}")
    
    # Save final model
    final_path = os.path.join(os.path.dirname(DATA_DIR), "models", "finetuned_minilm_final")
    model.save(final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()