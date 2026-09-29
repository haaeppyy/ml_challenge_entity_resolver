import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl
import faiss
from sentence_transformers import SentenceTransformer
from config import DATA_DIR


def embed_split(model, split, src, batch_size=4096):
    """Embed all entities in a split/source - save in chunks to avoid memory issues."""
    path = os.path.join(DATA_DIR, f"norm_{split}_s{src}.parquet")
    df = pl.read_parquet(path)
    
    # Combine name_norm and addr_norm, handle nulls
    df = df.with_columns(
        (pl.col("name_norm").fill_null("") + " " + pl.col("addr_norm").fill_null("")).alias("text")
    )
    
    texts = df["text"].to_list()
    entity_ids = df["entity_id"].to_list()
    n = len(texts)
    
    print(f"  Embedding {split} S{src}: {n:,} entities...")
    t0 = time.time()
    
    # Process and save in chunks (1M entities per chunk)
    chunk_size = 1_000_000
    chunk_files = []
    
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_texts = texts[chunk_start:chunk_end]
        chunk_ids = entity_ids[chunk_start:chunk_end]
        
        print(f"  Chunk {chunk_start//chunk_size + 1}: {len(chunk_texts):,} entities...")
        chunk_t0 = time.time()
        
        chunk_embeddings = []
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i+batch_size]
            emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            chunk_embeddings.append(emb)
        
        chunk_emb = np.vstack(chunk_embeddings).astype(np.float32)
        print(f"    Chunk done: {chunk_emb.shape} in {time.time()-chunk_t0:.1f}s")
        
        # Save chunk
        chunk_path = os.path.join(DATA_DIR, f"emb_{split}_s{src}_chunk{chunk_start//chunk_size:03d}.npy")
        np.save(chunk_path, chunk_emb)
        chunk_files.append((chunk_path, chunk_ids))
    
    # Save entity IDs
    id_path = os.path.join(DATA_DIR, f"emb_{split}_s{src}_ids.parquet")
    pl.DataFrame({"entity_id": entity_ids}).write_parquet(id_path)
    
    print(f"  Done {split} S{src}: {n:,} entities in {time.time()-t0:.1f}s ({len(chunk_files)} chunks)")
    return chunk_files


def load_embeddings(chunk_files):
    """Load all chunk embeddings into a single array."""
    all_embeddings = []
    all_ids = []
    for chunk_path, chunk_ids in chunk_files:
        emb = np.load(chunk_path)
        all_embeddings.append(emb)
        all_ids.extend(chunk_ids)
    return np.vstack(all_embeddings), all_ids


def build_faiss_index_incremental(embedding_chunks, d, index_type="IVF_FLAT", nlist=8192):
    """Build FAISS index by adding vectors incrementally."""
    print(f"Building FAISS index incrementally: {index_type}, nlist={nlist}, d={d}")
    
    if index_type == "IVF_FLAT":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train on a sample of all data
        print("  Collecting training sample...")
        train_samples = []
        for chunk_path, _ in embedding_chunks:
            emb = np.load(chunk_path)
            train_samples.append(emb)
        train_data = np.vstack(train_samples)
        
        # Use a subset for training (max 1M vectors)
        if train_data.shape[0] > 1_000_000:
            idx = np.random.choice(train_data.shape[0], 1_000_000, replace=False)
            train_data = train_data[idx]
        
        print("  Normalizing training data...")
        faiss.normalize_L2(train_data)
        print("  Training index...")
        index.train(train_data)
        del train_data
        
        # Add all vectors in chunks
        print("  Adding vectors in chunks...")
        for i, (chunk_path, _) in enumerate(embedding_chunks):
            emb = np.load(chunk_path)
            faiss.normalize_L2(emb)
            index.add(emb)
            print(f"    Added chunk {i+1}/{len(embedding_chunks)}: {emb.shape}")
    
    elif index_type == "HNSW":
        index = faiss.IndexHNSWFlat(d, 32)
        index.hnsw.efConstruction = 200
        
        for i, (chunk_path, _) in enumerate(embedding_chunks):
            emb = np.load(chunk_path)
            faiss.normalize_L2(emb)
            index.add(emb)
            print(f"    Added chunk {i+1}/{len(embedding_chunks)}: {emb.shape}")
    
    else:  # Flat
        index = faiss.IndexFlatIP(d)
        for i, (chunk_path, _) in enumerate(embedding_chunks):
            emb = np.load(chunk_path)
            faiss.normalize_L2(emb)
            index.add(emb)
            print(f"    Added chunk {i+1}/{len(embedding_chunks)}: {emb.shape}")
    
    print(f"  Index built: {index.ntotal:,} vectors")
    return index


def search_topk(index, query_embeddings, k=50, ef_search=100):
    """Search top-k neighbors for query embeddings."""
    if hasattr(index, 'hnsw'):
        index.hnsw.efSearch = ef_search
    elif hasattr(index, 'nprobe'):
        index.nprobe = min(128, index.nlist)
    
    faiss.normalize_L2(query_embeddings)
    distances, indices = index.search(query_embeddings, k)
    return distances, indices


def main():
    t0 = time.time()
    print("=== Embedding-based Blocking ===")
    print(f"PyTorch CUDA: {torch.cuda.is_available()}")
    
    # Load model
    model_path = os.path.join(os.path.dirname(DATA_DIR), "models", "finetuned_minilm_5000steps")
    if not os.path.exists(model_path):
        model_path = "all-MiniLM-L6-v2"
    print(f"Loading model: {model_path}")
    model = SentenceTransformer(model_path)
    print(f"Model loaded, embedding dim: {model.get_sentence_embedding_dimension()}")
    
    # Embed all test sources (for inference)
    test_chunks = {}
    for src in [1, 2, 3]:
        test_chunks[src] = embed_split(model, "test", src)
    
    # Embed all train sources (for validation if needed)
    train_chunks = {}
    for src in [1, 2, 3]:
        train_chunks[src] = embed_split(model, "train", src)
    
    # Build FAISS index on S2 + S3 test embeddings
    print("\nBuilding FAISS index on test S2 + S3...")
    
    # Load S2 test chunks
    s2_test_chunks = test_chunks[2]
    s3_test_chunks = test_chunks[3]
    
    # Get combined IDs
    s2_ids = pl.read_parquet(os.path.join(DATA_DIR, "emb_test_s2_ids.parquet"))["entity_id"].to_list()
    s3_ids = pl.read_parquet(os.path.join(DATA_DIR, "emb_test_s3_ids.parquet"))["entity_id"].to_list()
    combined_ids = s2_ids + s3_ids
    
    # Build index on S2 + S3
    all_test_chunks = s2_test_chunks + s3_test_chunks
    index = build_faiss_index_incremental(all_test_chunks, d=384, index_type="IVF_FLAT", nlist=8192)
    
    # Save index
    index_path = os.path.join(DATA_DIR, "faiss_index_test_s2s3.index")
    faiss.write_index(index, index_path)
    print(f"Index saved to {index_path}")
    
    # Save combined IDs
    pl.DataFrame({"entity_id": combined_ids}).write_parquet(
        os.path.join(DATA_DIR, "faiss_index_test_s2s3_ids.parquet")
    )
    
    # Search for each S1 test entity
    print("\nSearching top-50 neighbors for test S1...")
    s1_test_chunks = test_chunks[1]
    s1_emb, s1_ids = load_embeddings(s1_test_chunks)
    print(f"S1 test embeddings: {s1_emb.shape}")
    
    distances, indices = search_topk(index, s1_emb, k=50)
    
    # Save results
    np.save(os.path.join(DATA_DIR, "faiss_distances_test.npy"), distances)
    np.save(os.path.join(DATA_DIR, "faiss_indices_test.npy"), indices)
    print(f"Distances: {distances.shape}, Indices: {indices.shape}")
    
    # Create candidate pairs from FAISS results
    print("Creating candidate pairs from FAISS...")
    all_pairs = []
    for i, s1_id in enumerate(s1_ids):
        neighbor_ids = [combined_ids[idx] for idx in indices[i]]
        all_pairs.append((s1_id, neighbor_ids))
    
    # Save as parquet
    pair_df = pl.DataFrame({
        "entity_id_l": s1_ids,
        "entity_id_r": [",".join(ids) for ids in [p[1] for p in all_pairs]]
    })
    pair_df.write_parquet(os.path.join(DATA_DIR, "cand_faiss_test.parquet"))
    print(f"FAISS candidates saved: {pair_df.height:,} entities")
    
    # Also do for train (for validation)
    print("\nProcessing train split...")
    
    # Load train chunks
    s1_train_chunks = train_chunks[1]
    s2_train_chunks = train_chunks[2]
    s3_train_chunks = train_chunks[3]
    
    s1_ids_tr = pl.read_parquet(os.path.join(DATA_DIR, "emb_train_s1_ids.parquet"))["entity_id"].to_list()
    s2_ids_tr = pl.read_parquet(os.path.join(DATA_DIR, "emb_train_s2_ids.parquet"))["entity_id"].to_list()
    s3_ids_tr = pl.read_parquet(os.path.join(DATA_DIR, "emb_train_s3_ids.parquet"))["entity_id"].to_list()
    combined_ids_tr = s2_ids_tr + s3_ids_tr
    
    # Build train index
    all_train_chunks = s2_train_chunks + s3_train_chunks
    index_tr = build_faiss_index_incremental(all_train_chunks, d=384, index_type="IVF_FLAT", nlist=8192)
    
    faiss.write_index(index_tr, os.path.join(DATA_DIR, "faiss_index_train_s2s3.index"))
    pl.DataFrame({"entity_id": combined_ids_tr}).write_parquet(
        os.path.join(DATA_DIR, "faiss_index_train_s2s3_ids.parquet")
    )
    
    # Search train S1
    s1_emb_tr, s1_ids_tr = load_embeddings(s1_train_chunks)
    distances_tr, indices_tr = search_topk(index_tr, s1_emb_tr, k=50)
    
    np.save(os.path.join(DATA_DIR, "faiss_distances_train.npy"), distances_tr)
    np.save(os.path.join(DATA_DIR, "faiss_indices_train.npy"), indices_tr)
    
    pair_df_tr = pl.DataFrame({
        "entity_id_l": s1_ids_tr,
        "entity_id_r": [",".join([combined_ids_tr[idx] for idx in indices_tr[i]]) for i in range(len(s1_ids_tr))]
    })
    pair_df_tr.write_parquet(os.path.join(DATA_DIR, "cand_faiss_train.parquet"))
    print(f"Train FAISS candidates saved: {pair_df_tr.height:,} entities")
    
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()