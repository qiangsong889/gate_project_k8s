from __future__ import annotations

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ingest import Chunk
    
def embed_query(query: str) -> np.ndarray:
    return model.encode(query)

def embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, np.ndarray]]:
    chunk_text = [f"{c.heading_path}\n{c.text}" for c in chunks]
    embedings = model.encode(chunk_text)
    return list(zip(chunks, embedings))

def get_text_token_ratio(chunk: Chunk) -> float:
    n_tokens = len(model.tokenizer(chunk.text)["input_ids"])
    ratio = len(chunk.text) / n_tokens        # 平均每个 token 对应多少个字符
    return ratio

def get_heading_token_len(chunk: Chunk) -> float:
    n_tokens = len(model.tokenizer(chunk.heading_path)["input_ids"])
    return n_tokens