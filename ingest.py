import psycopg2
from sentence_transformers import SentenceTransformer
from classes import Heading, Doc, Chunk
from pgvector.psycopg2 import register_vector
from pathlib import Path
from dataclasses import dataclass
import re
import numpy as np
import hashlib
from db import insert_chunks, create_table, count_chunk
from embeds import embed_chunks
from documents import documents


def find_headings(text: str) -> list[Heading]:
    lines = text.splitlines()
    headings: list[Heading] = []
    in_fence: bool = False
    for line_no, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not in_fence:
            m = re.match(r"^(#{1,3}) +(.+)$", line)
            if m is not None:
                level = len(m.group(1))
                title = m.group(2).strip()
                headings.append(Heading(
                    line_no=line_no,
                    level=level,
                    title=title
                ))
    return headings


def chunk_doc(doc: Doc) -> list[Chunk]:
    headings = find_headings(doc.text)
    lines = doc.text.splitlines()
    chunks: list[Chunk] = []
    heading_path_list: list[Heading] = []
    for i, heading in enumerate(headings):
        heading_path_list = [h for h in heading_path_list if h.level < heading.level]
        heading_path_list.append(heading)
        end = headings[i + 1].line_no - 1 if i + 1 < len(headings) else len(lines)
        text = "\n".join(lines[heading.line_no:end]).strip()
        if text:
            chunks.append(Chunk(
                source=doc.source,
                heading_path=" > ".join([h.title for h in heading_path_list]),
                text=text
            ))
    return chunks


def get_all_chunks(docs: list[Doc]) -> list[Chunk]:
    return [chunk for d in docs for chunk in chunk_doc(d)]

def split_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    fence_char = ""
    fence_len = 0

    for line in text.splitlines():
        if fence_char:
            current.append(line)

            # 结束围栏必须同类型，且长度不短于开始围栏
            if re.fullmatch(
                rf" {{0,3}}{re.escape(fence_char)}{{{fence_len},}}[ \t]*",
                line,
            ):
                fence_char = ""

        else:
            fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)

            if fence and not (
                fence.group(1)[0] == "`" and "`" in fence.group(2)
            ):
                fence_char = fence.group(1)[0]
                fence_len = len(fence.group(1))
                current.append(line)

            elif line.strip():
                current.append(line)

            elif current:
                blocks.append("\n".join(current))
                current = []

    if current:
        blocks.append("\n".join(current))

    return blocks

def pack_blocks(blocks: list[str], max_char: int = 709) -> list[str]:
    p_blocks: list[str] = []
    current: str = ""
    for b in blocks:
        if not current:
            current = b
        elif len(current) + 2 + len(b) > max_char:
            p_blocks.append(current)
            current = b
        else:
            current += f"\n\n{b}"
    if current:
        p_blocks.append(current)
    return p_blocks
    
def split_chunk(chunk: Chunk, max_chars: int = 709) -> list[Chunk]:
    if len(chunk.text) <= max_chars:
        return [chunk]
    blocks = split_blocks(chunk.text)
    packed_blocks = pack_blocks(blocks, max_chars)
    
    return [Chunk(
        source=chunk.source,
        heading_path=chunk.heading_path,
        text=b
    ) for b in packed_blocks]

def get_final_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return [chunk for c in chunks for chunk in split_chunk(c)]
    
if __name__ == "__main__":
    create_table()
    chunked_docs = get_all_chunks(documents)
    final_chunks = get_final_chunks(chunked_docs)
    chunks_with_vecs = embed_chunks(final_chunks)
    insert_chunks(chunks_with_vecs)
    print(count_chunk())