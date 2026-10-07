from dataclasses import dataclass

@dataclass(frozen=True)
class Doc:
    source: str
    text: str
    
@dataclass(frozen=True)
class Heading:
    line_no: int
    level: int
    title: str
    
@dataclass(frozen=True)
class Chunk:
    source: str
    heading_path: str
    text: str