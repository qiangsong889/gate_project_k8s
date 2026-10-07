from pathlib import Path
from classes import Doc
import os 

path = Path(os.environ.get("NOTES_DIR", "/Users/shaynesong/randomshit/k8s/k8s-notes-site/src/content"))

def _load_docs(root: Path) -> list[Doc]:
    # 拿到每一个.md的绝对路径
    md_files = sorted(root.rglob("*.md"))
    return [
        Doc(
            source=str(path.relative_to(root).as_posix()),
            text=path.read_text(encoding="utf-8"),
        )
        for path in md_files
    ]

def _list_directories(root: Path) -> list[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir()])
    
documents = _load_docs(path)
directories = _list_directories(path)
