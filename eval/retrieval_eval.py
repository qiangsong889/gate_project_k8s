"""运行：python eval/retrieval_eval.py（需要已入库的笔记和数据库连接配置）。"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from pprint import pprint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = Path(__file__).with_name("questions.jsonl")


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    questions = []
    seen_ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "category", "question"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"第 {line_number} 行的 {key} 必须是非空字符串")
        gold = row.get("gold_sources")
        if not isinstance(gold, list) or not gold or not all(
            isinstance(source, str) and source.strip() for source in gold
        ):
            raise ValueError(f"第 {line_number} 行必须有非空 gold_sources 列表")
        if row["id"] in seen_ids:
            raise ValueError(f"题号重复：{row['id']}")
        seen_ids.add(row["id"])
        questions.append(row)
    if not questions:
        raise ValueError("评估题目为空")
    return questions


def calculate_metrics(sources: list[str], gold_sources: list[str], k: int) -> dict[str, float]:
    """recall 按来源去重；precision 按相关 chunk 数计算，分母固定为 k。"""
    if k <= 0:
        raise ValueError("k 必须大于 0")
    sources = sources[:k]
    gold = set(gold_sources)
    if not gold:
        raise ValueError("gold_sources 不能为空")
    matched = set(sources) & gold
    relevant_count = sum(source in gold for source in sources)
    return {
        "hit": float(bool(matched)),
        "recall": len(matched) / len(gold),
        "precision": relevant_count / k,
    }


def evaluate(questions: list[dict], search) -> list[dict]:
    results = []
    for question in questions:
        result = {"id": question["id"], "category": question["category"], "scores": {}, "sources": {}}
        for k in (3, 5):
            hits = search(question["question"], k=k)
            sources = [hit[0] for hit in hits[:k]]
            result["sources"][k] = sources
            result["scores"][k] = calculate_metrics(sources, question["gold_sources"], k)
        results.append(result)
    return results


def print_report(results: list[dict]) -> None:
    print(f"print_report run results=\n")
    pprint(results, width=100, sort_dicts=False)
    groups = defaultdict(list)
    for result in results:
        groups[result["category"]].append(result)

    print("检索器基线：每题直接用用户原问题搜索，不改写查询、不多轮检索。")
    print("hit：至少命中一篇；recall：命中不同来源数 / gold 来源数；precision：相关 chunk 数 / k。")
    print("总体按所有题目平均；k 表示检索片段数，重复 source 不会增加 recall。\n")
    print("| 类别 | 题数 | k | hit@k | recall@k | precision@k |")
    print("|---|---:|---:|---:|---:|---:|")
    print(f"[*groups.items(), ('总体', results)] \n")
    pprint([*groups.items(), ("总体", results)], width=300, sort_dicts=False)
    for category, rows in [*groups.items(), ("总体", results)]:
        for k in (3, 5):
            values = [mean(row["scores"][k][metric] for row in rows) for metric in ("hit", "recall", "precision")]
            print(f"| {category} | {len(rows)} | {k} | " + " | ".join(f"{value:.3f}" for value in values) + " |")

    print("\n没命中的题目（hit@5 = 0）：")
    misses = [row for row in results if row["scores"][5]["hit"] == 0]
    if not misses:
        print("无")
    for row in misses:
        print(f"- {row['id']}（{row['category']}）")
        for rank, source in enumerate(row["sources"][5], 1):
            print(f"  {rank}. {source}")
        if not row["sources"][5]:
            print("  未返回任何结果")


def main() -> None:
    questions = load_questions()
    # 直接运行 eval/ 下的脚本时也能导入项目模块。
    sys.path.insert(0, str(PROJECT_ROOT))
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    from tools import search_notes

    print_report(evaluate(questions, search_notes))


if __name__ == "__main__":
    main()
