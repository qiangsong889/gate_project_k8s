from dataclasses import dataclass

@dataclass(frozen=True)
class PromptVersion:
    name: str          # 这个 prompt 是干什么的
    version: str       # "v1", "v2"...
    template: str       # 实际的 prompt 文本
    model: str          # 这个版本是针对哪个模型调的（模型换了，prompt 效果可能变）
    changelog: str      # 这一版相比上一版改了什么、为什么改

PROMPTS: dict[str, list[PromptVersion]] = {
    "knowledge_search": [
        PromptVersion(
            name="knowledge_search", version="v1", model="claude-sonnet-5",
            template=(
                "回答只能依据工具返回的笔记内容，标注出处（source > heading_path）：\n"
                "笔记里找不到就说找不到，不要用自己的知识补\n"
                "第一次搜索结果不相关时，可以换个说法再搜；报错类问题同时用英文报错原文和中文描述搜\n"
                "片段不完整时用 read_note 读全文\n"
            ),
            changelog="初始版本",
        ),
        PromptVersion(
            name="knowledge_search", version="v2", model="claude-sonnet-5",
            template=(
                "回答只能依据工具返回的笔记内容，标注出处（source > heading_path）：\n"
                "查询时用自己的知识猜原因，完全没问题；最终回答时才必须只依据笔记\n"
                "笔记里找不到就说找不到，不要用自己的知识补\n"
                "因为检索系统对英文报错名和中文解释的匹配很弱 你要根据自己的理解 对于报错类问题同时用英文报错原文和中文描述搜 例如“exec format error” 要同时搜报错原文，和你猜的中文原因‘镜像架构不匹配’\n"
                "片段不完整时用 read_note 读全文\n"
            ),
            changelog="对报错类的问题 让模型根据自己的理解 同时检索中文版的错误",
        ),
        PromptVersion(
            name="knowledge_search", version="v3", model="claude-sonnet-5",
            template=(
                "回答只能依据工具返回的笔记内容，标注出处（source > heading_path）：\n"
                "写查询时可以用你的知识猜原因和关键词。\n"
                "写回答时，每一句事实都要来自工具结果并标出处；笔记里没有的内容不要写。如果确实需要补充背景，必须单独一段，开头写明'以下不是笔记内容'。\n"
                "因为检索系统对英文报错名和中文解释的匹配很弱 你要根据自己的理解 对于报错类问题同时用英文报错原文和中文描述搜 例如“exec format error” 要同时搜报错原文，和你猜的中文原因‘镜像架构不匹配’\n"
                "第一次搜索没结果时换个说法重搜\n"
                "片段不完整时用 read_note 读全文\n"
            ),
            changelog="强调查询和回答的规则 避免模型回答时使用自己的知识 但是可以适当补充",
        )
    ]
}

def get_prompt(name: str, version: str) -> PromptVersion:
    for p in PROMPTS[name]:
        if p.version == version:
            return p
    raise ValueError(f"找不到 {name} 的 {version} 版本")
