from agent import Conversation
from pprint import pprint
# from agent_gpt import Conversation
QUESTIONS = [
    "etcd 是干什么的？",
    "它挂了会怎样？",
    "isla 那个双容器项目，nginx 是怎么找到 python 后端的？",
    "那它的镜像推到哪里了？",
    "第 1 个问题搜索时，除了 etcd 本身，结果里还有哪些内容？",
]

def run(keep_last: int | None) -> None:
    conv = Conversation(prompt_version="v3", keep_last=keep_last)
    for i, q in enumerate(QUESTIONS, start=1):
        print(f"\n===== 第 {i} 问：{q}")
        conv.send(q)
    print("LOOP COMPLETED\n\n")
    print(f"keep_last={keep_last}  总花费=${conv.total_cost:.4f}")

    # pprint(conv.messages_send_to_ai, width=100, sort_dicts=False)
    # print(f"conv.message_turns: {conv.message_turns}")

if __name__ == "__main__":
    run(keep_last=None)     # 不精简
    run(keep_last=3)     # 不精简
