from documents import documents
import psycopg2
from db import connect_db
from anthropic.types import ToolParam, MessageParam
from embeds import embed_query
import anthropic
from dotenv import load_dotenv
import json
from prompts import get_prompt
from collections.abc import Callable
from documents import directories

tools: list[ToolParam] = [
    {
        "name": "read_note",
        "description": "按 source 路径读取整篇笔记原文，用于检索结果不完整、需要完整上下文时",
        "input_schema": {
            "type": "object",
            "properties": {
              "source": {
                  "type": "string",
                  "description": "笔记的相对路径，例如 basics/07-minikube.md，与 search_notes 返回的 source 一致"
              }
            },
            "required": ["source"]
        },
    },
    {
        "name": "search_notes",
        "description": "按语义搜索笔记片段，用于定位相关内容",
        "input_schema": {
            "type": "object",
            "properties": {
              "query": {
                  "type": "string",
                  "description": "要搜索的内容，用简短的自然语言或关键词描述，例如「etcd 的作用」。"
                },
              "category": {
                "type": "string",
                "enum": directories,
                "description": 
                    "可选。只在某个目录里搜索时使用。不确定问题属于哪个目录时不要填，搜索全部笔记。\n"
                    "各目录内容：\n"
                    "- basics：基础操作速查，macOS + minikube 上的 Docker/kubectl 实战（create vs apply、Pod、DNS、Service 类型、Namespace、踩坑与待学清单）\n"
                    "- k3s-deploy：Hostinger VPS 上部署 k3s 的全过程（硬件选型、系统准备、网络基础、PKI 证书、远程 kubectl、TLS 深挖）\n"
                    "- auth-flows：三种认证流程对比（kubectl 的 mTLS、浏览器 HTTPS 的 TLS、SSH 登录）\n"
                    "- app-practice：k8s-notes 静态站点项目的部署实践（镜像构建推送、Deployment、Ingress、cert-manager）\n"
                    "- app-concepts：部署中顺带搞懂的 K8s 原理（资源/对象/组件分类、Node 与 Controller、kube-proxy 转发、术语词典）\n"
                    "- app-pitfalls：部署过程中踩过的坑和排查思路（Headlamp Ingress 冲突、AAAA 记录导致证书报错）\n"
                    "- two-container-app：isla 一周岁照片站，nginx + python 双容器项目（私有镜像、PVC、BasicAuth）\n"
              }
            },
            "required": ["query"]
        },
    },
]


def read_note(source: str) -> str:
    doc = next(
        (d for d in documents if d.source == source),
        None,
    )
    if doc is not None:
        return doc.text
    supported_sources = "\n".join(
        f"- {s}" for s in sorted({d.source for d in documents})
    )
    return (   
        f"未找到 source：{source}\n"
        f"支持的 source 如下：\n{supported_sources}"
    )

def search_notes_for_agent(query: str, category: str = "") -> str:
    category = category.strip().rstrip("/")
    if category and category not in directories:
        return f"不存在的目录: {category}\n可用的目录: {', '.join(directories)}"
    return "\n\n".join(
        f"{source} {heading_path}\n{content}"
        for source, heading_path, content, distance in search_notes(query, category)
)

def search_notes(query: str, category: str = "", k: int = 3) -> list[tuple[str, str, str, float]]:
    conn = None
    query_vec = embed_query(query)
    params: list[object] = [query_vec]
    sql = """
        SELECT source, heading_path, content, embedding <=> %s AS distance FROM kb_chunks
    """
    if category:
        sql += " WHERE source LIKE %s"
        params.append(f"{category}/%")
        
    sql += " ORDER BY distance LIMIT %s"
    params.append(k)
    
    conn = connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    params
                )
                result = cur.fetchall()
                return result
    except psycopg2.OperationalError as e:
        print(f"数据库连接或运行异常：{e}")
        raise

    except psycopg2.IntegrityError as e:
        print(f"违反表的约束，例如主键重复、必填字段为空：{e}")
        raise

    except psycopg2.ProgrammingError as e:
        print(f"SQL、表结构或参数类型错误：{e}")
        raise

    except psycopg2.Error as e:
        print(f"其他数据库错误：{e}")
        raise
    finally:
        if conn is not None:
            conn.close()

TOOL_FUNCS: dict[str, Callable[..., str]] = {
    "read_note": read_note,
    "search_notes": search_notes_for_agent,
}
                            
        
if __name__ == "__main__":
    print("=== 不过滤 ===")
    print(search_notes_for_agent("镜像推送"))
    print("\n=== 只搜 two-container-app ===")
    print(search_notes_for_agent("镜像推送", category="two-container-app"))
    print("\n=== 不存在的目录 ===")
    print(search_notes_for_agent("镜像推送", category="isla"))
