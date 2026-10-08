"""单进程、内存会话服务：uvicorn app:app --reload。"""
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date
from threading import Lock
from uuid import uuid4
from pathlib import Path
import json
import logging
import os

import anthropic
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from contextlib import closing
from time import perf_counter, sleep

from agent import Conversation

DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "1.0"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("kb_assistant")
router = APIRouter(prefix="/api")

class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    
    message: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    answer: str


@dataclass
class Session:
    conversation: Conversation
    lock: Lock = field(default_factory=Lock)


sessions: dict[str, Session] = {}
sessions_lock = Lock()
daily_budget_lock = Lock()
daily_budget_date = date.today()
daily_spent_usd = 0.0


def check_daily_budget() -> None:
    global daily_budget_date, daily_spent_usd
    with daily_budget_lock:
        today = date.today()
        if today != daily_budget_date:
            daily_budget_date = today
            daily_spent_usd = 0.0
        if daily_spent_usd >= DAILY_BUDGET_USD:
            raise HTTPException(status_code=429, detail="今日额度已用完，明天再来")


def record_daily_cost(cost: float) -> None:
    global daily_budget_date, daily_spent_usd
    with daily_budget_lock:
        today = date.today()
        if today != daily_budget_date:
            daily_budget_date = today
            daily_spent_usd = 0.0
        daily_spent_usd += cost


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for session in sessions.values():
        session.conversation.close()
    sessions.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")

@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    # 同步 send 使用普通 def 路由，由 FastAPI 在线程池中执行。
    check_daily_budget()
    with sessions_lock:
        if body.session_id is None:
            session_id = str(uuid4())
            session = Session(Conversation(prompt_version="v3"))
            sessions[session_id] = session
        else:
            session_id = body.session_id
            existing = sessions.get(session_id)
            if existing is None:
                raise HTTPException(status_code=404, detail="session_id 不存在")
            session = existing

    if not session.lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="该会话正在处理另一个请求")
    start_cost = session.conversation.total_cost
    try:
        answer = session.conversation.send(body.message)
        return ChatResponse(session_id=session_id, answer=answer)
    except (anthropic.APIError, RuntimeError) as exc:
        # agent.Conversation 自己保存本轮记录；这里将错误转换成 HTTP 响应。
        raise HTTPException(
            status_code=502,
            detail={"session_id": session_id, "message": "服务器发生错误 稍后重试"},
        ) from exc
    finally:
        record_daily_cost(session.conversation.total_cost - start_cost)
        session.lock.release()

@router.post("/chat/stream")
def chat_stream(body: ChatRequest) -> StreamingResponse:
    check_daily_budget()
    with sessions_lock:
        if body.session_id is None:
            session_id = str(uuid4())
            session = Session(Conversation(prompt_version="v3"))
            sessions[session_id] = session
        else:
            session_id = body.session_id
            existing = sessions.get(session_id)
            if existing is None:
                raise HTTPException(status_code=404, detail="session_id 不存在")
            session = existing
 
    if not session.lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="该会话正在处理另一个请求")
    
    def event_source():
        start_time = perf_counter()
        status = "cancelled"
        start_cost = session.conversation.total_cost
        tools_used: list[str] = []
        try:
            with closing(
                session.conversation.send_stream(body.message)
            ) as events:
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
                # session.conversation.total_cost
                for event in events:
                    if isinstance(event, dict):
                        if event["type"] == "tool":
                            tools_used.append(event["name"])
                        if event["type"] == "done":
                            status = "ok"
            
                    yield (
                        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    )
                
        except (anthropic.APIError, RuntimeError) as exc:
            status = "error"
            logger.exception("stream error session=%s", session_id)
            # print(f"[stream error] session={session_id} {exc!r}")   # M4.3 会改成日志
            yield f"data: {json.dumps({'type': 'error', 'message': '服务器发生错误，稍后重试'}, ensure_ascii=False)}\n\n"
        finally:
            elapsed = perf_counter() - start_time
            cost = session.conversation.total_cost - start_cost
            record_daily_cost(cost)
            logger.info(
                "request session=%s status=%s takes=%s cost=%s total_cost=%s tool_used=%s",
                session_id, status, f"{elapsed:.3f}秒", f"${cost}", f"${session.conversation.total_cost}", ",".join(t for t in tools_used))
            session.lock.release()
        
    return StreamingResponse(event_source(), media_type="text/event-stream")  # ③

app.include_router(router)
