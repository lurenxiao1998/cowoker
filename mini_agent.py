import getpass
import os
import subprocess
from pathlib import Path
from typing import Literal, Optional

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

from typing_extensions import Annotated, TypedDict
import operator
import json
from langgraph.graph import StateGraph, START, END



# =========================
# 0) API Key
# =========================
def get_openai_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


get_openai_api_key()


# =========================
# 1) Model
# =========================
model = init_chat_model(
    "gpt-5-nano",
    model_provider="openai",
    temperature=0.2,  # coding 更稳一点
)

# =========================
# 2) Workspace config
# =========================
# 让 agent 在这个目录里工作：默认当前目录
REPO_ROOT = Path(os.environ.get("AGENT_REPO_ROOT", ".")).resolve()

# 安全限制：只允许在 repo 内读写
def _safe_path(path: str) -> Path:
    p = (REPO_ROOT / path).resolve()
    if REPO_ROOT not in p.parents and p != REPO_ROOT:
        raise ValueError(f"Path escapes repo root: {path}")
    return p


def _read_text(p: Path, max_chars: int = 12000) -> str:
    # 防止一次塞太多 token：截断
    data = p.read_text(encoding="utf-8", errors="replace")
    if len(data) > max_chars:
        head = data[: max_chars // 2]
        tail = data[-max_chars // 2 :]
        return head + "\n\n...<TRUNCATED>...\n\n" + tail
    return data


# =========================
# 3) Tools (Coding)
# =========================
@tool
def list_files(glob: str = "**/*", max_files: int = 300) -> str:
    """List files under repo root (relative paths). Use glob like '**/*.py'."""
    files = []
    for p in REPO_ROOT.glob(glob):
        if p.is_file():
            rel = p.relative_to(REPO_ROOT).as_posix()
            files.append(rel)
            if len(files) >= max_files:
                break
    files.sort()
    return "\n".join(files)


@tool
def read_file(path: str) -> str:
    """Read a text file under repo root. Returns (possibly truncated) content."""
    p = _safe_path(path)
    if not p.exists() or not p.is_file():
        return f"NOT_FOUND: {path} (file does not exist under repo root)"
    return _read_text(p)


@tool
def write_file(path: str, content: str) -> str:
    """Write text content to a file under repo root. Creates parent dirs if needed."""
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"WROTE {path} ({len(content)} chars)"


@tool
def run_cmd(cmd: str, timeout_s: int = 60) -> str:
    """Run a shell command in repo root and return stdout/stderr (truncated)."""
    # 注意：这是强能力工具。学习阶段 ok；生产要加 allowlist。
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if len(out) > 12000:
        out = out[:6000] + "\n\n...<TRUNCATED>...\n\n" + out[-6000:]
    return f"exit_code={proc.returncode}\n{out}"


tools = [list_files, read_file, write_file, run_cmd]
tools_by_name = {t.name: t for t in tools}

model_with_tools = model.bind_tools(tools)


# =========================
# 4) State schema
# =========================
class ActionRecord(TypedDict):
    name: str
    args: dict
    tool_call_id: str


class ObservationRecord(TypedDict):
    tool_call_id: str
    output: str


class ContextState(TypedDict):
    # task control
    # goal: str
    # plan: str
    # constraints: str
    # done_criteria: list[str]

    # loop control
    steps: int
    # status: str  # running | success | fail
    # last_error: str

    # core traces
    messages: Annotated[list[AnyMessage], operator.add]
    # actions: Annotated[list[ActionRecord], operator.add]
    # observations: Annotated[list[ObservationRecord], operator.add]

    # optional workspace hints
    # focus_files: list[str]


MAX_STEPS = 20


# =========================
# 5) Nodes
# =========================
CODING_SYSTEM_PROMPT = """You are a coding agent working inside a local repository.

Rules:
- Use tools to inspect the repository (list_files, read_file) before changing code.
- When you change code, use write_file.
- Validate with run_cmd (e.g., run tests) when appropriate.
- Keep changes minimal and respect constraints.
- Be explicit about next actions via tool calls rather than long explanations.
- Stop when done_criteria are satisfied; then summarize changes briefly.

You must not invent file contents. Always read files you need via tools.
"""

# 估算值：1 token ≈ 4 chars
# 128k context ≈ 500,000 chars
# 为了避免上下文过长导致超时和天价账单，添加熔断机制
# 安全起见，设置阈值为 200,000 chars (约 50k tokens)
MAX_CONTEXT_CHARS = 200000 
def llm_call(state: dict):
    """LLM decides next step and may call tools."""
    if state.get("steps", 0) >= MAX_STEPS:
        return {"status": "fail", "last_error": f"Reached MAX_STEPS={MAX_STEPS}"}

    total_chars = sum(len(msg.content) for msg in state["messages"])
    if total_chars > MAX_CONTEXT_CHARS:
        return {"status": "fail", "last_error": f"Context overflow! Total chars: {total_chars} > {MAX_CONTEXT_CHARS}. Please implement context compression"}

    # 给模型一段简短 task header（结构化 state -> prompt 视图）
#     header = f"""GOAL: {state.get('goal','')}
# CONSTRAINTS: {state.get('constraints','')}
# DONE_CRITERIA: {state.get('done_criteria', [])}
# CURRENT_PLAN: {state.get('plan','')}
# FOCUS_FILES: {state.get('focus_files', [])}
# """


    msg = model_with_tools.invoke(
        [SystemMessage(content=CODING_SYSTEM_PROMPT)]
        + state["messages"]
    )

    return {
        "messages": [msg],
        "steps": state.get("steps", 0) + 1,
        # "status": state.get("status", "running"),
    }


def tool_node(state: dict):
    """Execute tool calls from the last assistant message."""
    last = state["messages"][-1]
    actions: list[ActionRecord] = []
    observations: list[ObservationRecord] = []
    out_messages: list[AnyMessage] = []

    for tc in last.tool_calls:
        tool_fn = tools_by_name[tc["name"]]
        raw = tool_fn.invoke(tc["args"])
        raw_str = str(raw)

        # 给 LLM 的 ToolMessage（作为 observation 注入 messages）
        out_messages.append(ToolMessage(content=raw_str, tool_call_id=tc["id"]))

        # 给系统的结构化记录
        actions.append({"name": tc["name"], "args": tc["args"], "tool_call_id": tc["id"]})
        observations.append({"tool_call_id": tc["id"], "output": raw_str})

    return {
        "messages": out_messages,
        # "actions": actions,
        # "observations": observations,
        # "steps": state.get("steps", 0) + 1,
    }


def should_continue(state: dict) -> Literal["tool_node", END]:
    """Continue if LLM requested tools, else stop."""
    # 如果上一步已经 fail/success，就停
    # if state.get("status") in ("success", "fail"):
    #     return END
    if not state.get("messages"):
        return END
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        if last.tool_calls:
            return "tool_node"
    return END


# =========================
# 6) Graph
# =========================
builder = StateGraph(ContextState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")

agent = builder.compile()


# =========================
# 7) Run
# =========================
if __name__ == "__main__":
    # 你可以改成你的真实任务
    goal = "Inspect this repo and tell me how to run its tests. If tests fail, fix one failing test or bug with minimal changes."
    constraints = "Do not introduce new dependencies. Keep changes minimal. Only modify files under repo root."
    done_criteria = ["Explain how to run tests", "If you made changes, tests should pass for the changed area"]

    init_state: ContextState = {
        # "goal": goal,
        # "plan": "1) list files 2) find test command 3) run tests 4) fix minimal issue if needed 5) re-run relevant tests",
        # "constraints": constraints,
        # "done_criteria": done_criteria,
        "steps": 0,
        # "status": "running",
        # "last_error": "",
        "messages": [HumanMessage(content=goal)],
        # "actions": [],
        # "observations": [],
        # "focus_files": [],
    }

    out = agent.invoke(init_state)
    # write messages to file for debugging
    with open("messages.json", "w") as f:
        json.dump(out["messages"], f)

    print("\n===================== FINAL MESSAGES =====================")
    for m in out["messages"]:
        m.pretty_print()

    # print("\n===================== ACTIONS (structured) =====================")
    # for a in out.get("actions", []):
    #     print(a)

    # print("\n===================== OBSERVATIONS (structured) =====================")
    # for o in out.get("observations", []):
    #     print({"tool_call_id": o["tool_call_id"], "output_head": o["output"][:200].replace("\n", "\\n")})
