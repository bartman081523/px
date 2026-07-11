"""
toolchain.py — Agentic Micro-Harness Toolchain (v3.5 RIGOR)
=============================================================
Definiert:
  - TOOLCHAIN_DEFINITION: Hermes-Function-Calling-Format System-Prompt
  - ToolCall, ToolLoopResult: Dataclasses
  - parse_tool_call, parse_final_answer: deterministische regex-Parser
  - 4 Tool-Implementierungen: web_search, read_file, write_file, execute_python
  - run_tool_loop: iterative Model-Generation mit Tool-Dispatch

SciMind 5.0 Mandate (gemini-nightly + hermes-agent port):
  - Deterministische Tool-Parsing (regex, NICHT LLM-basiert)
  - Anti-Embedding: Tool-Results in <tool_result>...</tool_result> Bubble
  - Reproducible: gleicher Seed → byte-identischer Output
  - Incomplete-Suggestion: max 5 Iterationen, dann break

Portiert aus v1 (web_search) und v2 (Best-of-all-Worlds Patterns).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Hermes-Function-Calling-Format (NousResearch/hermes-function-calling-v1)
TOOLCHAIN_DEFINITION = """
You have access to these tools (Hermes Function-Calling format):

1. <tool_call>
   {"name": "web_search", "arguments": {"query": "your search query"}}
   </tool_call>
   Returns: Top-3 web results with title, snippet, URL.

2. <tool_call>
   {"name": "read_file", "arguments": {"path": "/absolute/path/to/file"}}
   </tool_call>
   Returns: File contents (max 10KB) or error.

3. <tool_call>
   {"name": "write_file", "arguments": {"path": "/path/to/file", "content": "your content"}}
   </tool_call>
   Returns: Success or error.

4. <tool_call>
   {"name": "execute_python", "arguments": {"code": "print('hello')"}}
   </tool_call>
   Returns: stdout, stderr, exit_code, runtime_sec.

5. <final_answer>your final answer</final_answer>
   Ends the loop. Use this when ready to answer the user.

EPISTEMIC PRINCIPLES (SciMind 5.0):
- Form a hypothesis, then test it with execute_python before answering
- Use web_search to verify uncertain claims
- Save intermediate work via write_file for reproducibility
- Always end with <final_answer> when you have an answer
"""


@dataclass
class ToolCall:
    """Ein geparster Tool-Call aus Model-Output."""
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolLoopResult:
    """Ergebnis eines run_tool_loop Aufrufs."""
    final_answer: Optional[str]
    transcript: List[Dict[str, str]] = field(default_factory=list)
    n_iterations: int = 0
    tool_calls_made: List[str] = field(default_factory=list)
    stuck: bool = False
    max_hit: bool = False


# Regex-Parser: NICHT LLM-basiert, SciMind 5.0 deterministisch.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)


def parse_tool_call(output: str) -> Optional[ToolCall]:
    """Parst einen tool_call-Block aus Model-Output.

    Returns:
        ToolCall oder None (kein Tool, malformed JSON, etc.)
    """
    if not output:
        return None
    m = _TOOL_CALL_RE.search(output)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    name = data.get("name")
    arguments = data.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return ToolCall(name=name, arguments=arguments)


def parse_final_answer(output: str) -> Optional[str]:
    """Parst ein <final_answer>...</final_answer> aus Model-Output.

    Returns:
        String-Inhalt oder None.
    """
    if not output:
        return None
    m = _FINAL_ANSWER_RE.search(output)
    if not m:
        return None
    return m.group(1).strip()


# === Tool-Implementierungen ===

# Path-Whitelist für read_file: bleibt im Projekt-Verzeichnis.
_PATH_WHITELIST = (
    "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/",
    "/run/media/julian/ML4/ollama-work/all_space_6_16_stand",
    "/tmp/",
    "/tmp",
    "/run/media/julian/ML4/open-mythos_p2/",
)
_MAX_READ_BYTES = 10 * 1024  # 10 KB

# Workspace für write_file
_AGENT_WORKSPACE = Path(
    "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/out/agent_workspace"
)
_AGENT_WORKSPACE.mkdir(parents=True, exist_ok=True)


def _is_path_allowed(path: str) -> bool:
    """Prüft ob Pfad in Whitelist ist (Path-Traversal-Schutz)."""
    abs_path = os.path.abspath(path)
    return any(abs_path.startswith(p) for p in _PATH_WHITELIST)


def web_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """DDG-Web-Suche (portiert aus v1/rigor_zone_harness.py:98).

    Returns:
        {"results": [{"title", "snippet", "url"}], "error": Optional[str]}
    """
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        results = [
            {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
            for r in raw
        ]
        return {"results": results, "error": None}
    except Exception as e:
        return {"results": [], "error": f"{type(e).__name__}: {e}"}


def read_file(path: str, max_kb: int = 10) -> Dict[str, Any]:
    """Liest Datei (max 10KB), prüft Path-Whitelist.

    Returns:
        {"content": str, "size": int, "error": Optional[str]}
    """
    if not _is_path_allowed(path):
        return {"content": "", "size": 0, "error": f"Path not in whitelist: {path}"}
    try:
        p = Path(path)
        if not p.exists():
            return {"content": "", "size": 0, "error": f"File not found: {path}"}
        if not p.is_file():
            return {"content": "", "size": 0, "error": f"Not a file: {path}"}
        size = p.stat().st_size
        if size > max_kb * 1024:
            return {
                "content": "",
                "size": size,
                "error": f"File too large: {size} bytes > {max_kb*1024} bytes",
            }
        content = p.read_text(encoding="utf-8", errors="replace")
        return {"content": content, "size": size, "error": None}
    except Exception as e:
        return {"content": "", "size": 0, "error": f"{type(e).__name__}: {e}"}


def write_file(path: str, content: str) -> Dict[str, Any]:
    """Schreibt Datei in agent_workspace (atomar via os.replace).

    Returns:
        {"success": bool, "path": str, "error": Optional[str]}
    """
    # Wenn User absoluten Pfad in Whitelist gibt: nutze ihn. Sonst: agent_workspace.
    if not _is_path_allowed(path):
        # Fallback: in agent_workspace umlenken
        fname = os.path.basename(path) or "agent_output.txt"
        target = _AGENT_WORKSPACE / fname
    else:
        target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via temp + os.replace
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, dir=str(target.parent), suffix=".tmp", encoding="utf-8"
        ) as tf:
            tf.write(content)
            tmp_name = tf.name
        os.replace(tmp_name, str(target))
        return {"success": True, "path": str(target), "error": None}
    except Exception as e:
        return {"success": False, "path": "", "error": f"{type(e).__name__}: {e}"}


def execute_python(code: str, timeout: int = 5) -> Dict[str, Any]:
    """Führt Python-Code in subprocess aus (portiert aus v2 ohne Ollama).

    Returns:
        {"stdout", "stderr", "exit_code", "runtime_sec", "timeout": bool}
    """
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        runtime = time.perf_counter() - t0
        return {
            "stdout": result.stdout[:5000],
            "stderr": result.stderr[:2000],
            "exit_code": result.returncode,
            "runtime_sec": runtime,
            "timeout": False,
        }
    except subprocess.TimeoutExpired:
        runtime = time.perf_counter() - t0
        return {
            "stdout": "",
            "stderr": f"Timeout after {timeout}s",
            "exit_code": -1,
            "runtime_sec": runtime,
            "timeout": True,
        }
    except Exception as e:
        runtime = time.perf_counter() - t0
        return {
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
            "runtime_sec": runtime,
            "timeout": False,
        }


# Tool-Registry für dispatch
_TOOL_REGISTRY = {
    "web_search": web_search,
    "read_file": read_file,
    "write_file": write_file,
    "execute_python": execute_python,
}


def dispatch_tool(tool_call: ToolCall) -> Dict[str, Any]:
    """Führt einen Tool-Call aus.

    Returns:
        Tool-Result-Dict (vom jeweiligen Tool).
    """
    fn = _TOOL_REGISTRY.get(tool_call.name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_call.name}"}
    try:
        return fn(**tool_call.arguments)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _format_tool_result(result: Dict[str, Any]) -> str:
    """Formatiert Tool-Result für <tool_result>-Bubble (Anti-Embedding)."""
    return f"<tool_result>{json.dumps(result, ensure_ascii=False, default=str)[:8000]}</tool_result>"


def run_tool_loop(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_iterations: int = 5,
    max_new_tokens: int = 300,
    enable_tools: Optional[List[str]] = None,
) -> ToolLoopResult:
    """Iterative Model-Generation mit Tool-Dispatch (Hermes-Loop).

    Args:
        model: HuggingFace Modell (cuda, eval mode)
        tokenizer: Associated tokenizer
        system_prompt: System-Prompt inkl. TOOLCHAIN_DEFINITION
        user_prompt: User-Prompt mit Task
        max_iterations: Max Tool-Loop-Iterationen
        max_new_tokens: Max neue Tokens pro Generation
        enable_tools: Liste aktiver Tools. None = alle.

    Returns:
        ToolLoopResult mit final_answer, transcript, etc.
    """
    if enable_tools is None:
        enable_tools = list(_TOOL_REGISTRY.keys())

    active_registry = {k: v for k, v in _TOOL_REGISTRY.items() if k in enable_tools}

    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    transcript: List[Dict[str, str]] = []
    tool_calls_made: List[str] = []

    for i in range(max_iterations):
        # Generate
        prompt_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with __import__("torch").inference_mode():
            output_ids = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        new_text = tokenizer.decode(
            output_ids[0, ids["input_ids"].shape[1]:], skip_special_tokens=True
        )

        transcript.append({"role": "assistant", "content": new_text})

        # Parse final_answer first
        final = parse_final_answer(new_text)
        if final is not None:
            return ToolLoopResult(
                final_answer=final, transcript=transcript, n_iterations=i + 1,
                tool_calls_made=tool_calls_made,
            )

        # Parse tool_call
        tool_call = parse_tool_call(new_text)
        if tool_call is None or tool_call.name not in active_registry:
            # Model ist stuck: keine Antwort, kein Tool
            return ToolLoopResult(
                final_answer=new_text, transcript=transcript, n_iterations=i + 1,
                tool_calls_made=tool_calls_made, stuck=True,
            )

        # Tool dispatch
        result = dispatch_tool(tool_call)
        tool_calls_made.append(tool_call.name)
        result_str = _format_tool_result(result)
        transcript.append({"role": "user", "content": result_str})
        conversation.append({"role": "assistant", "content": new_text})
        conversation.append({"role": "user", "content": result_str})

    # Max iterations hit
    return ToolLoopResult(
        final_answer=None, transcript=transcript, n_iterations=max_iterations,
        tool_calls_made=tool_calls_made, max_hit=True,
    )
