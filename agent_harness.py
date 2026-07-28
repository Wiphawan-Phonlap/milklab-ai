"""MilkLab Agent Harness (S2).

Usage:
    python agent_harness.py --cmd "บันทึกขายนมหมี 2 ขวด ขวดละ 65"

รับคำสั่งภาษาไทย ส่งให้ Gemini พร้อม tool schema parse response เป็น tool call
เรียก tool จริง print trace log

นักศึกษาต้องเติม TODO ใน 3 จุด ใน Session 2 Lab 2.3
"""

import argparse
import os
import sys
from collections.abc import Mapping

from dotenv import load_dotenv
from google import genai
from google.genai import types


TOOL_SCHEMA = [
    {
        "name": "log_sale",
        "description": "บันทึกการขายลง Google Sheets และส่ง notification",
        "parameters": {
            "type": "object",
            "properties": {
                "menu": {"type": "string", "description": "ชื่อเมนู"},
                "qty": {"type": "integer", "description": "จำนวนที่ขาย"},
                "price": {"type": "number", "description": "ราคาต่อหน่วย"},
            },
            "required": ["menu", "qty", "price"],
        },
    },
    {
        "name": "query_sales",
        "description": "ดูยอดขายของวันที่ระบุ",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "วันที่ format YYYY-MM-DD"},
            },
            "required": ["date"],
        },
    },
    {
        "name": "send_alert",
        "description": "ส่ง message แจ้งเตือนผ่าน Bot",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
    },
]


def log_sale(menu: str, qty: int, price: float) -> str:
    """Return a fixed success message for sale logging."""

    total = qty * price
    return f"OK: บันทึก {total:g} บาท notif via telegram"


def query_sales(date: str) -> str:
    """Return a fixed success message for daily sales lookup."""

    return f"OK: ยอดขายประจำวันที่ {date} คือ 1,250 บาท"


def send_alert(message: str) -> str:
    """Return a fixed success message for alert delivery."""

    return f"OK: ส่งการแจ้งเตือน '{message}' เรียบร้อยแล้ว"


def _build_tool_declarations() -> list[types.Tool]:
    return [
        types.Tool(
            functionDeclarations=[
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parametersJsonSchema=tool["parameters"],
                )
                for tool in TOOL_SCHEMA
            ]
        )
    ]


def _format_tool_call(name: str, args: Mapping[str, object] | None) -> str:
    if not args:
        return f"{name}()"

    rendered_args: list[str] = []
    for key, value in args.items():
        if isinstance(value, str):
            rendered_args.append(f"{key}={value!r}")
        else:
            rendered_args.append(f"{key}={value}")
    return f"{name}({', '.join(rendered_args)})"


def _extract_function_call(response: types.GenerateContentResponse) -> types.FunctionCall:
    function_calls = getattr(response, "function_calls", None) or []
    if function_calls:
        first_call = function_calls[0]
        if first_call:
            return first_call

    candidates = response.candidates or []
    for candidate in candidates:
        content = candidate.content
        if not content or not content.parts:
            continue
        for part in content.parts:
            function_call = getattr(part, "function_call", None)
            if function_call:
                return function_call

            legacy_function_call = getattr(part, "functionCall", None)
            if legacy_function_call:
                return legacy_function_call
    raise RuntimeError("LLM did not return a tool call")


def parse_command(cmd: str, api_key: str | None = None) -> dict:
    """Send cmd to Gemini and extract the first tool call."""

    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=cmd,
        config=types.GenerateContentConfig(
            temperature=0,
            tools=_build_tool_declarations(),
            toolConfig=types.ToolConfig(
                functionCallingConfig=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowedFunctionNames=[tool["name"]
                                          for tool in TOOL_SCHEMA],
                )
            ),
            systemInstruction=(
                "You are a command router for Thai CLI input. "
                "Call exactly one available function and do not return plain text."
            ),
        ),
    )

    function_call = _extract_function_call(response)
    if not function_call.name:
        raise RuntimeError("LLM returned a tool call without a function name")

    return {"tool": function_call.name, "args": function_call.args or {}}


def dispatch_tool(tool_call: dict) -> str:
    """Dispatch the resolved tool call to the local function implementation."""

    tool_name = tool_call.get("tool")
    args = tool_call.get("args") or {}

    if tool_name == "log_sale":
        return log_sale(**args)
    if tool_name == "query_sales":
        return query_sales(**args)
    if tool_name == "send_alert":
        return send_alert(**args)

    raise RuntimeError(f"Unknown tool: {tool_name}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True, help="คำสั่งภาษาไทย")
    args = parser.parse_args()

    try:
        print(f"[user] {args.cmd}")

        tool_call = parse_command(args.cmd)
        print(
            f"[llm]  tool_call: {_format_tool_call(tool_call['tool'], tool_call['args'])}")

        result = dispatch_tool(tool_call)
        print(f"[tool] return: {result}")
        print(f"[assistant] {result}")
        return 0
    except Exception as exc:
        print(f"[tool] error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
