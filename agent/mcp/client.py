import asyncio
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from tracing import log_event
except ImportError:
    from agent.tracing import log_event


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

SERVER_PARAMS = StdioServerParameters(
    command=str(VENV_PYTHON),
    args=["-m", "agent.mcp_tools.server"],
)


async def call_mcp_tool(tool_name: str, arguments: dict):
    async with stdio_client(SERVER_PARAMS) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool(
                tool_name,
                arguments,
            )

            return result


def call_mcp_tool_sync(
    name,
    arguments,
    conversation_id=None,
    workflow_id=None,
    expected_tool=None,
):
    start = time.perf_counter()

    try:
        result = asyncio.run(
            call_mcp_tool(
                name,
                arguments,
            )
        )

        parsed = _result_dict(result)

        is_error = getattr(result, "is_error", False)
        valid_payload = parsed is not None
        success = not is_error and valid_payload

        elapsed_ms = (time.perf_counter() - start) * 1000

        log_event(
            "mcp_tool_use",
            conversation_id=conversation_id,
            workflow_id=workflow_id,
            payload={
                "tool_name": name,
                "expected_tool": expected_tool,
                "success": success,
                "mcp_error": is_error,
                "valid_payload": valid_payload,
                "latency_ms": elapsed_ms,
            },
        )

        return result

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000

        log_event(
            "mcp_tool_use",
            conversation_id=conversation_id,
            workflow_id=workflow_id,
            payload={
                "tool_name": name,
                "expected_tool": expected_tool,
                "success": False,
                "mcp_error": True,
                "valid_payload": False,
                "latency_ms": elapsed_ms,
                "error": str(exc),
            },
        )

        raise