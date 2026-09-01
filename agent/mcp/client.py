import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PARAMS = StdioServerParameters(
    command=r".venv\Scripts\python.exe",
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


def call_mcp_tool_sync(tool_name: str, arguments: dict):
    return asyncio.run(
        call_mcp_tool(tool_name, arguments)
    )