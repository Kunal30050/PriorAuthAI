import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command=r".venv\Scripts\python.exe",
        args=["-m", "agent.mcp_tools.server"],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:

            await session.initialize()

            tools = await session.list_tools()

            print("\n=== Available MCP Tools ===")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            print("\n=== get_patient(P001) ===")

            result = await session.call_tool(
                "get_patient",
                {"patient_id": "P001"},
            )

            print(result)

            print("\n=== get_authorization_history(P001) ===")

            result = await session.call_tool(
                "get_authorization_history",
                {"patient_id": "P001"},
            )

            print(result)

            print("\n=== get_authorization_history(PAT-19873) ===")

            result = await session.call_tool(
                "get_authorization_history",
                {"patient_id": "PAT-19873"},
            )

            print(result)


            print("\n=== get_insurance_requirements(PAY-0445, HCPCS-E0607) ===")

            result = await session.call_tool(
                "get_insurance_requirements",
                {
                    "insurance_id": "PAY-0445",
                    "procedure_code": "HCPCS-E0607",
                },
            )
            print(result)

            print("\n=== check_documents() ===")

            result = await session.call_tool(
                "check_documents",
                {
                    "patient_id": "PAT-19873",
                    "required_documents": [
                        "clinical_note",
                        "diagnosis_evidence",
                        ],
                },
            )
            print(result)

            print("\n=== check_document_status() ===")
            result = await session.call_tool(
                "check_document_status",
                {
                    "document_status": {
                        "clinical_note": True,
                        "diagnosis_evidence": True,
                        "lab_report": True,
                        "imaging_report": False,
                        "medication_history": True,
                        "physician_order": False,
                    },
                    "required_documents": [
                        "clinical_note",
                        "diagnosis_evidence",
                        "lab_report",
                        "imaging_report",
                        "medication_history",
                        "physician_order",
                    ],
                },
            )
            print(result)


if __name__ == "__main__":
    asyncio.run(main())