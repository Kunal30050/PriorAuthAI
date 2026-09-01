from mcp.server.mcpserver import MCPServer
from agent.insurance_requirements import InsuranceRequirementsModule
from agent.patient_info import PatientInfoModule, PatientNotFoundError
from agent.document_check import DocumentCheckModule


server = MCPServer(
    name="prior-auth-mcp",
    version="0.1.0",
    description="MCP server for PriorAuthAI patient and authorization data",
)

patient_module = PatientInfoModule()
insurance_module = InsuranceRequirementsModule()
document_module = DocumentCheckModule()


@server.tool(
    name="health_check",
    description="Check whether the PriorAuthAI MCP server is running.",
)
def health_check() -> str:
    return "PriorAuth MCP server is healthy"


@server.tool(
    name="get_patient",
    description="Get basic patient information for a patient ID from the PriorAuthAI patient dataset.",
)
def get_patient(patient_id: str) -> dict:
    try:
        return patient_module.get_patient(patient_id)
    except PatientNotFoundError as exc:
        return {
            "patient_found": False,
            "patient_id": patient_id,
            "error": str(exc),
        }


@server.tool(
    name="get_authorization_history",
    description="Get the patient's previous authorization requests, most recent first.",
)
def get_authorization_history(patient_id: str) -> dict:
    history = patient_module.get_history(patient_id)

    return {
        "patient_id": patient_id,
        "history_count": len(history),
        "history": history,
    }


@server.tool(
    name="get_relevant_authorization_history",
    description="Get previous authorization requests for a patient matching a specific procedure or service code.",
)
def get_relevant_authorization_history(
    patient_id: str,
    procedure_code: str,
) -> dict:
    history = patient_module.get_relevant_history(
        patient_id,
        procedure_code,
    )

    return {
        "patient_id": patient_id,
        "procedure_code": procedure_code,
        "history_count": len(history),
        "history": history,
    }
@server.tool(
    name="get_insurance_requirements",
    description="Check prior authorization requirements for an insurance payer and procedure.",
)
def get_insurance_requirements(
    insurance_id: str,
    procedure_code: str,
) -> dict:
    return insurance_module.check(
        insurance_id,
        procedure_code,
    )

@server.tool(
    name="check_documents",
    description="Check which required documents are submitted for a patient using the existing document records.",
)
def check_documents(
    patient_id: str,
    required_documents: list[str],
) -> dict:
    return document_module.check(
        patient_id,
        required_documents,
    )

@server.tool(
    name="check_document_status",
    description="Check an abstract document checklist and determine which required documents are received or missing.",
)
def check_document_status(
    document_status: dict,
    required_documents: list[str],
) -> dict:
    return document_module.check_from_status(
        document_status,
        required_documents,
    )

if __name__ == "__main__":
    server.run("stdio")