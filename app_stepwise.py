# NOTE:
# This file implements the stepwise variant of the procure-to-pay workflow.
# Instead of passing all 5 steps as dense instructions to the LLM in a single Responses API call (see app.py),
# each step is performed sequentially via multiple Responses API calls.
# This approach avoids the 500 Internal Server Error sometimes encountered with dense instructions
# in the preview version of the Azure OpenAI Responses API.

import os
import json
import base64
import asyncio
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from call_computer_use import post_purchase_invoice_header, retrieve_contract

# Load environment variables
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL = os.getenv("MODEL_NAME2")
API_VERSION = os.getenv("AZURE_API_VERSION")
VECTOR_STORE_ID = os.getenv("vector_store_id")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    azure_ad_token_provider=token_provider,
    api_version=API_VERSION,
)


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# You are given a purchase invoice image. Extract the Contract ID and all invoice line items as a table. Return the result as JSON with keys 'contractId', 'purchase_invoice_number', 'supplier_id', 'total_invoice_value' and 'invoiceLines'.


async def extract_invoice_data(image_path):
    base64_image = encode_image_to_base64(image_path)
    user_prompt = """
You are given a purchase invoice image. Extract the following fields and return a JSON object with these exact key names:
{
  "contractId": <contract id>,
  "invoiceNumber": <invoice number>,
  "supplierId": <supplier id>,
  "totalInvoiceValue": <total invoice value>,
  "invoiceDate": <invoice date>,
  "invoiceLines": [
    {"itemId": <item id>, "quantity": <quantity>, "unitPrice": <unit price>, "totalPrice": <total price>, "description": <description>},
    ...
  ]
}
If any field is missing, set its value to null. Do not use any other key names. Only output the JSON object, nothing else.
"""
    input_messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high",
                },
            ],
        }
    ]
    response = client.responses.create(
        model=MODEL,
        input=input_messages,
        tools=[],
        parallel_tool_calls=False,
    )
    # Try to extract JSON from the response
    for output in response.output:
        if hasattr(output, "content") and output.content:
            text = output.content[0].text
            try:
                json_start = text.find("{")
                json_end = text.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    return json.loads(text[json_start : json_end + 1])
            except Exception:
                continue
    raise ValueError("Could not extract invoice data from model response.")


async def get_contract_details(contractid):
    # Use the retrieve_contract tool
    instructions = "Extract all contract header and contract line items as JSON."
    result = await retrieve_contract(contractid=contractid, instructions=instructions)
    try:
        return json.loads(result)
    except Exception:
        return result


async def get_business_rules():
    # Use file_search tool to retrieve business rules
    input_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Retrieve business rules for procure to pay anomaly detection as text.",
                }
            ],
        }
    ]
    tools_list = [
        {
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID],
            "max_num_results": 5,
        }
    ]
    response = client.responses.create(
        model=MODEL,
        input=input_messages,
        tools=tools_list,
        parallel_tool_calls=False,
    )
    # Return the first file content found
    for output in response.output:
        if hasattr(output, "content") and output.content:
            return output.content[0].text
    return None


async def detect_anomalies(invoice_data, contract_data, business_rules):
    user_prompt = f"""
Given the following:
- Invoice Data: {json.dumps(invoice_data)}
- Contract Data: {json.dumps(contract_data)}
- Business Rules: {business_rules}

Your task:
- Determine the status as either 'approved' or 'rejected'.
- Provide detailed_verdict: Output a Markdown table with details to justify the status above (compare invoice and contract, highlight discrepancies, etc).
- Provide summary_verdict: A summary of the justification for the verdict, including key aspects of the findings.

Return your response as a JSON object with the following keys:
{{
  "status": <'approved' or 'rejected'>,
  "detailed_verdict": <markdown table>,
  "summary_verdict": <summary justification>
}}
If contract data is missing, set status to 'rejected' and explain in summary_verdict.
"""
    input_messages = [
        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}
    ]
    response = client.responses.create(
        model=MODEL,
        input=input_messages,
        tools=[],
        parallel_tool_calls=False,
    )
    for output in response.output:
        if hasattr(output, "content") and output.content:
            text = output.content[0].text
            try:
                json_start = text.find("{")
                json_end = text.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    return json.loads(text[json_start : json_end + 1])
            except Exception:
                return text
    return None


async def post_invoice(invoice_data, verdict):
    """
    Compose instructions for posting invoice, including all required fields and the summary_verdict from the anomaly detection step.
    """
    purchase_invoice_no = invoice_data.get("invoiceNumber", "UNKNOWN")
    contract_reference = invoice_data.get("contractId", "UNKNOWN")
    supplier_id = invoice_data.get("supplierId", "UNKNOWN")
    total_invoice_value = invoice_data.get("totalInvoiceValue", "0.00")
    invoice_date = invoice_data.get("invoiceDate", "UNKNOWN")
    # verdict is expected to be a dict with keys: status, detailed_verdict, summary_verdict
    status = verdict.get("status", "rejected")
    remarks = verdict.get("summary_verdict", "No summary provided").replace("\n", " ")

    instructions = (
        f"Fill the form with purchase_invoice_no '{purchase_invoice_no}', "
        f"contract_reference '{contract_reference}', "
        f"supplier_id '{supplier_id}', "
        f"total_invoice_value - enter only the numeric part in this data{total_invoice_value}, "
        f"invoice_date '{invoice_date}', "
        f"status '{status}', remarks - paste a summary along with key information '{remarks}'. "
        f"Save this information by clicking on the 'save' button. "
        f"If the response message shows a dialog box or a message box, acknowledge it."
    )
    result = await post_purchase_invoice_header(instructions=instructions)
    return result


async def main(image_path=None, streamlit_mode=False):
    """
    Stepwise workflow for procure-to-pay automation.
    Args:
        image_path (str): Path to the invoice image file.
        streamlit_mode (bool): If True, returns a dict of all step results for Streamlit UI.
    Returns:
        If streamlit_mode: dict with keys 'invoice_data', 'contract_data', 'business_rules', 'verdict', 'post_result'.
        Else: None (prints to stdout).
    """
    if image_path is None:
        image_path = "data_files/Invoice-001.png"
    results = {}
    # Step 1: Extract invoice data
    try:
        invoice_data = await extract_invoice_data(image_path)
        print("Extracted invoice data:\n", json.dumps(invoice_data, indent=2))
    except Exception as e:
        invoice_data = {"error": f"Invoice extraction failed: {str(e)}"}
    results['invoice_data'] = invoice_data

    # Step 2: Retrieve contract details
    contractid = invoice_data.get("contractId") if isinstance(invoice_data, dict) else None
    try:
        contract_data = await get_contract_details(contractid) if contractid else {"error": "No contractId found in invoice data."}
    except Exception as e:
        contract_data = {"error": f"Contract retrieval failed: {str(e)}"}
    results['contract_data'] = contract_data

    # Step 3: Retrieve business rules
    try:
        business_rules = await get_business_rules()
        print("Business rules retrieved:\n", business_rules)
    except Exception as e:
        business_rules = f"Business rules retrieval failed: {str(e)}"
    results['business_rules'] = business_rules

    # Step 4: Detect anomalies
    try:
        verdict = await detect_anomalies(invoice_data, contract_data, business_rules)
    except Exception as e:
        verdict = {"error": f"Anomaly detection failed: {str(e)}"}
    results['verdict'] = verdict

    # Step 5: Post invoice
    try:
        post_result = await post_invoice(invoice_data, verdict)
    except Exception as e:
        post_result = f"Post invoice failed: {str(e)}"
    results['post_result'] = post_result

    if streamlit_mode:
        return results
    else:
        print("Step 1: Extracted invoice data:\n", invoice_data)
        print("Step 2: Contract details:\n", contract_data)
        print("Step 3: Business rules:\n", business_rules)
        print("Step 4: Verdict:\n", verdict)
        print("Step 5: Post result:\n", post_result)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process purchase invoice image (stepwise).')
    parser.add_argument('--image', type=str, default="data_files/Invoice-001.png", help='Path to the purchase invoice image file')
    parser.add_argument('--streamlit_mode', action='store_true', help='Return results as dict for Streamlit UI')
    args = parser.parse_args()
    asyncio.run(main(image_path=args.image, streamlit_mode=args.streamlit_mode))
