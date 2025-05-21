"""
Note: The Streamlit code is not fully functional yet. To run the application use app.py directly instead
Streamlit app for Purchase Invoice Anomaly Detection using Azure OpenAI and Playwright.

# NOTE:
# This Streamlit app now uses the stepwise workflow implemented in app_stepwise.py for robust, reliable processing.
# The stepwise approach avoids 500 Internal Server Errors from dense LLM instructions in the preview Azure OpenAI Responses API.

"""

import sys
import os

if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import base64
import json
import asyncio
from tempfile import NamedTemporaryFile
from PIL import Image
import io
import nest_asyncio

# Apply nest_asyncio to allow nested event loops (needed for asyncio in Streamlit)
nest_asyncio.apply()

# Import from app.py
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from call_computer_use import post_purchase_invoice_header, retrieve_contract
import app_stepwise

# Set page configuration
st.set_page_config(
    page_title="Purchase Invoice Anomaly Detection",
    page_icon="📝",
    layout="wide"
)

# Azure OpenAI configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL = os.getenv("MODEL_NAME2")
API_VERSION = os.getenv("AZURE_API_VERSION")
vector_store_id_to_use = os.getenv("vector_store_id")

# Set up Azure OpenAI client with managed identity authentication
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    azure_ad_token_provider=token_provider,
    api_version=API_VERSION
)

# Tool list for the Responses API
tools_list = [
    {
        "type": "file_search",
        "vector_store_ids": [vector_store_id_to_use],
        "max_num_results": 20,
    },
    {
        "type": "function",
        "name": "post_purchase_invoice_header",
        "description": "post the purchase invoice header data to the system",
        "parameters": {
            "type": "object",
            "properties": {
                "instructions": {
                    "type": "string",
                    "description": "The instructions to populate and post form data in the purchase invoice header form in the web page",
                },
            },
            "required": ["instructions"],
        },
    },
    {
        "type": "function",
        "name": "retrieve_contract",
        "description": "fetch contract details for the given contractid",
        "parameters": {
            "type": "object",
            "properties": {
                "contractid": {
                    "type": "string",
                    "description": "The contract id registered for the Supplier in the System",
                },
                "instructions": {
                    "type": "string",
                    "description": "The instructions to populate and post form data in the purchase invoice header form in the web page",
                },
            },
            "required": ["contractid","instructions"],
        },
    },
]

# Instructions for the model
model_instructions = """
This is a Procure to Pay process. You will be provided with the Purchase Invoice image as input.
Note that Step 3 can be performed only after Step 1 and Step 2 are completed.
Step 1: As a first step, you will extract the Contract ID from the Invoice and also all the line items from the Invoice in the form of a table.
Step 2: You will then use the function tool to call the computer using agent with the Contract ID to get the contract details.
Step 3: You will then use the file search tool to retrieve the business rules applicable to detection of anomalies in the Procure to Pay process.
Step 4: Then, apply the retrieved business rules to match the invoice line items with the contract details fetched from in step 2, and detect anomalies if any.
    - Perform validation of the Invoice against the Contract and determine if there are any anomalies detected.
    - **When giving the verdict, you must call out each Invoice and Invoice line detail where the discrepancy was. Use your knowledge of the domain to interpret the information right and give a response that the user can store as evidence**
    - Note that it is ok for the quantities in the invoice to be lesser than the quantities in the contract, but not the other way around.
    - When providing the verdict, depict the results in the form of a Markdown table, matching details from the Invoice and Contract side-by-side. Verification of Invoice Header against Contract Header should be in a separate .md table format. That for the Invoice Lines verified against the Contract lines in a separate .md table format.
    - If the Contract Data is not provided as an input when evaluating the Business rules, then desist from providing the verdict. State in the response that you could not provide the verdict since the Contract Data was not provided as an input. **DO NOT MAKE STUFF UP**.
    **Use chain of thought when processing the user requests**
Step 5: Finally, you will use the function tool to call the computer using agent with the Invoice details to post the invoice header data to the system.
    - use the content from step 4 above, under ### Final Verdict, for the value of the $remarks field, after replacing the new line characters with a space.
    - The instructions you must pass are: Fill the form with purchase_invoice_no '$PurchaseInvoiceNumber', contract_reference '$contract_reference', supplier_id '$supplierid', total_invoice_value $total_invoice_value (in 2335.00 format), invoice_date '$invoice_data' (string in mm/dd/yyyy format), status '$status', remarks '$remarks'. Save this information by clicking on the 'save' button. If the response message shows a dialog box or a message box, acknowledge it. \n An example of the user_input format you must send is -- 'Fill the form with purchase_invoice_no 'PInv_001', contract_reference 'contract997801', supplier_id 'supplier99010', total_invoice_value 23100.00, invoice_date '12/12/2024', status 'approved', remarks 'invoice is valid and approved'. Save this information by clicking on the 'save' button. If the response message shows a dialog box or a message box, acknowledge it'
"""

def encode_image_to_base64(image_bytes):
    """Encode image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode("utf-8")

def convert_to_jpeg(image_bytes):
    """Convert any image format to JPEG format"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if needed (e.g., for PNG with transparency)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Save as JPEG to bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"Error converting image: {str(e)}")
        return None

def run_async(func):
    """Helper function to run async functions in Streamlit"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(func)
    finally:
        loop.close()

async def process_invoice(user_prompt, image_bytes, status_area, response_area):
    """Process the invoice using the Azure OpenAI client"""
    # Convert image to JPEG if it's not already
    jpeg_image_bytes = convert_to_jpeg(image_bytes)
    if not jpeg_image_bytes:
        status_area.error("Failed to process the image. Please try again with a different image.")
        return "Failed to process the image."
    
    # Encode the image to base64
    base64_image = encode_image_to_base64(jpeg_image_bytes)
    
    # Prepare input messages with user prompt and image
    input_messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high",
                }
            ],
        }
    ]
    
    # Create a copy of the input messages to maintain state across API calls
    current_input_messages = input_messages.copy()
    completed = False
    step_tracker = 1
    last_response_text = ""
    all_responses = []
    
    try:
        while not completed:
            # Display progress
            status_area.info(f"Processing step {step_tracker} of the Procure to Pay workflow...")
            
            # Call the Responses API with the current state
            response = client.responses.create(
                model=os.getenv("MODEL_NAME2") or "gpt-4o",
                instructions=model_instructions,  # Use model_instructions instead of instructions
                input=current_input_messages,
                tools=tools_list,
                parallel_tool_calls=False,
            )
            
            # If there's a text response, display it
            if hasattr(response, 'output_text') and response.output_text:
                all_responses.append(response.output_text)
                response_area.markdown(response.output_text)
            
            # If there's no tool call or all steps are completed, we're done
            if not response.output or not any(output.type == "function_call" for output in response.output):
                print("No tool calls in response. Workflow may be complete.")
                last_response_text = response.output_text
                completed = True
                break
                
            # Process all tool calls in the response
            for tool_call in response.output:
                if tool_call.type == "function_call":
                    function_name = tool_call.name
                    status_area.info(f"Executing function: {function_name}")
                    
                    # Add function call to messages
                    current_input_messages.append(tool_call)
                    
                    # Execute the appropriate function based on name
                    try:
                        # Parse arguments
                        function_args = json.loads(tool_call.arguments)
                        
                        # Execute the function based on name
                        function_response = None
                        if function_name == "retrieve_contract":
                            contractid = function_args.get("contractid", "")
                            fn_instructions = function_args.get("instructions", "")  # Renamed to fn_instructions
                            function_response = await retrieve_contract(contractid, fn_instructions)
                        elif function_name == "post_purchase_invoice_header":
                            fn_instructions = function_args.get("instructions", "")  # Renamed to fn_instructions
                            function_response = await post_purchase_invoice_header(fn_instructions)
                        else:
                            raise ValueError(f"Unknown function: {function_name}")
                        
                        # Add function result to messages
                        current_input_messages.append({
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": str(function_response)
                        })
                        
                        status_area.info(f"Function {function_name} executed successfully")
                        
                        # Update step tracker based on the function called
                        if function_name == "retrieve_contract":
                            step_tracker = 3  # Moving to step 3 after contract retrieval
                        elif function_name == "post_purchase_invoice_header":
                            step_tracker = 5  # We're at the final step
                            completed = True  # Mark as completed when post_invoice is called
                    
                    except Exception as func_error:
                        error_message = f"Error executing function {function_name}: {str(func_error)}"
                        status_area.error(error_message)
                        
                        # Add error message to the conversation
                        current_input_messages.append({
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": f"ERROR: {error_message}"
                        })
                
                elif tool_call.type == "file_search_call":
                    # Handle file search tool calls
                    status_area.info("File search tool was called to retrieve business rules")
                    current_input_messages.append(tool_call)
                    step_tracker = 4  # Move to step 4 after retrieving business rules
            
            # After each batch of tool calls, check if we've reached the final step
            if step_tracker >= 5:
                completed = True
    
    except Exception as e:
        status_area.error(f"Error during processing: {str(e)}")
        
    # Return the final result
    if completed and last_response_text:
        status_area.success("Processing completed!")
        return "\n\n".join(all_responses)
    else:
        return "Process did not complete successfully."

# Streamlit UI
st.title("Purchase Invoice Anomaly Detection (Stepwise)")

uploaded_file = st.file_uploader("Upload a Purchase Invoice Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with open("data_files/streamlit_uploaded_invoice.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image("data_files/streamlit_uploaded_invoice.png", caption="Uploaded Invoice", use_column_width=True)

    if st.button("Process Invoice (Stepwise)"):
        with st.spinner("Processing invoice..."):
            # Run the stepwise workflow using the uploaded image
            result = asyncio.run(app_stepwise.main(image_path="data_files/streamlit_uploaded_invoice.png", streamlit_mode=True))
            st.success("Processing complete!")
            # Display each step's output in a user-friendly way
            st.subheader("Step 1: Invoice Extraction")
            st.json(result.get('invoice_data'))
            st.subheader("Step 2: Contract Details")
            st.json(result.get('contract_data'))
            st.subheader("Step 3: Business Rules")
            st.code(result.get('business_rules'), language='markdown')
            st.subheader("Step 4: Anomaly Detection Verdict")
            verdict = result.get('verdict')
            if isinstance(verdict, dict):
                st.write(f"**Status:** {verdict.get('status')}")
                st.markdown(verdict.get('detailed_verdict', ''))
                st.write(f"**Summary:** {verdict.get('summary_verdict', '')}")
            else:
                st.json(verdict)
            st.subheader("Step 5: Post Invoice Result")
            st.write(result.get('post_result'))
else:
    st.info("Please upload a purchase invoice image to begin.")