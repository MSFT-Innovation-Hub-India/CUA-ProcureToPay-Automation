from openai import AzureOpenAI
import base64
import json
import os
import asyncio
import argparse
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from call_computer_use import post_purchase_invoice_header, retrieve_contract

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process purchase invoice image.')
parser.add_argument('--image', type=str, default="data_files/Invoice-002.png", 
                    help='Path to the purchase invoice image file')
args = parser.parse_args()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL = os.getenv("MODEL_NAME2")
API_VERSION = os.getenv("AZURE_API_VERSION")
vector_store_id_to_use = os.getenv("vector_store_id")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    azure_ad_token_provider=token_provider,
    api_version=API_VERSION,
)

available_functions = {
    "retrieve_contract": retrieve_contract,
    "post_purchase_invoice_header": post_purchase_invoice_header,
}

# Use the image path from command line argument
image_path = args.image
print(f"Processing invoice image: {image_path}")

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Encode single image
base64_image = encode_image_to_base64(image_path)

# These are the tools that will be used by the Responses API.
tools_list =  [
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
                        "description": "The instructions to retrieve the contract details from the web page",
                    },
                },
                "required": ["contractid","instructions"],
            },
        },
    ]

instructions = """
This is a Procure to Pay process. You will be provided with the Purchase Invoice image as input.
Note that Step 3 can be performed only after Step 1 and Step 2 are completed.
Step 1: As a first step, you will use visual reasoning to read the Contract ID from the Purchase Invoice image along with the line items from the Invoice in the form of a table.
Step 2: Next, use the function tool by passing the Contract ID above to retrieve the contract details.
Step 3: Next, use the file search tool to retrieve the business rules applicable to detection of anomalies in the Procure to Pay process.
Step 4: Next, apply the retrieved business rules to match the invoice line items with the contract details fetched from in step 2, and detect anomalies if any.
    - Perform validation of the Invoice against the Contract and determine if there are any anomalies detected.
    - **When giving the verdict, you must call out each Invoice and Invoice line detail where the discrepancy was. Use your knowledge of the domain to interpret the information right and give a response that the user can store as evidence**
    - Note that it is ok for the quantities in the invoice to be lesser than the quantities in the contract, but not the other way around.
    - When providing the verdict, depict the results in the form of a Markdown table, matching details from the Invoice and Contract side-by-side. Verification of Invoice Header against Contract Header should be in a separate .md table format. That for the Invoice Lines verified against the Contract lines in a separate .md table format.
    - If the Contract Data is not provided as an input when evaluating the Business rules, then desist from providing the verdict. State in the response that you could not provide the verdict since the Contract Data was not provided as an input. **DO NOT MAKE STUFF UP**.
    **Use chain of thought when processing the user requests**
Step 5: Finally, you will use the function tool to post the purchase invoice into the system by passing the Invoice details.
    - use the content from step 4 above, under ### Final Verdict, for the value of the $remarks field, after replacing the new line characters with a space.
    - The instructions you must pass are: Fill the form with purchase_invoice_no '$PurchaseInvoiceNumber', contract_reference '$contract_reference', supplier_id '$supplierid', total_invoice_value $total_invoice_value (in 2335.00 format), invoice_date '$invoice_data' (string in mm/dd/yyyy format), status '$status', remarks '$remarks'. Save this information by clicking on the 'save' button. If the response message shows a dialog box or a message box, acknowledge it. \n An example of the user_input format you must send is -- 'Fill the form with purchase_invoice_no 'PInv_001', contract_reference 'contract997801', supplier_id 'supplier99010', total_invoice_value 23100.00, invoice_date '12/12/2024', status 'approved', remarks 'invoice is valid and approved'. Save this information by clicking on the 'save' button. If the response message shows a dialog box or a message box, acknowledge it'
"""

user_prompt = """
here are the Purchase Invoice image(s) as input. Detect anomalies in the procure to pay process and use that to post the invoice header and line items data to the system.
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
            }
        ],
    }
]


async def main():
    # Create a copy of the input messages to maintain state across API calls
    current_input_messages = input_messages.copy()
    completed = False
    step_tracker = 1  # Track which step of the process we're on
    last_response_text = ""
    
    try:
        while not completed:
            # Log the current step being processed
            print(f"Processing step {step_tracker} of the Procure to Pay workflow...")
            
            # Call the Responses API with the current state
            response = client.responses.create(
                model=os.getenv("MODEL_NAME2"),
                instructions=instructions,
                input=current_input_messages,
                tools=tools_list,
                parallel_tool_calls=False,
            )
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
                    print(f"Executing function: {function_name}")
                    
                    # Add function call to messages
                    current_input_messages.append(tool_call)
                    
                    # Get the appropriate function to call
                    function_to_call = available_functions.get(function_name)
                    if not function_to_call:
                        print(f"Error: Function {function_name} not found in available functions")
                        raise ValueError(f"Function {function_name} not found")
                    
                    try:
                        # Parse arguments and call the function
                        function_args = json.loads(tool_call.arguments)
                        
                        # Execute the function (async or sync)
                        if asyncio.iscoroutinefunction(function_to_call):
                            function_response = await function_to_call(**function_args)
                        else:
                            function_response = function_to_call(**function_args)
                        
                        # Add function result to messages
                        current_input_messages.append({
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": str(function_response)
                        })
                        
                        print(f"Function {function_name} executed successfully")
                        
                        # Update step tracker based on the function called
                        if function_name == "retrieve_contract":
                            step_tracker = 3  # Moving to step 3 after contract retrieval
                        elif function_name == "post_purchase_invoice_header":
                            step_tracker = 5  # We're at the final step
                            completed = True  # Mark as completed when post_invoice is called
                    
                    except Exception as func_error:
                        error_message = f"Error executing function {function_name}: {str(func_error)}"
                        print(error_message)
                        
                        # Add error message to the conversation
                        current_input_messages.append({
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": f"ERROR: {error_message}"
                        })
                
                elif tool_call.type == "file_search_call":
                    # Handle file search tool calls
                    print("File search tool was called to retrieve business rules")
                    current_input_messages.append(tool_call)
                    step_tracker = 4  # Move to step 4 after retrieving business rules
            
            # After each batch of tool calls, check if we've reached the final step
            if step_tracker >= 5:
                completed = True
    
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        
    # Print the final output or the last meaningful response
    if completed and last_response_text:
        print("\nFinal Output:")
        print(last_response_text)
    elif response and hasattr(response, 'output_text'):
        print("\nLast Response:")
        print(response.output_text)
    else:
        print("\nProcess did not complete successfully.")


# Run the async main function
# To run the script, use the command: python app.py --image data_files/Invoice-002.png
if __name__ == "__main__":
    asyncio.run(main())
