import config

from openai import AzureOpenAI
from contract_tools import retrieve_contract
import base64
import json
import os
import asyncio
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL = os.getenv("MODEL_NAME2")
API_VERSION = "2025-03-01-preview"
vector_store_id_to_use=config.az_vector_store_id

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)


# client = OpenAI(api_key=config.api_key)
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    azure_ad_token_provider=token_provider,
    api_version=API_VERSION
)

available_functions = {
    "retrieve_contract": retrieve_contract,
}

# read the Purchase Invoice image(s) to be sent as input to the model
image_paths = ["data_files/Invoice-002.png"]

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Encode images
base64_images = [encode_image_to_base64(image_path) for image_path in image_paths]

# These are the tools that will be used by the Responses API.
tools_list =  [
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_id_to_use],
            "max_num_results": 20,
        },
        {
            "type": "function",
            "name": "retrieve_contract",
            "description": "fetch contract details for the given contract_id",
            "parameters": {
                "type": "object",
                "properties": {
                    "contract_id": {
                        "type": "string",
                        "description": "The contract id registered for the Supplier in the System",
                    }
                },
                "required": ["contract_id"],
            },
        },
    ]

instructions="""
This is a Procure to Pay process. You will be provided with the Purchase Invoice image as input.
Note that Step 3 can be performed only after Step 1 and Step 2 are completed.
Step 1: As a first step, you will extract the Contract ID from the Invoice and also all the line items from the Invoice in the form of a table.
Step 2: You will then use the function tool to call the computer using agent with the Contract ID to get the contract details.
Step 3: You will then use the file search tool to retrieve the business rules applicable to detection of anomalies in the Procure to Pay process.
Step 4: Then, apply the retrieved business rules to match the invoice line items with the contract details fetched from in step 2, and detect anomalies if any.
    - Perform validation of the Invoice against the Contract and determine if there are any anomalies detected.
    - **When giving the verdict, you must call out each Invoice and Invoice line detail where the discrepancy was. Use your knowledge of the domain to interpret the information right and give a response that the user can store as evidence**
    - When providing the verdict, depict the results in the form of a Markdown table, matching details from the Invoice and Contract side-by-side. Verification of Invoice Header against Contract Header should be in a separate .md table format. That for the Invoice Lines verified against the Contract lines in a separate .md table format.
    - If the Contract Data is not provided as an input when evaluating the Business rules, then desist from providing the verdict. State in the response that you could not provide the verdict since the Contract Data was not provided as an input. **DO NOT MAKE STUFF UP**.
    **Use chain of thought when processing the user requests**
"""

user_prompt = """
here are the Purchase Invoice image(s) as input. Detect anomalies in the procure to pay process and give me a detailed report
"""

input_messages = [
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": user_prompt},
            *[
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high",
                }
                for base64_image in base64_images
            ],
        ],
    }
]

# Ensure `config.model` is set to a valid model name
if not config.model:
    config.model = os.getenv("MODEL_NAME2") or "gpt-4o"

async def main():
    # The following code is to call the Responses API with the input messages and tools
    response = client.responses.create(
        model=os.getenv("MODEL_NAME2"),
        instructions=instructions,
        input=input_messages,
        tools=tools_list,
        #tool_choice="auto",
        parallel_tool_calls=False,
    )
    tool_call = response.output[0]
    print(f"tool call: {response.output[0]}")

    # We know this needs a function call, that needs to be executed from here in the application code.
    # Lets get hold of the function name and arguments from the Responses API response.
    function_response = None
    function_to_call = None
    function_name = None
    
    
    if response.output[0].type == "function_call":
        function_name = response.output[0].name
        function_to_call = available_functions[function_name]
        function_args = json.loads(response.output[0].arguments)
        # Lets call the Logic app with the function arguments to get the contract details.
        if asyncio.iscoroutinefunction(function_to_call):
            function_response = await function_to_call(**function_args)
        else:
            function_response = function_to_call(**function_args)
            
    input_messages.append(tool_call)  # append model's function call message
    input_messages.append({           # append result message
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": str(function_response)
    })

    # Check if there's a function call in the response
    # function_calls = []
    # for output in response.output:
    #     if hasattr(output, 'type') and output.type == 'function_call':
    #         function_calls.append(output)
    
    # # Process function calls if any
    # if function_calls:
    #     for function_call in function_calls:
    #         function_name = function_call.name
    #         function_to_call = available_functions[function_name]
    #         function_args = json.loads(function_call.arguments)
            
    #         # Lets call the Logic app with the function arguments to get the contract details.
    #         if asyncio.iscoroutinefunction(function_to_call):
    #             function_response = await function_to_call(**function_args)
    #         else:
    #             function_response = function_to_call(**function_args)
            
    #         # Add the model's text response to the messages (this already contains the function call details)
    #         input_messages.append({
    #             "role": "assistant",
    #             "content": [{"type": "output_text", "text": response.output_text}]
    #         })
            
    #         # Add the function call result as input_text
    #         input_messages.append({
    #             "role": "user",
    #             "content": [{"type": "input_text", "text": f"Function {function_name} result: {str(function_response)}"}]
    #         })
    # else:
    #     # If no function call, append the response message directly
    #     input_messages.append({
    #         "role": "assistant",
    #         "content": [{"type": "output_text", "text": response.output_text}]
    #     })

    # This is the final call to the Responses API with the input messages and tools
    response_2 = client.responses.create(
        model=config.model,
        instructions=instructions,
        input=input_messages,
        tools=tools_list,
    )
    print(response_2.output_text)
    # print("Response from the model:")
    # print(json.dumps(response_2, default=lambda o: o.__dict__, indent=4))

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
