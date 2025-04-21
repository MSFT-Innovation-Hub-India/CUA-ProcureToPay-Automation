import os
import sys
import argparse
import asyncio
from common.computer import Computer
from playwright.async_api import async_playwright
from common.utils import check_blocklisted_url
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
import base64
import json
load_dotenv()
from common.local_playwright import LocalPlaywrightComputer

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL = os.getenv("MODEL_NAME")
DISPLAY_WIDTH = 1024
DISPLAY_HEIGHT = 768
API_VERSION = "2025-03-01-preview"
ITERATIONS = 5
contract_data_url = os.getenv("contract_data_url")
invoice_data_url = os.getenv("invoice_data_url")

# Setup argument parser
parser = argparse.ArgumentParser(description='Run Computer Use Assistant with Playwright.')
parser.add_argument('--url', type=str, help='URL to navigate to automatically on startup', default=None)
parser.add_argument('--instructions', type=str, help='Instructions to execute after navigating to the URL', default=None)
parser.add_argument('--exit_after_completion', action='store_true', help='Exit program automatically after completing instructions')
args = parser.parse_args()

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version=API_VERSION
    )

def acknowledge_safety_check_callback(message: str) -> bool:
    response = input(
        f"Safety Check Warning: {message}\nDo you want to acknowledge and proceed? (y/n): "
    ).lower()
    return response.strip() == "y"

async def async_handle_item(item, computer: Computer):
    """Handle each item; may cause a computer action + screenshot."""
    if hasattr(item, 'type'):  # Handle new response format with attributes
        item_type = item.type
    elif isinstance(item, dict):  # Handle old response format with dict
        item_type = item.get("type")
    else:
        print(f"Unknown item format: {type(item)}")
        return []

    # Check if the model is asking about saving the form
    if item_type == "message":  # print messages
        if hasattr(item, 'content') and hasattr(item.content[0], 'text'):
            message_text = item.content[0].text.lower()
            print(message_text)
            
            # Check if the model is asking about saving the form
            if 'save' in message_text or "'save'" in message_text or '"save"' in message_text:
                print("Automatically responding 'yes' to save the form")
                return [{"role": "user", "content": "Yes, please save the form by clicking the save button"}]
                
        elif isinstance(item, dict) and 'content' in item:
            message_text = item["content"][0]["text"].lower() if isinstance(item["content"][0]["text"], str) else ""
            print(message_text)
            
            # Check if the model is asking about saving the form
            if "would you like me to save this information?" in message_text or "would you like me to proceed with saving" in message_text:
                print("Automatically responding 'yes' to save the form")
                return [{"role": "user", "content": "Yes, please save the form by clicking the save button"}]

    if item_type == "computer_call":  # perform computer actions
        if hasattr(item, 'action'):
            action = item.action
            action_type = action.type
            action_args = {k: v for k, v in vars(action).items() if k != "type"}
        else:
            action = item["action"]
            action_type = action["type"]
            action_args = {k: v for k, v in action.items() if k != "type"}
            
        print(f"{action_type}({action_args})")

        # Ensure the page has loaded completely before interacting with elements
        await computer.wait_for_load_state()
        
        # Convert synchronous actions to asynchronous
        if action_type == "click":
            if 'selector' in action_args:
                selector = action_args.get("selector")
                try:
                    success = await computer.click(selector)
                    if not success:
                        raise Exception("Click failed with provided selector")
                except Exception as e:
                    print(f"Error clicking element with selector {selector}: {e}")
                    # Try to find a better selector for input fields
                    field_name = selector.replace('#', '').replace('.', '')
                    try:
                        # Try common form field selectors
                        success = await computer.click(f"input[name='{field_name}']")
                        if not success:
                            raise Exception("Click failed with name selector")
                    except Exception as e2:
                        print(f"Retry failed: {e2}")
                        try:
                            success = await computer.click(f"input[id='{field_name}']")
                            if not success:
                                raise Exception("Click failed with id selector")
                        except Exception as e3:
                            print(f"Second retry failed: {e3}")
                            # Last attempt with coordinates if provided
                            if 'x' in action_args and 'y' in action_args:
                                x, y = action_args.get('x'), action_args.get('y')
                                try:
                                    # Use page.mouse.click for direct coordinate clicking
                                    await computer._page.mouse.click(x, y)
                                except Exception as e4:
                                    print(f"Coordinate click failed: {e4}")
            elif 'x' in action_args and 'y' in action_args:
                # Handle coordinate-based clicking
                x, y = action_args.get('x'), action_args.get('y')
                try:
                    await computer._page.mouse.click(x, y)
                    # Give the browser a moment to process the click
                    await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"Error clicking at coordinates ({x}, {y}): {e}")
        
        elif action_type == "type":
            text = action_args.get("text", "")
            if not text:
                print("No text to type")
            else:
                success = False
                
                # Try to determine if we're filling a form field
                if 'selector' in action_args:
                    selector = action_args.get("selector")
                    try:
                        # Try the enhanced clear and type method first
                        success = await computer.clear_and_type(selector, text)
                    except Exception as e:
                        print(f"Clear and type failed: {e}")
                        
                        try:
                            # Fall back to standard fill
                            success = await computer.fill(selector, text)
                        except Exception as e2:
                            print(f"Fill failed: {e2}")
                            
                            # Try with common field selectors
                            field_name = selector.replace('#', '').replace('.', '')
                            try:
                                success = await computer.clear_and_type(f"input[name='{field_name}']", text)
                            except Exception as e3:
                                print(f"Clear and type by name failed: {e3}")
                                try:
                                    success = await computer.clear_and_type(f"input[id='{field_name}']", text)
                                except Exception as e4:
                                    print(f"Clear and type by id failed: {e4}")
                
                # If we have coordinates, try focus and type approach
                if not success and 'x' in action_args and 'y' in action_args:
                    x, y = action_args.get('x'), action_args.get('y')
                    try:
                        success = await computer.focus_and_type(x, y, text)
                    except Exception as e:
                        print(f"Focus and type at coordinates failed: {e}")
                
                # Last resort: try to type into whatever is currently focused
                if not success:
                    try:
                        # First click to ensure focus
                        if 'x' in action_args and 'y' in action_args:
                            x, y = action_args.get('x'), action_args.get('y')
                            await computer._page.mouse.click(x, y)
                        
                        # Wait a moment for focus
                        await asyncio.sleep(0.5)
                        
                        # Try to select all existing text and delete it
                        await computer._page.keyboard.press("Control+a")
                        await computer._page.keyboard.press("Delete")
                        
                        # Type with delay between keystrokes
                        await computer._page.keyboard.type(text, delay=100)
                        success = True
                    except Exception as e:
                        print(f"Last resort typing failed: {e}")
                        
                if not success:
                    print("WARNING: All typing methods failed")
                    # Try JavaScript as a final approach
                    try:
                        # Try to identify the active element and set its value via JavaScript
                        js_result = await computer.evaluate(f'''
                            (function() {{
                                let activeElement = document.activeElement;
                                if (activeElement && (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA')) {{
                                    activeElement.value = '{text}';
                                    return true;
                                }}
                                return false;
                            }})()
                        ''')
                        if js_result:
                            print("Successfully set input value using JavaScript")
                    except Exception as e:
                        print(f"JavaScript fallback failed: {e}")
        
        elif action_type == "goto":
            url = action_args.get("url")
            if url:
                try:
                    await computer.goto(url)
                    # Wait for page to load fully
                    await computer.wait_for_load_state()
                except Exception as e:
                    print(f"Error navigating to URL: {e}")
        
        elif action_type == "wait":
            # Wait could be due to page navigation after form submission
            # Give some time for the page to load
            await asyncio.sleep(1)

        # Take screenshot
        try:
            screenshot_bytes = await computer.screenshot()
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            return []

        # Remember the current URL for the next action to detect navigation
        if action_type == "click" or action_type == "goto":
            action_args["prev_url"] = computer._page.url

        # Get pending checks based on response format
        if hasattr(item, 'pending_safety_checks'):
            pending_checks = item.pending_safety_checks
        else:
            pending_checks = item.get("pending_safety_checks", [])
            
        for check in pending_checks:
            message = check.message if hasattr(check, 'message') else check["message"]
            if not acknowledge_safety_check_callback(message):
                raise ValueError(f"Safety check failed: {message}")

        # return value informs model of the latest screenshot
        call_output = {
            "type": "computer_call_output",
            "call_id": item.call_id if hasattr(item, 'call_id') else item["call_id"],
            "acknowledged_safety_checks": pending_checks,
            "output": {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{screenshot_base64}",
            },
        }

        # additional URL safety checks for browser environments
        try:
            # Use the computer instance to get the current URL
            current_url = computer._page.url
            call_output["output"]["current_url"] = current_url
            check_blocklisted_url(current_url)
        except Exception as e:
            print(f"Error getting current URL: {e}")

        return [call_output]

    return []

async def post_purchase_invoice_header(instructions: str):
    """
    Automates the process of creating a purchase invoice header using a Computer Use Assistant (CUA) with Playwright.
    This function navigates to a specified URL and follows given instructions to fill and submit a purchase invoice form.
    It continuously monitors the process until successful form submission is detected through URL change.
    Args:
        instructions (str): User instructions for filling out the purchase invoice form.
    Returns:
        None: The function returns None but prints success messages upon completion.
    Raises:
        ValueError: If no output is received from the model response.
    Notes:
        - The function uses LocalPlaywrightComputer for browser automation
        - Success is determined by detecting navigation from a URL containing '/create' to one that doesn't
        - Upon successful submission, captures and encodes a screenshot of the result
        - Implements a loop that continues until form submission is confirmed
        - Handles both synchronous and asynchronous operations for form filling

    """

    async with LocalPlaywrightComputer() as computer:
        tools = [
            {
                "type": "computer-preview",
                "display_width": computer.dimensions[0],
                "display_height": computer.dimensions[1],
                "environment": computer.environment,
            }
        ]

        items = []
        initial_url = invoice_data_url
        await computer.goto(invoice_data_url)
        user_input = instructions
        
        # Flag to track whether form submission was successful
        form_submitted_successfully = False
        
        # Start the form filling process
        items.append({"role": "user", "content": user_input})

        while not form_submitted_successfully:  # continue until successful completion
            response = client.responses.create(
                model="computer-use-preview",
                input=items,
                tools=tools,
                truncation="auto",
            )
            
            # Access the output items directly from response.output
            if not hasattr(response, 'output') or not response.output:
                raise ValueError("No output from model")

            items += response.output

            # Process each item in the output
            new_items = []
            for item in response.output:
                # Before processing the item, check if we've navigated away from the initial URL
                # This would indicate a successful form submission
                if hasattr(computer, '_page') and computer._page:
                    current_url = computer._page.url
                    if initial_url != current_url and "/create" in initial_url.lower() and "/create" not in current_url.lower():
                        print(f"\n✅ SUCCESS: Purchase invoice was created successfully!")
                        print(f"Navigation detected from {initial_url} to {current_url}")
                        form_submitted_successfully = True
                        
                        # Create a success output and add it to new_items
                        screenshot_bytes = await computer.screenshot()
                        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                        
                        success_output = {
                            "type": "computer_call_output",
                            "call_id": item.call_id if hasattr(item, 'call_id') else "success_detected",
                            "acknowledged_safety_checks": [],
                            "output": {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{screenshot_base64}",
                                "current_url": current_url,
                                "success": True,
                                "message": "Purchase invoice header was created successfully!"
                            },
                        }
                        
                        new_items.append(success_output)
                        break
                
                # Process the item normally if no navigation was detected
                if not form_submitted_successfully:
                    result = await async_handle_item(item, computer)
                    if result:
                        new_items.extend(result)
            
            if new_items:
                items.extend(new_items)
                
            # If form submission was successful, exit the loop
            if form_submitted_successfully:
                print("Task completed successfully. Invoice created.")
                return
            
            # Check if we received a final assistant message but no success was detected
            if items and isinstance(items[-1], dict) and items[-1].get("role") == "assistant":
                # If we reach here, we got a final assistant message but no success detection
                # This may happen if the model completes its response without detecting navigation
                # Ask the model to continue with form submission if needed
                items.append({"role": "user", "content": "Please continue with form filling and submission."})


async def retrieve_contract(contractid:str, instructions: str):
    """
    Asynchronously retrieves the contract header and contract details through web automation.
    This function navigates to a specified URL, follows given instructions to get the data on the page
    in the form of a JSON document. It uses Playwright for web automation.
    Args:
        contractid (str): The id of the contract for which the data is to be retrieved.
        instructions (str): User instructions for processing the data on this page.
    Returns:
        str: JSON string containing the contract data extracted from the page.
    Raises:
        ValueError: If no output is received from the model.
    """

    async with LocalPlaywrightComputer() as computer:
        tools = [
            {
                "type": "computer-preview",
                "display_width": computer.dimensions[0],
                "display_height": computer.dimensions[1],
                "environment": computer.environment,
            }
        ]

        items = []
        contract_url = contract_data_url + f"/{contractid}"
        print(f"Navigating to contract URL: {contract_url}")
        await computer.goto(contract_url)
        
        # Wait for page to load completely
        await computer.wait_for_load_state()
        
        # Take a screenshot to ensure the page content is captured
        screenshot_bytes = await computer.screenshot()
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        # Create very clear and specific instructions for the model
        user_input = "You are currently viewing a contract details page. Please extract ALL data visible on this page into a JSON format. Include all field names and values. Format the response as a valid JSON object with no additional text before or after."

        # Start the conversation with the screenshot and clear instructions - format fixed for image_url
        items.append({
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_input},
                {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_base64}"}
            ]
        })
        
        # Track if we received JSON data
        json_data = None
        max_iterations = 3  # Limit iterations to avoid infinite loops
        current_iteration = 0
        
        while json_data is None and current_iteration < max_iterations:
            current_iteration += 1
            print(f"Iteration {current_iteration} of {max_iterations}")
            
            response = client.responses.create(
                model="computer-use-preview",
                input=items,
                tools=tools,
                truncation="auto",
            )
            
            # Access the output items directly from response.output
            if not hasattr(response, 'output') or not response.output:
                raise ValueError("No output from model")

            print(f"Response: {response.output}")
            items += response.output

            # Process each item in the output
            new_items = []
            for item in response.output:
                # Process computer calls to capture screenshots
                if (hasattr(item, 'type') and item.type == "computer_call") or \
                   (isinstance(item, dict) and item.get("type") == "computer_call"):
                    result = await async_handle_item(item, computer)
                    if result:
                        new_items.extend(result)
                
                # Check for messages that might contain JSON data
                if (hasattr(item, 'type') and item.type == "message") or \
                   (isinstance(item, dict) and item.get("type") == "message"):
                    # Get content based on item structure
                    if hasattr(item, 'content'):
                        # Handle new response format
                        if hasattr(item.content[0], 'text'):
                            content = item.content[0].text
                        else:
                            content = ""
                    else:
                        # Handle dictionary format
                        content = item.get("content", [{}])[0].get("text", "")
                    
                    # Try to extract JSON from the response
                    try:
                        # Look for JSON-like content in the message
                        json_start = content.find('{')
                        json_end = content.rfind('}')
                        
                        if json_start >= 0 and json_end > json_start:
                            potential_json = content[json_start:json_end+1]
                            # Try to parse it as JSON
                            parsed_json = json.loads(potential_json)
                            json_data = potential_json
                            print("✅ Successfully extracted JSON data from response")
                            break  # Exit the item processing loop if JSON found
                    except json.JSONDecodeError:
                        # Try alternative JSON extraction methods
                        try:
                            # Look for code block markers
                            if "```json" in content:
                                json_block = content.split("```json")[1].split("```")[0].strip()
                                parsed_json = json.loads(json_block)
                                json_data = json_block
                                print("✅ Successfully extracted JSON data from code block")
                                break
                            elif "```" in content:
                                # Try to find any code block that might contain JSON
                                code_blocks = content.split("```")
                                for i in range(1, len(code_blocks), 2):
                                    block = code_blocks[i].strip()
                                    # Skip the language identifier line if present
                                    if block.startswith("json"):
                                        block = block[4:].strip()
                                    try:
                                        parsed_json = json.loads(block)
                                        json_data = block
                                        print("✅ Successfully extracted JSON data from generic code block")
                                        break
                                    except:
                                        continue
                                if json_data:
                                    break  # Exit the item processing loop if JSON found
                        except (IndexError, json.JSONDecodeError):
                            pass  # JSON not found in this format either
            
            if new_items:
                items.extend(new_items)
                
            # If JSON data was found, exit the loop
            if json_data:
                print("Contract data retrieved successfully")
                return json_data
            
            # If we're not on the last iteration, try again with more explicit instructions
            if current_iteration < max_iterations:
                # Take a fresh screenshot for the next iteration
                screenshot_bytes = await computer.screenshot()
                screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                
                # Craft a more explicit instruction for the next attempt
                if current_iteration == 1:
                    # First retry: be very explicit about the task
                    retry_message = "Look at the screenshot carefully. You are seeing a contract details page. Extract ALL data visible on the page as a JSON object. Format your entire response as a valid JSON object only, with field names and values from the page."
                else:
                    # Final retry: even more explicit
                    retry_message = "ONLY respond with a JSON object containing the data from the page. Look at every field and value on the screen. Do not include any explanatory text. Your entire response should be valid JSON that parses correctly."
                
                # Fixed image URL format here too
                items.append({
                    "role": "user", 
                    "content": [
                        {"type": "input_text", "text": retry_message},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_base64}"}
                    ]
                })
                
        # If we couldn't extract JSON after all attempts, create a simple JSON with error message
        if not json_data:
            print("Could not extract valid JSON data from contract page")
            # Try one last approach - manually extract data from the page using JavaScript
            try:
                # Execute JavaScript to extract form data
                extracted_data = await computer.evaluate('''
                    (function() {
                        // Try to collect all input fields, select fields, and their values
                        const data = {};
                        const labels = document.querySelectorAll('label');
                        labels.forEach(label => {
                            const text = label.textContent.trim();
                            const forAttr = label.getAttribute('for');
                            if (forAttr) {
                                const input = document.getElementById(forAttr);
                                if (input) {
                                    data[text] = input.value || input.textContent;
                                }
                            }
                        });

                        // Also try to find data in dt/dd pairs (common definition list pattern)
                        const dts = document.querySelectorAll('dt');
                        dts.forEach(dt => {
                            const dd = dt.nextElementSibling;
                            if (dd && dd.tagName === 'DD') {
                                data[dt.textContent.trim()] = dd.textContent.trim();
                            }
                        });

                        // Try to find tables with data
                        const tables = document.querySelectorAll('table');
                        tables.forEach((table, tableIndex) => {
                            const tableData = [];
                            const rows = table.querySelectorAll('tr');
                            rows.forEach(row => {
                                const rowData = {};
                                const cells = row.querySelectorAll('td, th');
                                cells.forEach((cell, index) => {
                                    rowData[`col${index}`] = cell.textContent.trim();
                                });
                                if (Object.keys(rowData).length > 0) {
                                    tableData.push(rowData);
                                }
                            });
                            if (tableData.length > 0) {
                                data[`table${tableIndex}`] = tableData;
                            }
                        });

                        // Look for any displayed field-value pairs
                        const divs = document.querySelectorAll('div');
                        divs.forEach(div => {
                            const text = div.textContent.trim();
                            if (text.includes(':')) {
                                const parts = text.split(':');
                                if (parts.length === 2) {
                                    data[parts[0].trim()] = parts[1].trim();
                                }
                            }
                        });

                        return data;
                    })()
                ''')
                
                if extracted_data and isinstance(extracted_data, dict) and len(extracted_data) > 0:
                    print("✅ Successfully extracted data using JavaScript")
                    return json.dumps(extracted_data)
            except Exception as e:
                print(f"JavaScript extraction failed: {e}")
            
            # Fall back to a minimal error object
            return json.dumps({"error": "Could not extract valid JSON data from contract page", "contractId": contractid})

async def main():
    """Run the CUA (Computer Use Assistant) loop, using Local Playwright."""
    async with LocalPlaywrightComputer() as computer:
        tools = [
            {
                "type": "computer-preview",
                "display_width": computer.dimensions[0],
                "display_height": computer.dimensions[1],
                "environment": computer.environment,
            }
        ]

        items = []
        initial_url = args.url
        await computer.goto(args.url)
        user_input = args.instructions if args.instructions else "Please fill the form."
        
        # Flag to track whether form submission was successful
        form_submitted_successfully = False
        
        # Start the form filling process
        items.append({"role": "user", "content": user_input})

        while not form_submitted_successfully:  # continue until successful completion
            response = client.responses.create(
                model="computer-use-preview",
                input=items,
                tools=tools,
                truncation="auto",
            )
            
            # Access the output items directly from response.output
            if not hasattr(response, 'output') or not response.output:
                raise ValueError("No output from model")

            items += response.output

            # Process each item in the output
            new_items = []
            for item in response.output:
                # Before processing the item, check if we've navigated away from the initial URL
                # This would indicate a successful form submission
                if hasattr(computer, '_page') and computer._page:
                    current_url = computer._page.url
                    if initial_url != current_url and "/create" in initial_url.lower() and "/create" not in current_url.lower():
                        print(f"\n✅ SUCCESS: Purchase invoice was created successfully!")
                        print(f"Navigation detected from {initial_url} to {current_url}")
                        form_submitted_successfully = True
                        
                        # Create a success output and add it to new_items
                        screenshot_bytes = await computer.screenshot()
                        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                        
                        success_output = {
                            "type": "computer_call_output",
                            "call_id": item.call_id if hasattr(item, 'call_id') else "success_detected",
                            "acknowledged_safety_checks": [],
                            "output": {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{screenshot_base64}",
                                "current_url": current_url,
                                "success": True,
                                "message": "Purchase invoice header was created successfully!"
                            },
                        }
                        
                        new_items.append(success_output)
                        break
                
                # Process the item normally if no navigation was detected
                if not form_submitted_successfully:
                    result = await async_handle_item(item, computer)
                    if result:
                        new_items.extend(result)
            
            if new_items:
                items.extend(new_items)
                
            # If form submission was successful, exit the loop
            if form_submitted_successfully:
                print("Task completed successfully. Invoice created.")
                return
            
            # Check if we received a final assistant message but no success was detected
            if items and isinstance(items[-1], dict) and items[-1].get("role") == "assistant":
                # If we reach here, we got a final assistant message but no success detection
                # This may happen if the model completes its response without detecting navigation
                # Ask the model to continue with form submission if needed
                items.append({"role": "user", "content": "Please continue with form filling and submission."})


if __name__ == "__main__":
    # if len(sys.argv) == 1:
    #     print("Usage examples:")
    #     print(f"  python {sys.argv[0]} --url https://p2p-erp-web.gentleflower-4e1ad251.swedencentral.azurecontainerapps.io/PurchaseInvoiceHeaders/Create")
    #     print(f"  python {sys.argv[0]} --url https://p2p-erp-web.gentleflower-4e1ad251.swedencentral.azurecontainerapps.io/PurchaseInvoiceHeaders/Create --instructions \"fill the form with purchase_invoice_no 'PInv_001', contract_reference 'contract997801', supplier_id 'supplier99010', total_invoice_value 23100.00, invoice_date '12/12/2024', status 'approved', remarks 'invoice is valid and approved'. Save by clicking on the 'save' button.\"")
    
    # asyncio.run(main())
    
    
    
    # invoice_creation_url = "https://p2p-erp-web.gentleflower-4e1ad251.swedencentral.azurecontainerapps.io/PurchaseInvoiceHeaders/Create"
    # invoice_creation_instructions = "fill the form with purchase_invoice_no 'PInv_001', contract_reference 'contract997801', supplier_id 'supplier99010', total_invoice_value 23100.00, invoice_date '12/12/2024', status 'approved', remarks 'invoice is valid and approved'. Save by clicking on the 'save' button."
    # asyncio.run(post_purchase_invoice_header( invoice_creation_instructions))
    
    
    contract_header_url = "https://p2p-erp-web.gentleflower-4e1ad251.swedencentral.azurecontainerapps.io/ContractHeaders/ContractLines"
    contract_id = "CON000001"
    contract_instructions = "get the details of the Contract Header along with the contract lines on the page as a JSON output"
    asyncio.run(retrieve_contract(contract_id, contract_instructions))