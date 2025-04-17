import json
import os
import asyncio
import base64
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from playwright.async_api import async_playwright, TimeoutError
import os
from dotenv import load_dotenv
load_dotenv()

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

# Configuration

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL = "computer-use-preview"
DISPLAY_WIDTH = 1024
DISPLAY_HEIGHT = 768
API_VERSION = "2025-03-01-preview"
ITERATIONS = 5 # Max number of iterations before forcing the model to return control to the human supervisor

# Key mapping for special keys in Playwright
KEY_MAPPING = {
    "/": "Slash", "\\": "Backslash", "alt": "Alt", "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft", "arrowright": "ArrowRight", "arrowup": "ArrowUp",
    "backspace": "Backspace", "ctrl": "Control", "delete": "Delete", 
    "enter": "Enter", "esc": "Escape", "shift": "Shift", "space": " ",
    "tab": "Tab", "win": "Meta", "cmd": "Meta", "super": "Meta", "option": "Alt"
}

def validate_coordinates(x, y):
    """Ensure coordinates are within display bounds."""
    return max(0, min(x, DISPLAY_WIDTH)), max(0, min(y, DISPLAY_HEIGHT))

async def handle_action(page, action):
    """Handle different action types from the model."""
    action_type = action.type
    
    if action_type == "drag":
        print("Drag action is not supported in this implementation. Skipping.")
        return
        
    elif action_type == "click":
        button = getattr(action, "button", "left")
        # Validate coordinates
        x, y = validate_coordinates(action.x, action.y)
        
        print(f"\tAction: click at ({x}, {y}) with button '{button}'")
        
        if button == "back":
            await page.go_back()
        elif button == "forward":
            await page.go_forward()
        elif button == "wheel":
            await page.mouse.wheel(x, y)
        else:
            button_type = {"left": "left", "right": "right", "middle": "middle"}.get(button, "left")
            await page.mouse.click(x, y, button=button_type)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=3000)
            except TimeoutError:
                pass
        
    elif action_type == "double_click":
        # Validate coordinates
        x, y = validate_coordinates(action.x, action.y)
        
        print(f"\tAction: double click at ({x}, {y})")
        await page.mouse.dblclick(x, y)
        
    elif action_type == "scroll":
        scroll_x = getattr(action, "scroll_x", 0)
        scroll_y = getattr(action, "scroll_y", 0)
        # Validate coordinates
        x, y = validate_coordinates(action.x, action.y)
        
        print(f"\tAction: scroll at ({x}, {y}) with offsets ({scroll_x}, {scroll_y})")
        await page.mouse.move(x, y)
        await page.evaluate(f"window.scrollBy({{left: {scroll_x}, top: {scroll_y}, behavior: 'smooth'}});")
        
    elif action_type == "keypress":
        keys = getattr(action, "keys", [])
        print(f"\tAction: keypress {keys}")
        mapped_keys = [KEY_MAPPING.get(key.lower(), key) for key in keys]
        
        if len(mapped_keys) > 1:
            # For key combinations (like Ctrl+C)
            for key in mapped_keys:
                await page.keyboard.down(key)
            await asyncio.sleep(0.1)
            for key in reversed(mapped_keys):
                await page.keyboard.up(key)
        else:
            for key in mapped_keys:
                await page.keyboard.press(key)
                
    elif action_type == "type":
        text = getattr(action, "text", "")
        print(f"\tAction: type text: {text}")
        await page.keyboard.type(text, delay=20)
        
    elif action_type == "wait":
        ms = getattr(action, "ms", 1000)
        print(f"\tAction: wait {ms}ms")
        await asyncio.sleep(ms / 1000)
        
    elif action_type == "screenshot":
        print("\tAction: screenshot")
        
    else:
        print(f"\tUnrecognized action: {action_type}")

async def take_screenshot(page):
    """Take a screenshot and return base64 encoding with caching for failures."""
    global last_successful_screenshot
    
    try:
        screenshot_bytes = await page.screenshot(full_page=False)
        last_successful_screenshot = base64.b64encode(screenshot_bytes).decode("utf-8")
        return last_successful_screenshot
    except Exception as e:
        print(f"Screenshot failed: {e}")
        print(f"Using cached screenshot from previous successful capture")
        if last_successful_screenshot:
            return last_successful_screenshot


async def process_model_response(client, response, page, max_iterations=ITERATIONS):
    """Process the model's response and execute actions."""
    response_string = ""
    json_content = None
    
    for iteration in range(max_iterations):
        if not hasattr(response, 'output') or not response.output:
            print("No output from model.")
            break
        
        # Safely access response id
        response_id = getattr(response, 'id', 'unknown')
        print(f"\nIteration {iteration + 1} - Response ID: {response_id}\n")
        
        # Track whether we have actions to execute
        has_actions = False
        
        # Process each output item from the model
        for item in response.output:
            # Handle text messages
            if hasattr(item, 'type') and item.type == "message":
                text_content = item.content[0].text
                print(f"\nModel message: {text_content}\n")
                response_string += text_content + "\n"
                
                # Check for JSON in the response
                if "```json" in text_content:
                    json_start = text_content.find("```json") + 7
                    json_end = text_content.find("```", json_start)
                    if json_end > json_start:
                        json_content = text_content[json_start:json_end].strip()
                        print(f"JSON content extracted: {json_content}")
            
            # Handle computer actions
            elif hasattr(item, 'type') and item.type == "computer_use_actions":
                has_actions = True
                print(f"Processing {len(item.actions)} computer actions...")
                for action in item.actions:
                    print(f"Executing action: {action.type}")
                    try:
                        await handle_action(page, action)
                        # Small delay after each action to let the page update
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        print(f"Error executing action {action.type}: {str(e)}")
        
        # If the model provided no actions but gave us a message with JSON, we can return
        if json_content and not has_actions:
            print("Found JSON content in response and no more actions, returning JSON")
            return json_content
        
        # If the model provided no actions and no JSON, but gave us some text response, we can check if it's done
        if not has_actions and response_string and not json_content:
            # If the response looks like it might contain contract data, return it
            if any(marker in response_string.lower() for marker in 
                  ['contractheader', 'contract header', 'contractlines', 'contract lines', 
                   'description', 'quantity', 'price', 'supplier', '"id"']):
                print("Response contains contract data markers, returning text")
                return response_string
            else:
                print("No more actions to execute, but response doesn't contain contract data")
                break
        
        # Take a new screenshot and send it back to the model for the next iteration
        if iteration < max_iterations - 1 and has_actions:
            try:
                # Take a new screenshot
                screenshot_base64 = await take_screenshot(page)
                
                # Send the screenshot to the model to continue the interaction
                print("Sending updated screenshot to model...")
                response = client.responses.create(
                    model=MODEL,
                    tools=[{
                        "type": "computer_use_preview",
                        "display_width": DISPLAY_WIDTH,
                        "display_height": DISPLAY_HEIGHT,
                        "environment": "browser"
                    }],
                    input=[{
                        "role": "user",
                        "content": [{
                            "type": "input_text",
                            "text": "Continue extracting ALL the contract header and contract line items data and return it as a structured JSON object."
                        }, {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{screenshot_base64}"
                        }]
                    }],
                    reasoning={"generate_summary": "concise"},
                    truncation="auto"
                )
            except Exception as e:
                print(f"Error sending updated screenshot to model: {str(e)}")
                break
    
    if json_content:
        return json_content
    
    if response_string:
        # Do a final check for JSON-like content in the response string
        if response_string.strip().startswith("{") and response_string.strip().endswith("}"):
            try:
                # Try to parse and reformat any JSON-like content
                parsed = json.loads(response_string.strip())
                return json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                pass
        return response_string
    
    if iteration >= max_iterations - 1:
        print("Reached maximum number of iterations. Stopping.")
    
    return response_string

async def retrieve_contract(contract_id:str) -> str:  
    """
    get the contract details for the given contract_id

    :param contract_id (str): The contract id against which the Supplier fulfils order and raises the Purchase Invoice.
    :return: retrieved contract details.
    :rtype: Any
    """
    response_string = None
    # Initialize OpenAI client
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version=API_VERSION
    )
    
    # Validate contract_id
    if not contract_id or not isinstance(contract_id, str):
        error_msg = f"Invalid contract_id: {contract_id}. Must be a non-empty string."
        print(error_msg)
        return json.dumps({"error": error_msg, "status": "failed"})
    
    try:
        print(f"Starting contract retrieval process for contract ID: {contract_id}")
        
        # Initialize Playwright
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=False,
                args=[f"--window-size={DISPLAY_WIDTH},{DISPLAY_HEIGHT}", "--disable-extensions"]
            )
            
            context = await browser.new_context(
                viewport={"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT},
                accept_downloads=True
            )
            
            page = await context.new_page()
            
            # Navigate to starting page with enhanced wait and retry logic
            try:
                print(f"Navigating to contract page for ID: {contract_id}")
                
                # Navigate with longer timeout and wait for network idle
                await page.goto(
                    f"https://p2p-erp-web.gentleflower-4e1ad251.swedencentral.azurecontainerapps.io/ContractHeaders/ContractLines/{contract_id}", 
                    wait_until="networkidle",  # Wait until network is idle (improved from domcontentloaded)
                    timeout=60000  # Extended timeout to 60 seconds to ensure full page load
                )
                
                # Enhanced waiting for content to be visible
                try:
                    # Wait for table elements that would contain contract data
                    await page.wait_for_selector("table", timeout=30000)
                    print("Tables detected on page, waiting for data to fully load...")
                    
                    # Wait for specific content that would indicate data is loaded
                    # Adjust these selectors based on the actual page structure
                    await page.wait_for_selector("table tbody tr", timeout=30000)
                    print("Table rows detected, contract data appears to be loaded")
                    
                    # Additional wait to ensure any AJAX or delayed rendering completes
                    await asyncio.sleep(2)
                    
                except Exception as wait_error:
                    print(f"Warning: Could not verify all content loaded: {wait_error}")
                    # Continue anyway - we'll take a screenshot and let the CUA model determine if data is visible
                    
                print("Successfully navigated to contract page and waited for content")
            except Exception as e:
                error_msg = f"Failed to navigate to contract page: {str(e)}"
                print(error_msg)
                await context.close()
                await browser.close()
                return json.dumps({"error": error_msg, "status": "failed"})
            
            # Main interaction loop
            waiting_for_data = True
            retry_count = 0
            max_retries = 3
            
            try:
                while waiting_for_data and retry_count < max_retries:
                    print(f"\n{'='*50}\nAttempt {retry_count + 1} of {max_retries} to extract contract data")
                    
                    # Take initial screenshot with verification
                    try:
                        # First verify that we can see content on the page before taking screenshot
                        content_visible = await page.evaluate("""() => {
                            // Check if there are tables with data
                            const tables = document.querySelectorAll('table');
                            if (tables.length === 0) return false;
                            
                            // Check if tables have rows
                            for (const table of tables) {
                                if (table.querySelectorAll('tbody tr').length > 0) {
                                    return true;
                                }
                            }
                            return false;
                        }""")
                        
                        if not content_visible:
                            print("Warning: Tables or data rows not yet visible on page")
                            # Try refreshing the page to trigger content load
                            if retry_count > 0:
                                await page.reload(wait_until="networkidle", timeout=30000)
                                await asyncio.sleep(3)  # Additional wait after reload
                        
                        # Take screenshot after content check
                        screenshot_base64 = await take_screenshot(page)
                        print("Successfully captured screenshot with contract data")
                    except Exception as e:
                        error_msg = f"Failed to capture screenshot: {str(e)}"
                        retry_count += 1
                        await asyncio.sleep(2)
                        continue
                    
                    l_instructions = """
                    You are an AI agent with the ability to control a browser. You can control the keyboard and mouse. You take a screenshot after each action to check if your action was successful.
                    
                    Your current task:
                    1. You are looking at a contract details page that contains contract header information and contract line items.
                    2. Extract ALL the data from both the contract header section and the contract lines section.
                    3. Return the data as a structured JSON object with two main sections: "contractHeader" and "contractLines".
                    4. For contract lines, make sure to capture all columns and all rows.
                    5. If you need to scroll to see all data, please do so.
                    
                    Once you have completed the data extraction, return the JSON response and stop running.
                    """
                    
                    # Initial request to the model with enhanced instructions
                    try:
                        print("Sending model instructions and screenshot...")
                        response = client.responses.create(
                            model=MODEL,
                            tools=[{
                                "type": "computer_use_preview",
                                "display_width": DISPLAY_WIDTH,
                                "display_height": DISPLAY_HEIGHT,
                                "environment": "browser"
                            }],
                            instructions=l_instructions,
                            input=[{
                                "role": "user",
                                "content": [{
                                    "type": "input_text",
                                    "text": "Extract ALL the contract header and contract line items data from this page and return it as a structured JSON object. Make sure to include every field and value visible on the page."
                                }, {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{screenshot_base64}"
                                }]
                            }],
                            reasoning={"generate_summary": "concise"},
                            truncation="auto"
                        )
                        print("Successfully sent model instructions and screenshot")
                    except Exception as e:
                        error_msg = f"Failed to create model response: {str(e)}"
                        print(error_msg)
                        retry_count += 1
                        await asyncio.sleep(2)
                        continue

                    # Process model response using the enhanced processor that handles actions
                    try:
                        response_string = await process_model_response(client, response, page)
                        
                        # Debug logging - truncated to avoid too much console output
                        if response_string:
                            print(f"CONTRACT DATA (first 200 chars): {response_string[:200]}...")
                        else:
                            print("No response string received from model")
                        
                        if response_string:
                            # Check if response contains actual data by looking for specific markers
                            has_data = any(marker in response_string.lower() for marker in 
                                ['contractheader', 'contract header', 'contractlines', 'contract lines', 
                                 'description', 'quantity', 'price', 'supplier'])
                            
                            if has_data:
                                print("Successfully retrieved contract details with data")
                                
                                # Display full contract data in a formatted way
                                try:
                                    # Try to parse as JSON for pretty printing
                                    if response_string.strip().startswith("{"):
                                        try:
                                            contract_json = json.loads(response_string)
                                            
                                            # Print contract header data
                                            print("\n" + "="*80)
                                            print("CONTRACT HEADER DATA:")
                                            print("="*80)
                                            if 'contractHeader' in contract_json:
                                                for key, value in contract_json['contractHeader'].items():
                                                    print(f"{key}: {value}")
                                            elif 'ContractHeader' in contract_json:
                                                for key, value in contract_json['ContractHeader'].items():
                                                    print(f"{key}: {value}")
                                            else:
                                                print("Contract header data not found in expected format")
                                                
                                            # Print contract line items
                                            print("\n" + "="*80)
                                            print("CONTRACT LINE ITEMS:")
                                            print("="*80)
                                            if 'contractLines' in contract_json:
                                                for i, line in enumerate(contract_json['contractLines']):
                                                    print(f"\nLine Item #{i+1}:")
                                                    if isinstance(line, dict):
                                                        for key, value in line.items():
                                                            print(f"  {key}: {value}")
                                                    else:
                                                        print(f"  {line}")
                                            elif 'ContractLines' in contract_json:
                                                for i, line in enumerate(contract_json['ContractLines']):
                                                    print(f"\nLine Item #{i+1}:")
                                                    if isinstance(line, dict):
                                                        for key, value in line.items():
                                                            print(f"  {key}: {value}")
                                                    else:
                                                        print(f"  {line}")
                                            else:
                                                print("Contract line items not found in expected format")
                                        except json.JSONDecodeError:
                                            # If not valid JSON, print the raw response
                                            print("\n" + "="*80)
                                            print("CONTRACT DATA (RAW FORMAT - COULD NOT PARSE AS JSON):")
                                            print("="*80)
                                            print(response_string)
                                    else:
                                        # If not JSON format, print the raw response
                                        print("\n" + "="*80)
                                        print("CONTRACT DATA (RAW FORMAT):")
                                        print("="*80)
                                        print(response_string)
                                except Exception as print_error:
                                    print(f"Error displaying contract data: {print_error}")
                                    print("Raw response data:", response_string)
                                
                                waiting_for_data = False
                                
                                # Try to validate and clean up JSON if it looks like JSON
                                if response_string.strip().startswith("{"):
                                    try:
                                        json_obj = json.loads(response_string)
                                        return json.dumps(json_obj, indent=2)
                                    except json.JSONDecodeError:
                                        print("Response looks like JSON but isn't valid. Returning as-is.")
                                
                                return response_string
                            else:
                                print("Response doesn't contain expected contract data markers, retrying...")
                                retry_count += 1
                                await asyncio.sleep(3)  # Longer wait before retry
                        else:
                            print("Empty response received from model, retrying...")
                            retry_count += 1
                            # If we get an empty response, try refreshing the page
                            await page.reload(wait_until="networkidle", timeout=30000)
                            await asyncio.sleep(3)
                    except Exception as e:
                        error_msg = f"Error processing model response: {str(e)}"
                        print(error_msg)
                        retry_count += 1
                        await asyncio.sleep(2)
                
                if retry_count >= max_retries:
                    # Final attempt with a more direct approach
                    print("Regular attempts failed. Trying direct DOM extraction...")
                    
                    try:
                        # Directly extract data from DOM as fallback
                        contract_data = await page.evaluate("""() => {
                            // Helper function to extract table data
                            function extractTableData(tableSelector) {
                                const table = document.querySelector(tableSelector);
                                if (!table) return [];
                                
                                const rows = table.querySelectorAll('tbody tr');
                                const headers = Array.from(table.querySelectorAll('thead th')).map(th => th.innerText.trim());
                                
                                return Array.from(rows).map(row => {
                                    const cells = row.querySelectorAll('td');
                                    const rowData = {};
                                    
                                    headers.forEach((header, index) => {
                                        if (cells[index]) {
                                            rowData[header] = cells[index].innerText.trim();
                                        }
                                    });
                                    
                                    return rowData;
                                });
                            }
                            
                            // Extract header info from all visible label-value pairs
                            const headerData = {};
                            document.querySelectorAll('.row').forEach(row => {
                                const labels = row.querySelectorAll('label, strong, b');
                                labels.forEach(label => {
                                    const labelText = label.innerText.trim();
                                    if (labelText) {
                                        // Try to find the value right after the label
                                        let valueElement = label.nextSibling;
                                        while (valueElement && valueElement.nodeType === 3) {
                                            valueElement = valueElement.nextSibling;
                                        }
                                        
                                        // If found, extract the value
                                        if (valueElement) {
                                            headerData[labelText] = valueElement.innerText.trim();
                                        }
                                    }
                                });
                            });
                            
                            // Extract all tables we can find
                            const tables = document.querySelectorAll('table');
                            const lineItems = Array.from(tables).flatMap(table => 
                                Array.from(table.querySelectorAll('tbody tr')).map(row => {
                                    const cells = row.querySelectorAll('td');
                                    return Array.from(cells).map(cell => cell.innerText.trim());
                                })
                            );
                            
                            return {
                                contractHeader: headerData,
                                contractLines: lineItems
                            };
                        }""")
                        
                        if contract_data and (contract_data.get('contractHeader') or contract_data.get('contractLines')):
                            print("Successfully extracted contract data directly from DOM")
                            
                            # Print the DOM-extracted contract data
                            print("\n" + "="*80)
                            print("CONTRACT DATA (EXTRACTED DIRECTLY FROM DOM):")
                            print("="*80)
                            
                            # Print header data
                            print("\nCONTRACT HEADER:")
                            if contract_data.get('contractHeader'):
                                for key, value in contract_data['contractHeader'].items():
                                    print(f"{key}: {value}")
                            else:
                                print("No contract header data found")
                                
                            # Print line items
                            print("\nCONTRACT LINES:")
                            if contract_data.get('contractLines'):
                                for i, line in enumerate(contract_data['contractLines']):
                                    print(f"\nLine Item #{i+1}: {line}")
                            else:
                                print("No contract line items found")
                            
                            return json.dumps(contract_data, indent=2)
                        else:
                            error_msg = "Failed to extract contract data after all attempts"
                            print(error_msg)
                            return json.dumps({"error": error_msg, "status": "failed"})
                    except Exception as e:
                        error_msg = f"Failed to extract data directly: {str(e)}"
                        print(error_msg)
                        return json.dumps({"error": error_msg, "status": "failed"})
                    
            except Exception as e:
                error_msg = f"An error occurred during contract retrieval: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                return json.dumps({"error": error_msg, "status": "failed"})
            
            finally:
                # Close browser
                try:
                    await context.close()
                    await browser.close()
                    print("Browser closed.")
                except Exception as e:
                    print(f"Error closing browser: {str(e)}")
    
    except Exception as e:
        error_msg = f"Failed to initialize contract retrieval process: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return json.dumps({"error": error_msg, "status": "failed"})
    
    # Default return if we somehow get here
    return json.dumps({"error": "Unknown error occurred", "status": "failed"})
