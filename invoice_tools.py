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
    response_string=""
    json_content = None
    
    for iteration in range(max_iterations):
        if not hasattr(response, 'output') or not response.output:
            print("No output from model.")
            break
        
        # Safely access response id
        response_id = getattr(response, 'id', 'unknown')
        print(f"\nIteration {iteration + 1} - Response ID: {response_id}\n")
        print("response output:", response.output)
        
        # Print text responses and reasoning
        for item in response.output:
            # Handle output message
            if hasattr(item, 'type') and item.type == "message":
                text_content = item.content[0].text
                print(f"\nModel message: {text_content}\n")
                response_string += text_content + "\n"
                
                # Check if the response contains JSON
                if "```json" in text_content:
                    # Extract the JSON content between ```json and ```
                    json_start = text_content.find("```json") + 7
                    json_end = text_content.find("```", json_start)
                    if json_end > json_start:
                        json_content = text_content[json_start:json_end].strip()
                        print(f"JSON content extracted: {json_content}")
                        return json_content
    
    if json_content:
        return json_content
    
    if response_string:
        return response_string
    
    if iteration >= max_iterations - 1:
        print("Reached maximum number of iterations. Stopping.")
        return response_string

async def post_invoice(purchase_invoice:str, status:str, remarks:str) -> str:  
    """
    Post the purchase invoice header, lines based on the purchase invoice header and lines data, and the status of the purchase invoice and remarks on the approval or rejection of the purchase invoice.
    :param purchase_invoice (str): The purchase invoice header and lines data in markdown or JSON format.
    :param status (str): status of approval/rejection of the Purchase Invoice.
    :param remarks (str): the remarks provided by the Invoice approver, when the Invoice status is rejected.
    :return: a string indicating whether the posting was successful or not, with details in case of failure.
    :rtype: str
    """
    response_string = None
    
    # Input validation
    if not purchase_invoice:
        error_msg = "Missing purchase invoice data"
        print(error_msg)
        return json.dumps({"error": error_msg, "status": "failed"})
        
    if not status or status.lower() not in ["approved", "rejected"]:
        error_msg = f"Invalid status value: {status}. Must be 'Approved' or 'Rejected'."
        print(error_msg)
        return json.dumps({"error": error_msg, "status": "failed"})
        
    if status.lower() == "rejected" and not remarks:
        error_msg = "Remarks are required when invoice status is 'Rejected'"
        print(error_msg)
        return json.dumps({"error": error_msg, "status": "failed"})
    
    # Initialize OpenAI client
    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            azure_ad_token_provider=token_provider,
            api_version=API_VERSION
        )
        
        print(f"Starting invoice posting process with status: {status}")
        
        # Initialize Playwright with proper error handling
        async with async_playwright() as playwright:
            try:
                browser = await playwright.chromium.launch(
                    headless=False,
                    args=[f"--window-size={DISPLAY_WIDTH},{DISPLAY_HEIGHT}", "--disable-extensions"]
                )
                
                context = await browser.new_context(
                    viewport={"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT},
                    accept_downloads=True
                )
                
                page = await context.new_page()
                
                # Navigate to invoice creation page
                try:
                    print("Navigating to invoice creation page")
                    await page.goto(
                        "https://p2p-erp-web.gentleflower-4e1ad251.swedencentral.azurecontainerapps.io/PurchaseInvoiceHeaders/Create", 
                        wait_until="domcontentloaded",
                        timeout=30000  # Increased timeout to 30 seconds
                    )
                    print("Successfully navigated to invoice creation page")
                except Exception as e:
                    error_msg = f"Failed to navigate to invoice creation page: {str(e)}"
                    print(error_msg)
                    await context.close()
                    await browser.close()
                    return json.dumps({"error": error_msg, "status": "failed"})
                
                # Prepare user input with the invoice data
                l_user_input = f"""
                On the current page you are in, you need to perform the following tasks:
                
                Here is the invoice data to enter:
                {purchase_invoice}
                
                The invoice status is: {status}
                
                {f"Rejection remarks: {remarks}" if status.lower() == "rejected" else ""}
                
                Step 1) Create the Purchase Invoice Header data.
                1. Enter the purchase invoice header data (PurchaseInvoiceNo, ContractReference, SupplierId, TotalInvoiceValue, InvoiceDate) based on the input provided above.
                2. Set the Status field value to '{status}'.
                3. {f"Enter the remarks in the Remarks field: {remarks}" if status.lower() == "rejected" else "Leave the Remarks field empty."}
                4. Save the purchase invoice header data.
                5. Acknowledge the success message displayed on the screen.
                
                Step 2) Create the Purchase Invoice Lines data.
                1. Click on the 'Add Line' button to add a new line item.
                2. Enter the purchase invoice lines data (Description, Quantity, Unit Price, Total Price) based on the input provided above.
                3. Click on the 'Save' button to save the purchase invoice lines data.
                4. Acknowledge the success message displayed on the screen.
                5. Repeat the steps 1 to 4 for all the purchase invoice lines data provided above.
                
                Step 3) If the Purchase Invoice Header and Lines data is successfully posted, then return 'True', else return 'False'.
                """
                
                # Main interaction loop with retry mechanism
                waiting_for_data = True
                retry_count = 0
                max_retries = 3
                
                try:
                    while waiting_for_data and retry_count < max_retries:
                        print(f"\n{'='*50}\nAttempt {retry_count + 1} of {max_retries}")
                        
                        # Take initial screenshot
                        try:
                            screenshot_base64 = await take_screenshot(page)
                            print("Successfully captured initial screenshot")
                        except Exception as e:
                            error_msg = f"Failed to capture screenshot: {str(e)}"
                            print(error_msg)
                            retry_count += 1
                            await asyncio.sleep(1)  # Wait before retry
                            continue
                        
                        l_instructions = """
                        You are an AI agent with the ability to control a browser. You can control the keyboard and mouse. You take a screenshot after each action to check if your action was successful.
                        Once you have completed the requested task you should stop running and pass back control to your human supervisor.
                        """
                        
                        # Make API call to model
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
                                        "text": l_user_input
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
                            await asyncio.sleep(2)  # Wait before retry
                            continue
                        
                        # Process model response
                        try:
                            response_string = await process_model_response(client, response, page)
                            print(f"Response from posting invoice: {response_string}")
                            
                            if response_string is not None:
                                # Check for success indicators in the response
                                if "true" in response_string.lower() or "success" in response_string.lower():
                                    result = {
                                        "status": "success",
                                        "message": "Purchase invoice successfully posted",
                                        "details": response_string
                                    }
                                    print("Invoice posting completed successfully")
                                else:
                                    result = {
                                        "status": "partial",
                                        "message": "Purchase invoice may not have been fully posted",
                                        "details": response_string
                                    }
                                    print("Invoice posting may not have completed successfully")
                                
                                waiting_for_data = False
                                return json.dumps(result)
                            else:
                                print("No response received from model, retrying...")
                                retry_count += 1
                                await asyncio.sleep(2)  # Wait before retry
                        except Exception as e:
                            error_msg = f"Error processing model response: {str(e)}"
                            print(error_msg)
                            retry_count += 1
                            await asyncio.sleep(2)  # Wait before retry
                    
                    if retry_count >= max_retries:
                        error_msg = f"Failed to post invoice data after {max_retries} attempts"
                        print(error_msg)
                        return json.dumps({"error": error_msg, "status": "failed"})
                
                except Exception as e:
                    error_msg = f"An error occurred during invoice posting: {str(e)}"
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
                error_msg = f"Failed to initialize browser for invoice posting: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                return json.dumps({"error": error_msg, "status": "failed"})
    
    except Exception as e:
        error_msg = f"Failed to initialize invoice posting process: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return json.dumps({"error": error_msg, "status": "failed"})
    
    # Default return if we somehow get here
    return json.dumps({"error": "Unknown error occurred", "status": "failed"})
