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
            
            # Add a small delay after clicking to ensure the element is ready
            await asyncio.sleep(0.5)
        
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
        
        # Enhanced typing that ensures text is entered correctly
        await page.keyboard.type(text, delay=20)  # Increased delay between keystrokes
        
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
    for iteration in range(max_iterations):
        if not hasattr(response, 'output') or not response.output:
            print("No output from model.")
            break
        
        # Safely access response id
        response_id = getattr(response, 'id', 'unknown')
        print(f"\nIteration {iteration + 1} - Response ID: {response_id}\n")
        
        # Print text responses and reasoning
        for item in response.output:
            # Handle text output
            if hasattr(item, 'type') and item.type == "text":
                print(f"\nModel message: {item.text}\n")
                
            # Handle reasoning output
            if hasattr(item, 'type') and item.type == "reasoning":
                # Extract meaningful content from the reasoning
                meaningful_content = []
                
                if hasattr(item, 'summary') and item.summary:
                    for summary in item.summary:
                        # Handle different potential formats of summary content
                        if isinstance(summary, str) and summary.strip():
                            meaningful_content.append(summary)
                        elif hasattr(summary, 'text') and summary.text.strip():
                            meaningful_content.append(summary.text)
                
                # Only print reasoning section if there's actual content
                if meaningful_content:
                    print("=== Model Reasoning ===")
                    for idx, content in enumerate(meaningful_content, 1):
                        print(f"{content}")
                    print("=====================\n")
        
        # Extract computer calls
        computer_calls = [item for item in response.output 
                         if hasattr(item, 'type') and item.type == "computer_call"]
        
        if not computer_calls:
            print("No computer call found in response. Reverting control to human operator")
            break
        
        computer_call = computer_calls[0]
        if not hasattr(computer_call, 'call_id') or not hasattr(computer_call, 'action'):
            print("Computer call is missing required attributes.")
            break
        
        call_id = computer_call.call_id
        action = computer_call.action
        
        # Handle safety checks
        acknowledged_checks = []
        if hasattr(computer_call, 'pending_safety_checks') and computer_call.pending_safety_checks:
            pending_checks = computer_call.pending_safety_checks
            print("\nSafety checks required:")
            for check in pending_checks:
                print(f"- {check.code}: {check.message}")
            
            if input("\nDo you want to proceed? (y/n): ").lower() != 'y':
                print("Operation cancelled by user.")
                break
            
            acknowledged_checks = pending_checks
        
        # Execute the action
        try:
           await page.bring_to_front()
           await handle_action(page, action)
           
           # Check if a new page was created after the action
           if action.type in ["click"]:
               await asyncio.sleep(1.5)
               # Get all pages in the context
               all_pages = page.context.pages
               # If we have multiple pages, check if there's a newer one
               if len(all_pages) > 1:
                   newest_page = all_pages[-1]  # Last page is usually the newest
                   if newest_page != page and newest_page.url not in ["about:blank", ""]:
                       print(f"\tSwitching to new tab: {newest_page.url}")
                       page = newest_page  # Update our page reference
           elif action.type != "wait":
               await asyncio.sleep(0.5)
               
        except Exception as e:
           print(f"Error handling action {action.type}: {e}")
           import traceback
           traceback.print_exc()    

        # Take a screenshot after the action
        screenshot_base64 = await take_screenshot(page)

        print("\tNew screenshot taken")
        
        # Prepare input for the next request
        input_content = [{
            "type": "computer_call_output",
            "call_id": call_id,
            "output": {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{screenshot_base64}"
            }
        }]
        
        # Add acknowledged safety checks if any
        if acknowledged_checks:
            acknowledged_checks_dicts = []
            for check in acknowledged_checks:
                acknowledged_checks_dicts.append({
                    "id": check.id,
                    "code": check.code,
                    "message": check.message
                })
            input_content[0]["acknowledged_safety_checks"] = acknowledged_checks_dicts
        
        # Add current URL for context
        try:
            current_url = page.url
            if current_url and current_url != "about:blank":
                input_content[0]["current_url"] = current_url
                print(f"\tCurrent URL: {current_url}")
        except Exception as e:
            print(f"Error getting URL: {e}")
        
        # Send the screenshot back for the next step
        try:
            response = client.responses.create(
                model=MODEL,
                previous_response_id=response_id,
                tools=[{
                    "type": "computer_use_preview",
                    "display_width": DISPLAY_WIDTH,
                    "display_height": DISPLAY_HEIGHT,
                    "environment": "browser"
                }],
                input=input_content,
                truncation="auto"
            )

            print("\tModel processing screenshot")
        except Exception as e:
            print(f"Error in API call: {e}")
            import traceback
            traceback.print_exc()
            break
    
    if iteration >= max_iterations - 1:
        print("Reached maximum number of iterations. Stopping.")


async def post_invoice(purchase_invoice_no:str, contract_reference: str, supplier_id:str, total_invoice_value: int, invoice_date: str, status:str, remarks:str) -> str:  
    """
    Post a purchase invoice with the specified details to the P2P ERP system.
    
    Args:
        purchase_invoice_no (str): The purchase invoice number
        contract_reference (str): Reference number of the contract
        supplier_id (str): ID of the supplier
        total_invoice_value (int): Total value of the invoice
        invoice_date (str): Date of the invoice
        status (str): Status of approval/rejection of the purchase invoice
        remarks (str): Remarks provided by the invoice approver, especially when invoice is rejected
        
    Returns:
        str: Response string indicating whether the posting was successful or not
    """
    response_string = None
    last_successful_screenshot = None  # Initialize the global variable
    
    # Initialize OpenAI client
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version=API_VERSION
    )
    
    
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
        
        # Navigate to starting page
        await page.goto("https://p2p-erp-web.gentleflower-4e1ad251.swedencentral.azurecontainerapps.io/PurchaseInvoiceHeaders/Create", wait_until="domcontentloaded")
        print("Browser initialized to Purchae invoice Creation page")
        
        # Main interaction loop
        try:
            while True:
                print("\n" + "="*50)
                user_input = input("Enter a task to perform (or 'exit' to quit): ")
                
                if user_input.lower() in ('exit', 'quit'):
                    break
                
                if not user_input.strip():
                    continue
                
                # Take initial screenshot
                screenshot_base64 = await take_screenshot(page)
                print("\nTake initial screenshot")
                
                # Initial request to the model
                response = client.responses.create(
                    model=MODEL,
                    tools=[{
                        "type": "computer_use_preview",
                        "display_width": DISPLAY_WIDTH,
                        "display_height": DISPLAY_HEIGHT,
                        "environment": "browser"
                    }],
                    instructions = "You are an AI agent with the ability to control a browser. You can control the keyboard and mouse. You take a screenshot after each action to check if your action was successful. Once you have completed the requested task you should stop running and pass back control to your human operator.",
                    input=[{
                        "role": "user",
                        "content": [{
                            "type": "input_text",
                            "text": user_input
                        }, {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{screenshot_base64}"
                        }]
                    }],
                    reasoning={"generate_summary": "concise"},
                    truncation="auto"
                )
                print("\nSending model initial screenshot and instructions")

                # Process model actions
                await process_model_response(client, response, page)
                
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Close browser
            await context.close()
            await browser.close()
            print("Browser closed.")
