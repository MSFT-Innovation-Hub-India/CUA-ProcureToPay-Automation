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
            
            # Navigate to starting page with retry logic
            try:
                print(f"Navigating to contract page for ID: {contract_id}")
                await page.goto(f"https://p2p-erp-web.gentleflower-4e1ad251.swedencentral.azurecontainerapps.io/ContractHeaders/ContractLines/{contract_id}", 
                               wait_until="domcontentloaded",
                               timeout=30000)  # Increased timeout to 30 seconds
                
                print("Successfully navigated to contract page")
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
                    print("\n" + "="*50)
                    user_input = "Get the contract header data and contract lines data from the current page you are in and return that as a JSON object."
                    
                    # Take initial screenshot
                    try:
                        screenshot_base64 = await take_screenshot(page)
                        print("Successfully captured initial screenshot")
                    except Exception as e:
                        error_msg = f"Failed to capture screenshot: {str(e)}"
                        print(error_msg)
                        retry_count += 1
                        continue
                    
                    l_instructions = """
                    You are an AI agent with the ability to control a browser. You can control the keyboard and mouse. You take a screenshot after each action to check if your action was successful.
                    Once you have completed the requested task you should stop running and pass back control to your human supervisor.
                    """
                    
                    # Initial request to the model
                    try:
                        response = client.responses.create(
                            model=MODEL,
                            tools=[{
                                "type": "computer_use_preview",
                                "display_width": DISPLAY_WIDTH,
                                "display_height": DISPLAY_HEIGHT,
                                "environment": "browser"
                            }],
                            instructions = l_instructions,
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
                        print("Successfully sent model initial screenshot and instructions")
                    except Exception as e:
                        error_msg = f"Failed to create model response: {str(e)}"
                        print(error_msg)
                        retry_count += 1
                        continue

                    # Process model actions
                    try:
                        response_string = await process_model_response(client, response, page)
                        print(f"CONTRACT DATA: {response_string[:200]}...") # Print first 200 chars to avoid console clutter
                        
                        if response_string is not None:
                            # Try to validate JSON if it's supposed to be JSON
                            if response_string.strip().startswith("{"):
                                try:
                                    json.loads(response_string)
                                    print("Successfully validated JSON response")
                                except json.JSONDecodeError:
                                    print("Warning: Response looks like JSON but is not valid. Returning as string.")
                            
                            print("Successfully retrieved contract details")
                            waiting_for_data = False
                            return response_string
                        else:
                            print("No response received from model, retrying...")
                            retry_count += 1
                    except Exception as e:
                        error_msg = f"Error processing model response: {str(e)}"
                        print(error_msg)
                        retry_count += 1
                
                if retry_count >= max_retries:
                    error_msg = f"Failed to retrieve contract data after {max_retries} attempts"
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
