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
        await page.goto(f"https://p2p-erp-web.gentleflower-4e1ad251.swedencentral.azurecontainerapps.io/ContractHeaders/ContractLines/{contract_id}", wait_until="domcontentloaded")
        print("Browser initialized to Bing.com")
        
        # Main interaction loop
        waiting_for_data = True
        try:
            while waiting_for_data:
                print("\n" + "="*50)
                # user_input = input("Enter a task to perform (or 'exit' to quit): ")
                user_input = "Get the contract header data and contract lines data from the current page you are in and return that as a JSON object."
                
                if user_input.lower() in ('exit', 'quit'):
                    break
                
                if not user_input.strip():
                    continue
                
                # Take initial screenshot
                screenshot_base64 = await take_screenshot(page)
                print("\nTake initial screenshot")
                
                
                l_instructions = """
                You are an AI agent with the ability to control a browser. You can control the keyboard and mouse. You take a screenshot after each action to check if your action was successful.
                Once you have completed the requested task you should stop running and pass back control to your human supervisor.
                """
                
                # Initial request to the model
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
                print("\nSending model initial screenshot and instructions")

                # Process model actions
                response_string=await process_model_response(client, response, page)
                print("CONTRACT DATA:", response_string)
                if response_string is not None:
                    print(f"Returning the contract details back to the caller \n{response_string}")
                    waiting_for_data = False
                    return response_string
                
        except Exception as e:
            print(f"An error occurred: {e}")
            response_string = f"An error occurred: {e}"
            import traceback
            traceback.print_exc()
        
        finally:
            # Close browser
            await context.close()
            await browser.close()
            print("Browser closed.")
        return response_string
    
# if __name__ == "__main__":
#     asyncio.run(main())
