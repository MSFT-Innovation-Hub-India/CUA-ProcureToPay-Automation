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
                for action in item.actions:
                    print(f"Executing action: {action.type}")
                    try:
                        await handle_action(page, action)
                        # Take a screenshot after each action to verify it worked
                        await take_screenshot(page)
                    except Exception as e:
                        print(f"Error executing action {action.type}: {str(e)}")
        
        # If the model provided no actions, we should stop the loop
        if not has_actions:
            print("No more actions to execute, exiting process_model_response loop")
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
    last_successful_screenshot = None  # Initialize the global variable
    
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
    
    # Format the purchase invoice data to make it more easily parsable by the CUA model
    try:
        # Print the original invoice data for debugging
        print("\n" + "="*80)
        print("ORIGINAL INVOICE DATA:")
        print("="*80)
        print(purchase_invoice[:500] + "..." if len(purchase_invoice) > 500 else purchase_invoice)
        
        # Extract key fields for form filling
        invoice_fields = {}
        line_items = []
        
        # Try to parse JSON if it looks like JSON
        if purchase_invoice.strip().startswith("{") or purchase_invoice.strip().startswith("["):
            try:
                invoice_data = json.loads(purchase_invoice)
                # Extract header fields if this is a structured JSON
                if isinstance(invoice_data, dict):
                    # Look for common field names
                    for field in ["PurchaseInvoiceNo", "InvoiceNumber", "ContractReference", "ContractId", 
                                  "SupplierId", "SupplierID", "Supplier", "TotalInvoiceValue", "Total", 
                                  "InvoiceDate", "Date"]:
                        if field in invoice_data:
                            invoice_fields[field] = invoice_data[field]
                    
                    # Look for line items array
                    for field in ["Lines", "LineItems", "Items", "InvoiceLines"]:
                        if field in invoice_data and isinstance(invoice_data[field], list):
                            line_items = invoice_data[field]
                            break
                
                # Format as a clear, structured string for the CUA
                formatted_invoice = "INVOICE HEADER:\n"
                for key, value in invoice_fields.items():
                    formatted_invoice += f"{key}: {value}\n"
                
                formatted_invoice += "\nINVOICE LINE ITEMS:\n"
                for i, line in enumerate(line_items):
                    formatted_invoice += f"\nItem {i+1}:\n"
                    if isinstance(line, dict):
                        for key, value in line.items():
                            formatted_invoice += f"  {key}: {value}\n"
                    else:
                        formatted_invoice += f"  {line}\n"
                
                print("\nFormatted invoice data for CUA:")
                print(formatted_invoice)
                
            except json.JSONDecodeError:
                print("Invoice data looks like JSON but couldn't be parsed. Using as-is.")
                formatted_invoice = purchase_invoice
        else:
            # For non-JSON formats (like markdown tables), keep as is
            formatted_invoice = purchase_invoice
        
        print(f"Prepared invoice data for form filling")
    except Exception as e:
        print(f"Error formatting invoice data: {str(e)}")
        # Continue with original data if formatting fails
        formatted_invoice = purchase_invoice
    
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
                
                # Enable detailed console logging for page interactions
                page.on("console", lambda msg: print(f"BROWSER CONSOLE: {msg.text}"))
                
                # Navigate to invoice creation page with enhanced waiting
                try:
                    print("Navigating to invoice creation page")
                    await page.goto(
                        "https://p2p-erp-web.gentleflower-4e1ad251.swedencentral.azurecontainerapps.io/PurchaseInvoiceHeaders/Create", 
                        wait_until="networkidle",  # Wait until network is idle for better form loading
                        timeout=30000  # Increased timeout to 30 seconds
                    )
                    
                    # Wait for form fields to be present and visible
                    await page.wait_for_selector("form", timeout=10000)
                    print("Form detected on page")
                    
                    # Additional wait to ensure form is fully interactive
                    await asyncio.sleep(2)
                    
                    print("Successfully navigated to invoice creation page and verified form presence")
                except Exception as e:
                    error_msg = f"Failed to navigate to invoice creation page: {str(e)}"
                    print(error_msg)
                    await context.close()
                    await browser.close()
                    return json.dumps({"error": error_msg, "status": "failed"})
                
                # Prepare user input with form-specific instructions based on OpenAI CUA documentation
                l_user_input = f"""
                I need you to fill out a purchase invoice form. Follow these specific steps in order:

                FORM DATA:
                {formatted_invoice}

                Status: {status}
                {f'Remarks: {remarks}' if status.lower() == 'rejected' else ''}

                INSTRUCTIONS (EXECUTE THESE STEPS IN ORDER):

                1. HEADER FORM FILLING:
                   - For each form field, first CLICK on the field explicitly
                   - Then TYPE the corresponding value
                   - Verify the field contains the correct value before moving to the next field
                   - Fill these fields in order:
                     a) PurchaseInvoiceNo (look for this field in the form)
                     b) ContractReference (look for this field in the form)
                     c) SupplierId (look for this field in the form)
                     d) TotalInvoiceValue (look for this field in the form)
                     e) InvoiceDate (look for this field in the form, use MM/DD/YYYY format)
                     f) Status (select "{status}" from dropdown if available)
                     g) {f'Remarks: Type "{remarks}"' if status.lower() == 'rejected' else 'Skip Remarks field'}

                2. SAVE THE HEADER:
                   - Look for a "Create", "Save", or "Submit" button
                   - Click it to save the header data
                   - Wait for the page to refresh or update after saving
                   - Verify the header was saved successfully

                3. ADD LINE ITEMS:
                   - For each line item in the invoice data:
                     a) Find and click "Add Line" or similar button
                     b) Wait for the line item form to appear
                     c) Fill in each field by explicitly clicking then typing:
                        - Description (from line item data)
                        - Quantity (from line item data)
                        - Unit Price (from line item data)
                        - Total Price (from line item data)
                     d) Click "Save" for this line item
                     e) Wait for confirmation
                     f) Repeat for all line items

                4. VERIFICATION:
                   - After all data is entered, verify it appears correctly on the page
                   - Return "True" if the process completed successfully

                IMPORTANT: Take your time with each step. Click explicitly on each field before typing. Wait for page updates between actions.
                """
                
                # Main interaction loop with retry mechanism
                waiting_for_data = True
                retry_count = 0
                max_retries = 3
                
                try:
                    while waiting_for_data and retry_count < max_retries:
                        print(f"\n{'='*50}\nAttempt {retry_count + 1} of {max_retries}")
                        
                        # Take initial screenshot with verification
                        try:
                            # First verify that form fields are visible on the page
                            form_fields_visible = await page.evaluate("""() => {
                                const formElements = document.querySelectorAll('input, select, textarea');
                                return formElements.length > 0;
                            }""")
                            
                            if not form_fields_visible:
                                print("Warning: Form fields not visible on page. Waiting...")
                                await asyncio.sleep(3)
                            
                            # Take screenshot
                            screenshot_base64 = await take_screenshot(page)
                            print("Successfully captured initial screenshot")
                        except Exception as e:
                            error_msg = f"Failed to capture screenshot: {str(e)}"
                            print(error_msg)
                            retry_count += 1
                            await asyncio.sleep(1)
                            continue
                        
                        # Enhanced instructions for CUA model based on OpenAI documentation
                        l_instructions = """
                        You are an AI agent controlling a web browser to fill out forms. Follow these guidelines:
                        
                        1. FORM FILLING BEST PRACTICES:
                           - Always CLICK on a field before typing into it
                           - After clicking, verify the field is focused before typing
                           - Use explicit clicks for buttons and controls
                           - Wait briefly after each action to let the page respond
                           - If a dropdown needs selecting, first click it, then click the correct option
                        
                        2. FORM NAVIGATION:
                           - Use TAB key only as a last resort - prefer explicit clicks
                           - For date fields, use the correct format specified
                           - If a field doesn't accept input, try clicking elsewhere and coming back
                        
                        3. VERIFICATION:
                           - After entering data in a field, verify it was correctly entered
                           - After clicking buttons, wait for page changes
                           - If an action doesn't work, try an alternative approach
                        
                        4. DEBUGGING:
                           - If you encounter issues, describe what you're seeing
                           - If a field is not visible, try scrolling to reveal it
                        
                        Be methodical and precise with each interaction.
                        """
                        
                        # Make API call to model with enhanced instructions
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
                            await asyncio.sleep(2)
                            continue
                        
                        # Process model response and execute browser actions with enhanced debugging
                        try:
                            print("Processing model response and executing actions...")
                            
                            # Debug the model's initial response
                            if hasattr(response, 'output') and response.output:
                                for item in response.output:
                                    if hasattr(item, 'type') and item.type == "computer_use_actions":
                                        print(f"Model returned {len(item.actions)} initial actions")
                                        for idx, action in enumerate(item.actions):
                                            print(f"  Action {idx+1}: {action.type}")
                                            if action.type == "click" and hasattr(action, "x") and hasattr(action, "y"):
                                                print(f"    Click at ({action.x}, {action.y})")
                                            elif action.type == "type" and hasattr(action, "text"):
                                                print(f"    Type: '{action.text}'")
                            
                            # Now process the full response
                            response_string = await process_model_response(client, response, page)
                            print(f"Response from posting invoice (truncated): {response_string[:200]}..." if response_string and len(response_string) > 200 else response_string)
                            
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
                                # Try to inspect form field values for debugging
                                try:
                                    form_values = await page.evaluate("""() => {
                                        const result = {};
                                        document.querySelectorAll('input, select, textarea').forEach(el => {
                                            if (el.id) result[el.id] = el.value;
                                            else if (el.name) result[el.name] = el.value;
                                        });
                                        return result;
                                    }""")
                                    print("Current form field values:")
                                    for field, value in form_values.items():
                                        print(f"  {field}: {value}")
                                except Exception as e:
                                    print(f"Error inspecting form values: {e}")
                                
                                await asyncio.sleep(2)
                        except Exception as e:
                            error_msg = f"Error processing model response: {str(e)}"
                            print(error_msg)
                            retry_count += 1
                            await asyncio.sleep(2)
                    
                    if retry_count >= max_retries:
                        error_msg = f"Failed to post invoice data after {max_retries} attempts"
                        print(error_msg)
                        # Try direct form filling as a last resort
                        try:
                            print("Attempting direct form filling via JavaScript...")
                            result = await page.evaluate("""(invoiceData, status, remarks) => {
                                // Try to parse invoice data
                                let invoice;
                                try {
                                    if (typeof invoiceData === 'string' && 
                                        (invoiceData.trim().startsWith('{') || invoiceData.trim().startsWith('['))) {
                                        invoice = JSON.parse(invoiceData);
                                    }
                                } catch (e) {
                                    console.error('Failed to parse invoice data:', e);
                                }
                                
                                // Simple direct field filling attempt
                                const fillForm = () => {
                                    // Get all input fields
                                    const inputs = document.querySelectorAll('input, select, textarea');
                                    let filled = 0;
                                    
                                    // Try to match common field names
                                    inputs.forEach(input => {
                                        const name = input.name?.toLowerCase() || input.id?.toLowerCase() || '';
                                        
                                        // Try to find matching data from invoice
                                        if (name.includes('invoiceno') || name.includes('purchaseinvoiceno')) {
                                            if (invoice?.PurchaseInvoiceNo) input.value = invoice.PurchaseInvoiceNo;
                                            else if (invoice?.InvoiceNumber) input.value = invoice.InvoiceNumber;
                                            filled++;
                                        }
                                        else if (name.includes('contract')) {
                                            if (invoice?.ContractReference) input.value = invoice.ContractReference;
                                            else if (invoice?.ContractId) input.value = invoice.ContractId;
                                            filled++;
                                        }
                                        else if (name.includes('supplier')) {
                                            if (invoice?.SupplierId) input.value = invoice.SupplierId;
                                            else if (invoice?.Supplier) input.value = invoice.Supplier;
                                            filled++;
                                        }
                                        else if (name.includes('total') || name.includes('value')) {
                                            if (invoice?.TotalInvoiceValue) input.value = invoice.TotalInvoiceValue;
                                            else if (invoice?.Total) input.value = invoice.Total;
                                            filled++;
                                        }
                                        else if (name.includes('date')) {
                                            if (invoice?.InvoiceDate) input.value = invoice.InvoiceDate;
                                            else if (invoice?.Date) input.value = invoice.Date;
                                            filled++;
                                        }
                                        else if (name.includes('status')) {
                                            input.value = status;
                                            filled++;
                                        }
                                        else if (name.includes('remark') || name.includes('comment')) {
                                            input.value = remarks;
                                            filled++;
                                        }
                                        
                                        // Trigger change events to ensure form validation runs
                                        const event = new Event('change', { bubbles: true });
                                        input.dispatchEvent(event);
                                    });
                                    
                                    return filled;
                                };
                                
                                const filledCount = fillForm();
                                return { 
                                    success: filledCount > 0, 
                                    message: `Attempted to fill ${filledCount} fields directly` 
                                };
                            }""", formatted_invoice, status, remarks)
                            
                            print(f"Result of direct form filling: {result}")
                            
                            # Take a final screenshot to see the result
                            final_screenshot = await take_screenshot(page)
                            
                            return json.dumps({
                                "error": error_msg,
                                "status": "attempted_direct_fill",
                                "direct_fill_result": result
                            })
                        except Exception as e:
                            print(f"Error during direct form filling: {e}")
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
