from playwright.async_api import async_playwright, Browser, Page
import asyncio

class LocalPlaywrightComputer:
    """Launches a local Chromium instance using Playwright async API."""

    def __init__(self, headless: bool = False):
        self._playwright = None
        self._browser = None
        self._page = None
        self.headless = headless
        self.environment = "browser"
        self.dimensions = (1024, 768)

    async def __aenter__(self):
        # Start Playwright and get browser/page
        self._playwright = await async_playwright().start()
        await self._get_browser_and_page()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def _get_browser_and_page(self):
        width, height = self.dimensions
        launch_args = [f"--window-size={width},{height}", "--disable-extensions", "--disable-file-system"]
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=launch_args,
            env={"DISPLAY": ":0"}
        )
        
        context = await self._browser.new_context()
        
        # Add event listeners for page creation and closure
        context.on("page", self._handle_new_page)
        
        self._page = await context.new_page()
        await self._page.set_viewport_size({"width": width, "height": height})
        self._page.on("close", self._handle_page_close)

        # Initialize with a blank page instead of hardcoding a specific URL
        await self._page.goto("about:blank")
        
    def _handle_new_page(self, page):
        """Handle the creation of a new page."""
        print("New page created")
        self._page = page
        page.on("close", self._handle_page_close)
        
    def _handle_page_close(self, page):
        """Handle the closure of a page."""
        print("Page closed")
        if self._page == page:
            if self._browser.contexts[0].pages:
                self._page = self._browser.contexts[0].pages[-1]
            else:
                print("Warning: All pages have been closed.")
                self._page = None
    
    async def screenshot(self):
        """Capture screenshot of the current page."""
        return await self._page.screenshot(full_page=False)
    
    async def click(self, selector):
        """Click on an element matching the selector."""
        try:
            # Wait for the element to be visible and enabled before clicking
            await self._page.wait_for_selector(selector, state="visible", timeout=5000)
            await self._page.click(selector)
            return True
        except Exception as e:
            print(f"Click failed for selector {selector}: {e}")
            return False
    
    async def fill(self, selector, text):
        """Fill text into an input field matching the selector."""
        try:
            # Wait for the element to be visible before filling
            await self._page.wait_for_selector(selector, state="visible", timeout=5000)
            # Clear the field first
            await self._page.evaluate(f"document.querySelector('{selector}').value = ''")
            # Then fill it
            await self._page.fill(selector, text)
            return True
        except Exception as e:
            print(f"Fill failed for selector {selector}: {e}")
            return False
    
    async def type_into_focused(self, text):
        """Type text into the currently focused element."""
        try:
            await self._page.keyboard.type(text, delay=50)
            return True
        except Exception as e:
            print(f"Type into focused element failed: {e}")
            return False
            
    async def clear_and_type(self, selector, text):
        """Clear an input field and type text into it."""
        try:
            # Focus the element first
            await self._page.focus(selector)
            # Select all text
            await self._page.keyboard.press("Control+a")
            # Delete the selected text
            await self._page.keyboard.press("Delete")
            # Type the new text with delay between keystrokes
            await self._page.keyboard.type(text, delay=50)
            return True
        except Exception as e:
            print(f"Clear and type failed for selector {selector}: {e}")
            return False
    
    async def focus_and_type(self, x, y, text):
        """Click at coordinates to focus and then type text."""
        try:
            # Click to focus
            await self._page.mouse.click(x, y)
            # Wait a moment for focus to take effect
            await asyncio.sleep(0.3)
            # Select all existing text
            await self._page.keyboard.press("Control+a")
            # Delete the selected text
            await self._page.keyboard.press("Delete")
            # Type new text with delay
            await self._page.keyboard.type(text, delay=50)
            return True
        except Exception as e:
            print(f"Focus and type failed for coordinates ({x}, {y}): {e}")
            return False
    
    async def goto(self, url):
        """Navigate to a URL."""
        await self._page.goto(url)
        
    async def evaluate(self, js_expression):
        """Execute JavaScript in the browser context."""
        return await self._page.evaluate(js_expression)
    
    async def wait_for_load_state(self):
        """Wait for the page to reach a stable load state."""
        try:
            await self._page.wait_for_load_state("networkidle", timeout=10000)
            return True
        except Exception as e:
            print(f"Wait for load state failed: {e}")
            return False