import os
import json
import re
from bs4 import BeautifulSoup
import modal
import typer
import asyncio
import requests

app = modal.App(
    name="product-scraper-app",
    image=modal.Image.debian_slim().pip_install(
        "requests",
        "beautifulsoup4",
        "google-generativeai",
        "typer",
        "google-api-python-client",
        "lxml",
        "playwright",
    ).run_commands("playwright install --with-deps"),
)

@app.function(secrets=[modal.Secret.from_name("my-google-search-secret")])
def search_for_vendor_urls(product_name: str, num_results_per_query: int = 10) -> list[str]:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    try:
        api_key = os.environ["GOOGLE_API_KEY"]
        search_engine_id = os.environ["SEARCH_ENGINE_ID"]
    except KeyError:
        return []

    # Expanded search queries for international and multilingual results
    search_templates = {
        'en': [
            'buy "{product}"', '"{product}" supplier', '"{product}" manufacturer price',
            '"{product}" vendor', '"{product}" for sale'
        ],
        'de': [ # German
            '"{product}" kaufen online', '"{product}" lieferant'
        ]
    }

    # Updated country codes to target as per your request
    target_countries = ['uk', 'us', 'de']

    all_urls = set()
    try:
        service = build("customsearch", "v1", developerKey=api_key)
    except Exception:
        return []

    # Iterate through each language and its corresponding query templates
    for lang, templates in search_templates.items():
        for template in templates:
            query = template.format(product=product_name)
            # Run a general search and then targeted searches for the specified countries
            for country in [None] + target_countries:
                try:
                    search_params = {
                        'q': query,
                        'cx': search_engine_id,
                        'num': num_results_per_query,
                        'lr': f'lang_{lang}'
                    }
                    if country:
                        search_params['cr'] = f'country{country.upper()}'
                        search_params['gl'] = country

                    result = service.cse().list(**search_params).execute()

                    if "items" in result:
                        for item in result["items"]:
                            all_urls.add(item["link"])
                except (HttpError, Exception):
                    continue # Continue to next country/query if one fails

    return list(all_urls)

async def get_webpage_text_with_browser(url: str) -> str | None:
    from playwright.async_api import async_playwright

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, timeout=30000)
            await page.wait_for_load_state('networkidle')
            content = await page.content()
            await browser.close()

        soup = BeautifulSoup(content, "lxml")
        for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
            element.decompose()
        text = soup.get_text(separator='\n', strip=True)
        return text
    except Exception:
        return None

@app.function(
    secrets=[modal.Secret.from_name("my-secret-key-3")],
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=5.0),
    timeout=180,
    max_containers=20
)
async def get_vendor_info_from_url(url: str, product_name: str) -> dict | None:
    import google.generativeai as genai

    webpage_text = await get_webpage_text_with_browser(url)

    if not webpage_text:
        return None

    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
    except Exception:
        return None

    prompt = f"""
    You are a meticulous data extraction bot for a science procurement project.
    Your task is to analyze the text from a webpage and determine if it's a vendor selling the specific product: '{product_name}'.

    From the provided webpage text, extract the following information:
    1.  **vendor_name**: The name of the company/vendor selling the product. Be as precise as possible.
    2.  **product_page_url**: The direct URL to the product page for '{product_name}'. This must be the most specific page possible for the product.

    **Definition of a Product Page:**
    A "product page" is a URL where the specific product ('{product_name}') can be directly purchased, has a detailed specification table, or has a form to "request a quote" for that specific item.
    It must have some sort of buy/purchase button so that we know it is a product page.
    
    **Pages to AVOID:**
    - General homepages (e.g., vendor.com)
    - "About Us" or "Contact" pages.
    - Broad category pages that list many different types of products.
    - Blog posts or news articles about the product.

    The original URL provided was {url}. Use this URL only if it meets the definition of a product page. If you find a more specific link within the page text, use that instead.

    --- WEB PAGE TEXT (first 30,000 characters) ---
    {webpage_text[:30000]}
    --- END OF TEXT ---

    CRITICAL INSTRUCTIONS:
    - Respond ONLY with a single, valid JSON object.
    - The JSON object must have exactly two keys: "vendor_name" and "product_page_url".
    - If the page is not a valid product page as defined above, or if you cannot find the information, respond with a JSON object where both values are null.
    - Do not return any homepages.
    - If the vendor name is unicode, or its not in plain english and legible (for example \u745c\u6b23\u5e73\u745e), use the URL to get the vendor name nad fix it before outputting.
    """

    try:
        response = await model.generate_content_async(prompt)
        
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not match:
            return None

        cleaned_response = match.group(0)
        result = json.loads(cleaned_response)

        if result.get("vendor_name") and result.get("product_page_url"):
            # Normalize vendor name for better de-duplication
            result["vendor_name"] = result["vendor_name"].strip().lower()
            return result
        else:
            return None
            
    except (json.JSONDecodeError, Exception):
        return None

def deduplicate_and_prioritize_vendors(vendors: list[dict]) -> list[dict]:
    """Groups vendors by name and selects the one with the longest (most specific) URL."""
    unique_vendors = {}
    for vendor in vendors:
        name = vendor.get("vendor_name")
        if not name:
            continue

        if name not in unique_vendors:
            unique_vendors[name] = vendor
        else:
            # If the new URL is longer, it's likely more specific (a product page vs a homepage)
            current_url = unique_vendors[name].get("product_page_url", "")
            new_url = vendor.get("product_page_url", "")
            if len(new_url) > len(current_url):
                unique_vendors[name] = vendor
                
    return list(unique_vendors.values())

@app.local_entrypoint()
def main(product: str = typer.Option(..., help="The product to search for, e.g., 'silicon wafers'.")):
    potential_urls = search_for_vendor_urls.remote(product)
    if not potential_urls:
        final_output = {"total_vendors": 0, "vendors": []}
        output_filename = f"vendors_{product.replace(' ', '_')}.json"
        with open(output_filename, 'w') as f:
            json.dump(final_output, f, indent=2)
        return

    vendor_results = list(get_vendor_info_from_url.map(potential_urls, kwargs={"product_name": product}))

    valid_vendors = [res for res in vendor_results if res]
    
    # De-duplicate the results before final output
    final_vendors = deduplicate_and_prioritize_vendors(valid_vendors)
    
    final_output = {
        "total_vendors": len(final_vendors),
        "vendors": final_vendors
    }
    
    output_filename = f"vendors_{product.replace(' ', '_')}.json"
    with open(output_filename, 'w') as f:
        json.dump(final_output, f, indent=2)
