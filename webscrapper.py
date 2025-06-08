import os
import re
import json
import urllib.request
from typing import Dict, Any, List, TypedDict, Optional

import modal
from bs4 import BeautifulSoup
import google.generativeai as genai

# Define a TypedDict for clarity and type hinting for the JSON structure
class VendorProduct(TypedDict):
    vendor_name: str
    product_page_url: str

class VendorData(TypedDict):
    total_vendors: int
    vendors: List[VendorProduct]

# ---------------------------------------------------------------------------
# Helper function to load URLs from a JSON file
# ---------------------------------------------------------------------------
def load_urls_from_json(file_path: str) -> Dict[str, List[str]]:
    """
    Loads vendor URLs from a JSON file and formats them into the
    url_dict structure: {"Vendor Name": ["url1", "url2"], ...}
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data: VendorData = json.load(f)
        
        url_dict: Dict[str, List[str]] = {}
        for vendor_info in data['vendors']:
            vendor_name = vendor_info['vendor_name']
            product_url = vendor_info['product_page_url']
            
            if vendor_name not in url_dict:
                url_dict[vendor_name] = []
            url_dict[vendor_name].append(product_url)
            
        return url_dict
    except FileNotFoundError:
        print(f"Error: JSON file not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Check file format.")
        return {}
    except KeyError as e:
        print(f"Error: JSON file is missing expected key: {e}. Check file structure.")
        return {}


# ---------------------------------------------------------------------------
# Modal setup (installs deps into the container)
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim()
    .pip_install("beautifulsoup4", "google-generativeai")
    .env({"GEMINI_MODEL": "models/gemini-1.5-flash-latest"})
)

app = modal.App(name="product-info-scraper-gemini", image=image)

# ---------------------------------------------------------------------------
# Helper utilities – identical scraping logic
# ---------------------------------------------------------------------------

def _download(url: str) -> str:
    """Return raw HTML for *url* (UTF‑8 decoded)."""
    try:
        with urllib.request.urlopen(url, timeout=15) as resp: 
            return resp.read().decode("utf-8", errors="ignore")
    except urllib.error.URLError as e:
        if isinstance(e, urllib.error.HTTPError):
            print(f"HTTP Error {e.code} for {url}: {e.reason}")
        else:
            print(f"URLError for {url}: {e.reason}")
        raise
    except Exception as e:
        print(f"Generic Error downloading {url}: {e}")
        raise


def _scrape_candidate_fragments(soup: BeautifulSoup) -> List[str]:
    """Collect text fragments likely to contain product data."""
    frags: List[str] = []

    # 1. <title>
    if soup.title and soup.title.string:
        frags.append(soup.title.string.strip())

    # 2. Meta tags (OG, Twitter, product-price, etc.)
    meta_selectors = [
        ("meta", {"property": "og:title"}),
        ("meta", {"property": "og:description"}),
        ("meta", {"property": "product:price:amount"}),
        ("meta", {"name": "twitter:title"}),
        ("meta", {"name": "twitter:description"}),
    ]
    for tag_name, attrs in meta_selectors:
        tag = soup.find(tag_name, attrs=attrs)
        if tag and tag.get("content"):
            frags.append(tag["content"].strip())

    # 3. JSON‑LD product schema
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("@type") in {"Product", "product"}:
                        frags.append(json.dumps(item))
            elif isinstance(data, dict) and data.get("@type") in {"Product", "product"}:
                frags.append(json.dumps(data))
        except (TypeError, ValueError):
            continue

    # 4. Visible headings
    for h in soup.select("h1, h2, h3"):
        if h.get_text(strip=True):
            frags.append(h.get_text(strip=True))

    # 5. Common elements for price and description
    common_selectors = [
        re.compile(r"price|amount", re.I),
        re.compile(r"description|details|content", re.I),
    ]

    for selector in common_selectors:
        for tag in soup.find_all(class_=selector):
            text = tag.get_text(strip=True)
            if text:
                if "price" in selector.pattern.lower() and re.search(r'\d', text):
                    frags.append(f"Price_Candidate: {text}")
                elif "description" in selector.pattern.lower() and len(text) > 50:
                    frags.append(f"Description_Candidate: {text}")
                else:
                    frags.append(text)

    # 6. More specific selectors for Product Specifications (lists, definition lists, tables)
    spec_containers = soup.select(
        ".product-description-container, .product-specifications, .spec-table, .product-details-table, #product-details, .tech-specs, .features"
    )
    for container in spec_containers:
        for li in container.find_all("li"):
            text = li.get_text(strip=True)
            if text and ":" in text and len(text) > 5:
                frags.append(f"Spec_Candidate: {text}")
        
        for dt_dd in container.find_all(["dt", "dd"]):
            text = dt_dd.get_text(strip=True)
            if text and len(text) > 5:
                frags.append(f"Spec_Candidate: {text}")

        for table in container.find_all("table"):
            for row in table.find_all("tr"):
                header_cells = [th.get_text(strip=True) for th in row.find_all("th")]
                data_cells = [td.get_text(strip=True) for td in row.find_all("td")]
                
                if header_cells and data_cells and len(header_cells) == len(data_cells):
                    for i in range(len(header_cells)):
                        frags.append(f"Table_Spec_Candidate: {header_cells[i]}: {data_cells[i]}")
                elif data_cells:
                    frags.append(f"Table_Data_Candidate: {'; '.join(data_cells)}")

    # 7. General paragraph text from common content areas
    for p_tag in soup.select("div.content, article.product-content, section.product-details, main p"):
        text = p_tag.get_text(strip=True)
        if text and len(text) > 100:
            frags.append(f"Paragraph_Content: {text}")

    frags = list(dict.fromkeys(frags))

    return frags

# ---------------------------------------------------------------------------
# Gemini‑powered extraction
# ---------------------------------------------------------------------------
@app.function(secrets=[modal.Secret.from_name("my-api-key-secret2")])
def _llm_extract_product(text: str, *, temperature: float = 0.0, material_context: Optional[str] = None) -> Dict[str, Any]:
    """Use **Gemini** to turn raw *text* into structured product JSON."""
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment. Ensure modal secret 'my-api-key-secret2' is created and attached.")

    genai.configure(api_key=gemini_api_key)

    model_name = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")
    model = genai.GenerativeModel(model_name)

    sys_msg = (
        "You are a world-class extraction engine. Given arbitrary text about a product, "
        "return a JSON object with keys: name, brand (nullable), price (float or null), "
        "currency (ISO‑4217 or null), description (nullable), image_url (nullable), "
        "**specifications (dictionary of key-value pairs or null)**. "
        "Return only JSON—no code fences. "
        "Be as exhaustive as possible for the **description**, "
        "extract accurate numerical **price** and **ISO-4217 currency code** if available. "
        "For **specifications**, focus *only* on parameters that define the product's **technical, physical, or performance characteristics**. "
        "These should be distinct, factual attributes relevant to the specific product described in the text. "
        "Exclude general commercial terms, shipping details, or marketing fluff. "
        "Parse all such relevant specifications into the 'specifications' dictionary, "
        "using a concise specification name (e.g., 'Color', 'Processor', 'Material', 'Capacity') as the key and its value as the string value. "
        "Example for a hypothetical product: {'Color': 'Black', 'Processor': 'XYZ-Chip', 'Storage': '256GB'}. "
    )

    if material_context:
        sys_msg += (
            f"Additionally, provide a 'match_score' (float between 0 and 1) indicating "
            f"how well the product's extracted specifications and overall context "
            f"match the user-defined material context: '{material_context}'. "
            f"A score of 1.0 means a perfect match, 0.0 means no match."
        )

    user_msg = "Extract product information from the following text:\n" + text

    full_prompt = f"{sys_msg}\n\n{user_msg}"

    try:
        response = model.generate_content(
            [
                {"role": "user", "parts": [full_prompt]},
            ],
            generation_config={
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        )

        content = response.text.strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error during Gemini content generation or JSON parsing: {e}")
        return {"llm_raw_error": str(e), "llm_raw_content": response.text.strip() if 'response' in locals() else "No response"}


# ---------------------------------------------------------------------------
# Modal function – callable from your Python code OR via `modal run`
# ---------------------------------------------------------------------------

@app.function(timeout=120)
def get_product_info(url: str, material_context: Optional[str] = None) -> Dict[str, Any]:
    """End‑to‑end pipeline: download → scrape fragments → Gemini → JSON."""
    try:
        html = _download(url)
        soup = BeautifulSoup(html, "html.parser")
        fragments = _scrape_candidate_fragments(soup)
        concatenated = "\n".join(fragments)[:8000]
        if not concatenated.strip():
            return {"error": "No relevant text scraped for LLM.", "source_url": url}

        # Pass material_context to _llm_extract_product
        info = _llm_extract_product.remote(concatenated, material_context=material_context)
        info["source_url"] = url

        # Price validation and flagging
        extracted_price = info.get("price")
        if extracted_price is None or (isinstance(extracted_price, (float, int)) and extracted_price == 0.0):
            info["error_price_missing"] = "No valid price extracted (price is None or 0.0)."
        
        return info
    except Exception as exc:
        # Catch and report network/downloading errors
        print(f"Error in get_product_info for {url}: {exc}")
        # Return a dictionary indicating the error, which main can then filter
        return {"error": str(exc), "source_url": url}

# ---------------------------------------------------------------------------
# Local entrypoint – processes all URLs and outputs to JSON file
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    json_file_path: str = "urls.json",
    output_file_name: str = "product_data.json",
    material_context: str = "",
):
    """
    Invoked by: `modal run your_script_name.py [--json-file-path urls.json] [--output-file-name product_data.json] [--material-context "Sapphire Wafer"]`
    
    Processes URLs from a JSON file, extracts product info, and saves to a JSON output file.
    """
    
    global_material_context = material_context
    if global_material_context:
        print(f"Processing with material context: '{global_material_context}'")

    loaded_url_data = load_urls_from_json(json_file_path)
    if not loaded_url_data:
        print("No URLs loaded. Exiting.")
        return

    final_output_data: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for vendor_name, urls in loaded_url_data.items():
        # Initialize vendor entry if it doesn't exist
        vendor_products: Dict[str, Dict[str, Any]] = {} 

        for url in urls:
            print(f"Fetching {url} for {vendor_name}...")
            # Pass global_material_context to get_product_info.remote()
            extracted_data = get_product_info.remote(url, material_context=global_material_context)

            # --- NEW LOGIC: Filter out HTTP errors and price-missing errors ---
            if "error" in extracted_data and "HTTP Error" in extracted_data["error"]:
                print(f"Skipping {url} for '{vendor_name}' due to HTTP error: {extracted_data['error']}")
                continue # Skip this URL
            
            if "error_price_missing" in extracted_data:
                print(f"Skipping {url} for '{vendor_name}' due to missing or invalid price: {extracted_data['error_price_missing']}")
                continue # Skip this URL
            # --- End NEW LOGIC ---

            # Prepare the product data for output, skipping nulls
            product_entry: Dict[str, Any] = {}
            
            # Standard fields (only add if not None, not empty string, and not empty list/dict)
            for key in ["name", "brand", "price", "currency", "description", "image_url", "source_url", "match_score"]: # Added match_score here
                value = extracted_data.get(key)
                # Check for None, empty string, empty list, or empty dict
                if value is not None and value != "" and (not isinstance(value, (list, dict)) or value):
                    product_entry[key] = value
            
            # Flatten specifications directly into the product entry, only if meaningful
            specifications = extracted_data.get("specifications")
            if isinstance(specifications, dict) and specifications:
                for spec_key, spec_value in specifications.items():
                    if spec_value is not None and spec_value != "":
                        product_entry[spec_key] = spec_value
            elif specifications is not None:
                product_entry["specifications_raw"] = specifications

            # Include LLM-specific errors, but not general HTTP or price errors, as we're filtering those
            if "llm_raw_error" in extracted_data:
                product_entry["llm_raw_error"] = extracted_data["llm_raw_error"]
            if "llm_raw_content" in extracted_data:
                product_entry["llm_raw_content"] = extracted_data["llm_raw_content"]

            # Only add the product entry if it contains any extracted data beyond just source_url and potential LLM errors
            # A product is considered valid if it has at least 'source_url' and one other meaningful field, OR if it has meaningful fields.
            if len(product_entry) > 1 or (len(product_entry) == 1 and "source_url" in product_entry and (product_entry.get("name") or product_entry.get("brand") or product_entry.get("price") or product_entry.get("currency") or product_entry.get("description") or product_entry.get("image_url") or product_entry.get("specifications_raw") or product_entry.get("llm_raw_error") or product_entry.get("match_score"))):
                vendor_products[url] = product_entry
        
        # Only add vendor to final_output_data if it has collected any valid products
        if vendor_products:
            final_output_data[vendor_name] = vendor_products
    
    # Write the collected data to a JSON file
    try:
        with open(output_file_name, 'w', encoding='utf-8') as f:
            json.dump(final_output_data, f, ensure_ascii=False, indent=2)
        print(f"\nSuccessfully extracted data and saved to '{output_file_name}'")
    except Exception as e:
        print(f"\nError saving data to JSON file: {e}")

