import os
import re
import json
import urllib.request
from typing import Dict, Any, List

import modal
from bs4 import BeautifulSoup
import google.generativeai as genai

# ---------------------------------------------------------------------------
# Vendor catalogue – add more URLs freely
# ---------------------------------------------------------------------------
url_dict: Dict[str, List[str]] = {
    "MSE Supplies": [
        "https://www.msesupplies.com/products/mse-pro-4-inch-sapphire-wafer-c-plane-single-or-double-side-polish-al-sub-2-sub-o-sub-3-sub-single-crystal",
    ],
    "CRYSCORE": [
        "https://www.cryscore.com/products/4-inch-c-plane-0001-sapphire-wafers.html",
    ],
    "University Wafer": [
        "https://www.universitywafer.com/4-inch-sapphire-wafers.html"
    ],
    "Sokatec": [
        "https://sokatec.com/products/4-inch-sapphire-substrate"
    ],
    "Ultra Nanotech": [
        "https://ultrananotec.com/product/sapphire-wafer-4-inch/"
    ],
    "Precision Micro-Optics": [
        "https://www.pmoptics.com/sapphires_wafers.html"
    ],
    "Shanghai Famous Trade Co., Ltd.": [
        "https://www.sapphire-substrate.com/buy-sapphire_wafer.html"
    ],
    "Gallium Nitride Wafer": [
        "https://www.galliumnitridewafer.com/sale-53297255-sapphire-wafer-4-dia-76-2mm-0-1mm-thickness-550um-c-plane-99-99-pure.html"
    ],
    "WDQ Optics": [
        "https://www.wdqoptics.com/products/4-inch-c-plane0001-sapphire-wafers"
    ]
}

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
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8", errors="ignore")


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
            if isinstance(data, dict) and data.get("@type") in {"Product", "product"}:
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
        ".product-description-container, .product-specifications, .spec-table, .product-details-table, #product-details, .tech-specs"
    )
    for container in spec_containers:
        # Look for list items
        for li in container.find_all("li"):
            text = li.get_text(strip=True)
            if text and ":" in text and len(text) > 5:
                frags.append(f"Spec_Candidate: {text}")
        
        # Look for definition lists
        for dt_dd in container.find_all(["dt", "dd"]):
            text = dt_dd.get_text(strip=True)
            if text and len(text) > 5:
                frags.append(f"Spec_Candidate: {text}")

        # Look for table data (header + data cells)
        for table in container.find_all("table"):
            for row in table.find_all("tr"):
                header_cells = [th.get_text(strip=True) for th in row.find_all("th")]
                data_cells = [td.get_text(strip=True) for td in row.find_all("td")]
                
                if header_cells and data_cells and len(header_cells) == len(data_cells):
                    for i in range(len(header_cells)):
                        frags.append(f"Table_Spec_Candidate: {header_cells[i]}: {data_cells[i]}")
                elif data_cells:
                    frags.append(f"Table_Data_Candidate: {'; '.join(data_cells)}")


    frags = list(dict.fromkeys(frags))

    return frags

# ---------------------------------------------------------------------------
# Gemini‑powered extraction
# ---------------------------------------------------------------------------
@app.function(secrets=[modal.Secret.from_name("my-api-key-secret")])
def _llm_extract_product(text: str, *, temperature: float = 0.0) -> Dict[str, Any]:
    """Use **Gemini** to turn raw *text* into structured product JSON."""
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment. Ensure modal secret 'my-api-key-secret' is created and attached.")

    genai.configure(api_key=gemini_api_key)

    model_name = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")
    model = genai.GenerativeModel(model_name)

    # UPDATED PROMPT: Product-agnostic, but contextual for specifications
    sys_msg = (
        "You are a world-class extraction engine. Given arbitrary text about a product, "
        "return a JSON object with keys: name, brand (nullable), price (float or null), "
        "currency (ISO‑4217 or null), description (nullable), image_url (nullable), "
        "and **specifications (dictionary of key-value pairs or null)**. "
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
def get_product_info(url: str) -> Dict[str, Any]:
    """End‑to‑end pipeline: download → scrape fragments → Gemini → JSON."""
    try:
        html = _download(url)
        soup = BeautifulSoup(html, "html.parser")
        fragments = _scrape_candidate_fragments(soup)
        concatenated = "\n".join(fragments)[:8000]
        info = _llm_extract_product.remote(concatenated)
        info["source_url"] = url
        return info
    except Exception as exc:
        print(f"Error in get_product_info for {url}: {exc}")
        return {"error": str(exc), "source_url": url}

# ---------------------------------------------------------------------------
# Local entrypoint – processes all URLs and formats as a simpler table
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Invoked by: `modal run get_started.py` (no --url needed now)"""
    all_results: List[Dict[str, Any]] = []

    for vendor, urls in url_dict.items():
        for url in urls:
            print(f"Fetching {url} for {vendor}...")
            data = get_product_info.remote(url)
            all_results.append(data)

    print("\n--- All Product Information ---")
    
    headers = ["Vendor", "Name", "Brand", "Price", "Currency", "Image URL", "Source URL", "Error"]
    
    # Define column widths for simple alignment (adjust as needed)
    col_widths = {
        "Vendor": 15,
        "Name": 30,
        "Brand": 15,
        "Price": 10,
        "Currency": 10,
        "Image URL": 20,
        "Source URL": 40,
        "Error": 20,
    }

    # Print header
    header_line = "| " + " | ".join([h.ljust(col_widths.get(h, 10)) for h in headers]) + " |"
    print(header_line)
    print("-" * len(header_line))

    # Print data rows
    for item in all_results:
        vendor_name = "N/A"
        for v, u_list in url_dict.items():
            if item.get("source_url") in u_list:
                vendor_name = v
                break

        row_values = [
            vendor_name,
            item.get("name", "N/A"),
            item.get("brand", "N/A"),
            item.get("price", "N/A"),
            item.get("currency", "N/A"),
            item.get("image_url", "N/A"),
            item.get("source_url", "N/A"),
            item.get("error", "N/A"),
        ]
        
        formatted_row = []
        for i, val in enumerate(row_values):
            header_name = headers[i]
            s_val = str(val)
            max_width = col_widths.get(header_name, 10)

            if header_name == "Description" or header_name == "Specifications": 
                display_val = "<See Full Details Below>" 
            elif len(s_val) > max_width:
                display_val = s_val[:max_width - 3] + "..."
            else:
                display_val = s_val
            
            formatted_row.append(display_val.ljust(max_width))

        print("| " + " | ".join(formatted_row) + " |")

    print("\n--- Full Descriptions (if available) ---")
    for item in all_results:
        if item.get("description"):
            print(f"URL: {item['source_url']}\nDescription: {item['description']}\n---\n")

    print("\n--- Product Specifications (if available) ---")
    for item in all_results:
        specs = item.get("specifications")
        if specs:
            print(f"URL: {item['source_url']}\nSpecifications:")
            if isinstance(specs, dict):
                for key, value in specs.items():
                    print(f"  - {key}: {value}")
            else:
                print(f"  {specs}")
            print("---\n")