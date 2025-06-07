
url_dict = {
  "MSE Supplies": [
    "https://www.msesupplies.com/products/mse-pro-4-inch-sapphire-wafer-c-plane-single-or-double-side-polish-al-sub-2-sub-o-sub-3-sub-single-crystal"
  ],
  "CRYSCORE": [
    "https://www.cryscore.com/products/4-inch-c-plane-0001-sapphire-wafers.html"
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


import modal
import requests
from bs4 import BeautifulSoup
import re
import urllib.request
import subprocess

import requests
app = modal.App(name="link-scraper")


@app.function()
def get_links(url):
    response = urllib.request.urlopen(url)
    html = response.read().decode("utf8")
    links = []
    for match in re.finditer('href="(.*?)"', html):
        links.append(match.group(1))
    return links


@app.local_entrypoint()
def main(url):
    links = get_links.remote(url)
    print(links)


if __name__ == "__main__":
    for vendor, urls in url_dict.items():
        for url in urls:
            print(f"Running Modal job for: {vendor} - {url}")
            subprocess.run([
                "python", "-m", "modal", "run", "webscrapper.py",
                "--url", url
            ])




