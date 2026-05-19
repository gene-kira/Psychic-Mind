#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_price_analyzer.py

Hybrid price analyzer:
- Mode 1: URL-based (normal HTTP fetch, no bypass tricks)
- Mode 2: HTML-based (you paste HTML manually)
- Compares all visible prices and reports the lowest.

Includes a "hard autoloader" that installs missing libraries via pip.
"""

import sys
import subprocess
import re
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Hard autoloader for required libraries
# ---------------------------------------------------------------------------

REQUIRED_LIBS = {
    "requests": "requests",
    "bs4": "beautifulsoup4",
}


def ensure_libs():
    """Ensure all required libraries are installed, installing via pip if needed."""
    for module_name, pip_name in REQUIRED_LIBS.items():
        try:
            __import__(module_name)
        except ImportError:
            print(f"[AUTOLOADER] Missing library '{module_name}'. Installing '{pip_name}' via pip...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                print(f"[AUTOLOADER] Successfully installed '{pip_name}'.")
            except Exception as e:
                print(f"[AUTOLOADER] Failed to install '{pip_name}': {e}")
                print("Exiting because required dependencies could not be installed.")
                sys.exit(1)


ensure_libs()

import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore

# ---------------------------------------------------------------------------
# Core price extraction logic
# ---------------------------------------------------------------------------


def extract_price_from_text(text: str) -> Optional[float]:
    """
    Extract a single price from a text fragment.
    Matches patterns like:
      $39.99
      39.99
      €12.50
    Returns the first match as float, or None.
    """
    # Allow optional currency symbol and spaces, then number with 2 decimals
    match = re.search(r"[€$£]?\s*(\d+\.\d{2})", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_all_prices_from_html(html: str) -> List[float]:
    """
    Parse HTML and extract all visible price-like values.
    This is generic: it scans all text nodes for price patterns.
    """
    soup = BeautifulSoup(html, "html.parser")
    prices: List[float] = []

    # Get all text nodes that look like they might contain prices
    text_nodes = soup.find_all(string=re.compile(r"[€$£]?\s*\d+\.\d{2}"))

    for node in text_nodes:
        price = extract_price_from_text(str(node))
        if price is not None:
            prices.append(price)

    return prices


def get_lowest_price_from_html(html: str) -> Optional[float]:
    """
    Given raw HTML, return the lowest price found, or None if no prices.
    """
    prices = extract_all_prices_from_html(html)
    if not prices:
        return None
    return min(prices)


# ---------------------------------------------------------------------------
# URL-based mode (normal HTTP fetch)
# ---------------------------------------------------------------------------


def fetch_html_from_url(url: str, timeout: int = 15) -> Optional[str]:
    """
    Fetch HTML from a URL using a normal HTTP GET request.
    No special tricks, just a standard request.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[ERROR] Failed to fetch URL '{url}': {e}")
        return None


def analyze_urls(urls: List[str]) -> Tuple[List[Tuple[str, Optional[float]]], Optional[float]]:
    """
    For each URL:
      - Fetch HTML
      - Extract lowest price from that page
    Returns:
      - list of (url, lowest_price_on_that_page)
      - overall lowest price across all URLs (or None)
    """
    results: List[Tuple[str, Optional[float]]] = []
    all_prices: List[float] = []

    for url in urls:
        print(f"\n[INFO] Fetching: {url}")
        html = fetch_html_from_url(url)
        if html is None:
            results.append((url, None))
            continue

        lowest = get_lowest_price_from_html(html)
        results.append((url, lowest))
        if lowest is not None:
            all_prices.append(lowest)

    overall_lowest = min(all_prices) if all_prices else None
    return results, overall_lowest


# ---------------------------------------------------------------------------
# HTML-based mode (user-pasted HTML)
# ---------------------------------------------------------------------------


def analyze_html_blocks(html_blocks: List[str]) -> Tuple[List[Tuple[int, Optional[float]]], Optional[float]]:
    """
    For each HTML block:
      - Extract lowest price
    Returns:
      - list of (index, lowest_price_in_block)
      - overall lowest price across all blocks (or None)
    """
    results: List[Tuple[int, Optional[float]]] = []
    all_prices: List[float] = []

    for idx, html in enumerate(html_blocks, start=1):
        lowest = get_lowest_price_from_html(html)
        results.append((idx, lowest))
        if lowest is not None:
            all_prices.append(lowest)

    overall_lowest = min(all_prices) if all_prices else None
    return results, overall_lowest


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def prompt_urls() -> List[str]:
    """
    Prompt user to enter one or more URLs (comma-separated or line-by-line).
    """
    print("\nEnter one or more product URLs.")
    print("You can either:")
    print("  - Paste them comma-separated on one line, or")
    print("  - Enter them one per line, then press ENTER on an empty line to finish.\n")

    line = input("URLs (comma-separated or leave blank to enter line-by-line): ").strip()
    urls: List[str] = []

    if line:
        urls = [u.strip() for u in line.split(",") if u.strip()]
    else:
        print("Enter URLs one per line. Press ENTER on an empty line when done.")
        while True:
            u = input("URL: ").strip()
            if not u:
                break
            urls.append(u)

    urls = [u for u in urls if u]
    return urls


def prompt_html_blocks() -> List[str]:
    """
    Prompt user to paste one or more HTML blocks.
    Each block is terminated by a line containing only 'END'.
    An empty ENTER at the start exits.
    """
    print("\nHTML mode:")
    print("You can paste one or more HTML blocks.")
    print("For each block:")
    print("  - Paste the HTML")
    print("  - On a new line, type: END")
    print("  - Then press ENTER")
    print("Press ENTER on an empty line at the start to stop.\n")

    blocks: List[str] = []

    while True:
        print("Paste HTML block (or press ENTER to stop):")
        first_line = input()
        if not first_line.strip():
            break

        lines = [first_line]
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)

        html_block = "\n".join(lines)
        blocks.append(html_block)
        print("[INFO] HTML block captured.\n")

    return blocks


def print_url_results(results: List[Tuple[str, Optional[float]]], overall_lowest: Optional[float]):
    print("\n================ URL MODE RESULTS ================")
    for url, price in results:
        if price is None:
            print(f"URL: {url}\n  -> No prices found.")
        else:
            print(f"URL: {url}\n  -> Lowest price on page: {price:.2f}")
    if overall_lowest is not None:
        print(f"\n>>> Overall lowest price across all URLs: {overall_lowest:.2f}")
    else:
        print("\n>>> No prices found across any URL.")


def print_html_results(results: List[Tuple[int, Optional[float]]], overall_lowest: Optional[float]):
    print("\n================ HTML MODE RESULTS ================")
    for idx, price in results:
        if price is None:
            print(f"HTML Block #{idx}:\n  -> No prices found.")
        else:
            print(f"HTML Block #{idx}:\n  -> Lowest price in block: {price:.2f}")
    if overall_lowest is not None:
        print(f"\n>>> Overall lowest price across all HTML blocks: {overall_lowest:.2f}")
    else:
        print("\n>>> No prices found across any HTML block.")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main():
    print("===================================================")
    print(" HYBRID PRICE ANALYZER (URL + HTML, Hard Autoloader)")
    print("===================================================\n")

    while True:
        print("Select mode:")
        print("  1) URL-based price analysis")
        print("  2) HTML-based price analysis (paste HTML)")
        print("  3) Exit")
        choice = input("\nEnter choice (1/2/3): ").strip()

        if choice == "1":
            urls = prompt_urls()
            if not urls:
                print("[WARN] No URLs provided.\n")
                continue
            results, overall_lowest = analyze_urls(urls)
            print_url_results(results, overall_lowest)
            print("\n")
        elif choice == "2":
            blocks = prompt_html_blocks()
            if not blocks:
                print("[WARN] No HTML blocks provided.\n")
                continue
            results, overall_lowest = analyze_html_blocks(blocks)
            print_html_results(results, overall_lowest)
            print("\n")
        elif choice == "3":
            print("Exiting Hybrid Price Analyzer.")
            break
        else:
            print("[WARN] Invalid choice. Please enter 1, 2, or 3.\n")


if __name__ == "__main__":
    main()
