#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ultimate_hybrid_price_analyzer.py

Features:
- Hybrid Mode:
    - URL-based analysis (normal HTTP fetch, no bypass tricks)
    - HTML-based analysis (user-pasted HTML)
- Hard autoloader for dependencies (requests, beautifulsoup4)
- Multi-threaded URL fetching
- Universal price extraction with:
    - Multi-currency detection ($, €, £, ¥, etc.)
    - Basic confidence scoring
- HTML sanitization (removes script/style)
- Price map (where each price was found, with context snippet)
- JSON export of results
- Simple GUI (Tkinter) + CLI
- Plugin system for site-specific extraction overrides
"""

import sys
import subprocess
import re
import json
import threading
from queue import Queue
from typing import List, Optional, Tuple, Dict, Any
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Hard autoloader
# ---------------------------------------------------------------------------

REQUIRED_LIBS = {
    "requests": "requests",
    "bs4": "beautifulsoup4",
    "tkinter": None,  # stdlib on most Python installs; no pip
}


def ensure_libs():
    for module_name, pip_name in REQUIRED_LIBS.items():
        try:
            __import__(module_name)
        except ImportError:
            if pip_name is None:
                print(f"[AUTOLOADER] Missing stdlib module '{module_name}'. "
                      f"Please ensure your Python installation includes it.")
                sys.exit(1)
            print(f"[AUTOLOADER] Missing library '{module_name}'. Installing '{pip_name}' via pip...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                print(f"[AUTOLOADER] Successfully installed '{pip_name}'.")
            except Exception as e:
                print(f"[AUTOLOADER] Failed to install '{pip_name}': {e}")
                sys.exit(1)


ensure_libs()

import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
import tkinter as tk  # type: ignore
from tkinter import scrolledtext, filedialog, messagebox  # type: ignore

# ---------------------------------------------------------------------------
# Plugin system (site-specific extraction overrides)
# ---------------------------------------------------------------------------

# Each plugin: func(html: str) -> List[Dict[str, Any]]
# Return list of price entries:
#   { "value": float, "currency": str, "confidence": float, "context": str }

PLUGIN_REGISTRY: Dict[str, Any] = {}


def register_plugin(domain: str):
    def decorator(func):
        PLUGIN_REGISTRY[domain.lower()] = func
        return func
    return decorator


def get_domain_from_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return None


# Example plugin (placeholder, generic pattern; you can customize per site)
@register_plugin("example.com")
def plugin_example_com(html: str) -> List[Dict[str, Any]]:
    # This is just a demo plugin; real ones would use site-specific selectors.
    soup = BeautifulSoup(html, "html.parser")
    prices = []
    for span in soup.find_all("span", class_="price"):
        text = span.get_text(strip=True)
        price, currency = extract_price_and_currency(text)
        if price is not None:
            prices.append({
                "value": price,
                "currency": currency or "UNKNOWN",
                "confidence": 0.9,
                "context": text[:120]
            })
    return prices


# ---------------------------------------------------------------------------
# Core price extraction logic
# ---------------------------------------------------------------------------

CURRENCY_SYMBOLS = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
}


def sanitize_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return str(soup)


def extract_price_and_currency(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract price and currency from text.
    Supports multiple currency symbols.
    """
    # Look for currency symbol + number OR just number
    match = re.search(r"([€$£¥₹])?\s*(\d+\.\d{2})", text)
    if not match:
        return None, None

    symbol = match.group(1)
    num_str = match.group(2)
    try:
        value = float(num_str)
    except ValueError:
        return None, None

    currency = CURRENCY_SYMBOLS.get(symbol, None) if symbol else None
    return value, currency


def compute_confidence(node_text: str, tag_name: str) -> float:
    """
    Basic confidence scoring:
    - Higher if in typical price tags (span, div, strong)
    - Higher if text is short
    """
    base = 0.5
    tag_bonus = {
        "span": 0.2,
        "div": 0.1,
        "strong": 0.2,
        "b": 0.2,
        "p": 0.05,
    }
    length = len(node_text)
    length_factor = 0.3 if length < 40 else 0.1 if length < 100 else 0.0
    return min(1.0, base + tag_bonus.get(tag_name.lower(), 0.0) + length_factor)


def extract_prices_with_map(html: str) -> List[Dict[str, Any]]:
    """
    Extract all prices with context and confidence.
    Returns list of dicts:
      { "value": float, "currency": str, "confidence": float, "context": str }
    """
    sanitized = sanitize_html(html)
    soup = BeautifulSoup(sanitized, "html.parser")

    prices: List[Dict[str, Any]] = []

    # Search all text nodes that look like they might contain prices
    for element in soup.find_all(string=re.compile(r"[€$£¥₹]?\s*\d+\.\d{2}")):
        text = element.strip()
        if not text:
            continue

        value, currency = extract_price_and_currency(text)
        if value is None:
            continue

        parent = element.parent
        tag_name = parent.name if parent and parent.name else "unknown"
        full_context = parent.get_text(" ", strip=True) if parent else text
        context_snippet = full_context[:200]

        confidence = compute_confidence(full_context, tag_name)

        prices.append({
            "value": value,
            "currency": currency or "UNKNOWN",
            "confidence": confidence,
            "context": context_snippet,
        })

    return prices


def get_lowest_price_entry(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not entries:
        return None
    return min(entries, key=lambda e: e["value"])


# ---------------------------------------------------------------------------
# URL-based mode with multi-threading
# ---------------------------------------------------------------------------

def fetch_html_from_url(url: str, timeout: int = 15) -> Optional[str]:
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


def analyze_single_url(url: str) -> Dict[str, Any]:
    """
    Analyze a single URL:
      - Fetch HTML
      - Apply plugin if available
      - Fallback to generic extraction
    Returns dict with:
      {
        "url": str,
        "prices": [ ... ],
        "lowest": { ... } or None
      }
    """
    html = fetch_html_from_url(url)
    if html is None:
        return {"url": url, "prices": [], "lowest": None}

    domain = get_domain_from_url(url)
    entries: List[Dict[str, Any]] = []

    if domain and domain in PLUGIN_REGISTRY:
        try:
            plugin_entries = PLUGIN_REGISTRY[domain](html)
            if plugin_entries:
                entries.extend(plugin_entries)
        except Exception as e:
            print(f"[PLUGIN ERROR] {domain}: {e}")

    # Always also run generic extractor
    generic_entries = extract_prices_with_map(html)
    entries.extend(generic_entries)

    lowest = get_lowest_price_entry(entries)
    return {"url": url, "prices": entries, "lowest": lowest}


def analyze_urls_multithread(urls: List[str], max_workers: int = 5) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Multi-threaded URL analysis.
    Returns:
      - list of per-URL results
      - overall lowest price entry across all URLs
    """
    results: List[Dict[str, Any]] = []
    q: Queue = Queue()

    for url in urls:
        q.put(url)

    lock = threading.Lock()

    def worker():
        while True:
            try:
                url = q.get_nowait()
            except Exception:
                break
            res = analyze_single_url(url)
            with lock:
                results.append(res)
            q.task_done()

    threads = []
    for _ in range(min(max_workers, len(urls))):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)

    q.join()

    all_entries: List[Dict[str, Any]] = []
    for r in results:
        if r["lowest"] is not None:
            all_entries.append({
                "url": r["url"],
                **r["lowest"]
            })

    overall_lowest = min(all_entries, key=lambda e: e["value"]) if all_entries else None
    return results, overall_lowest


# ---------------------------------------------------------------------------
# HTML-based mode
# ---------------------------------------------------------------------------

def analyze_html_blocks(html_blocks: List[str]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Analyze multiple HTML blocks.
    Returns:
      - list of per-block results
      - overall lowest price entry across all blocks
    """
    results: List[Dict[str, Any]] = []
    all_entries: List[Dict[str, Any]] = []

    for idx, html in enumerate(html_blocks, start=1):
        entries = extract_prices_with_map(html)
        lowest = get_lowest_price_entry(entries)
        block_result = {
            "block_index": idx,
            "prices": entries,
            "lowest": lowest,
        }
        results.append(block_result)
        if lowest is not None:
            all_entries.append({
                "block_index": idx,
                **lowest
            })

    overall_lowest = min(all_entries, key=lambda e: e["value"]) if all_entries else None
    return results, overall_lowest


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_to_json(data: Any, filename: Optional[str] = None) -> str:
    if filename is None:
        filename = "price_analysis_results.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return filename


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def prompt_urls() -> List[str]:
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

    return [u for u in urls if u]


def prompt_html_blocks() -> List[str]:
    print("\nHTML mode:")
    print("Paste one or more HTML blocks.")
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


def print_cli_url_results(results: List[Dict[str, Any]], overall_lowest: Optional[Dict[str, Any]]):
    print("\n================ URL MODE RESULTS ================")
    for r in results:
        url = r["url"]
        lowest = r["lowest"]
        if lowest is None:
            print(f"URL: {url}\n  -> No prices found.")
        else:
            print(f"URL: {url}")
            print(f"  -> Lowest price: {lowest['value']:.2f} {lowest['currency']}")
            print(f"  -> Confidence: {lowest['confidence']:.2f}")
            print(f"  -> Context: {lowest['context'][:120]}")

    if overall_lowest is not None:
        print("\n>>> Overall lowest price across all URLs:")
        print(f"  Value: {overall_lowest['value']:.2f} {overall_lowest['currency']}")
        print(f"  From URL: {overall_lowest.get('url', 'N/A')}")
        print(f"  Confidence: {overall_lowest['confidence']:.2f}")
        print(f"  Context: {overall_lowest['context'][:160]}")
    else:
        print("\n>>> No prices found across any URL.")


def print_cli_html_results(results: List[Dict[str, Any]], overall_lowest: Optional[Dict[str, Any]]):
    print("\n================ HTML MODE RESULTS ================")
    for r in results:
        idx = r["block_index"]
        lowest = r["lowest"]
        if lowest is None:
            print(f"HTML Block #{idx}:\n  -> No prices found.")
        else:
            print(f"HTML Block #{idx}:")
            print(f"  -> Lowest price: {lowest['value']:.2f} {lowest['currency']}")
            print(f"  -> Confidence: {lowest['confidence']:.2f}")
            print(f"  -> Context: {lowest['context'][:120]}")

    if overall_lowest is not None:
        print("\n>>> Overall lowest price across all HTML blocks:")
        print(f"  Value: {overall_lowest['value']:.2f} {overall_lowest['currency']}")
        print(f"  From block: {overall_lowest.get('block_index', 'N/A')}")
        print(f"  Confidence: {overall_lowest['confidence']:.2f}")
        print(f"  Context: {overall_lowest['context'][:160]}")
    else:
        print("\n>>> No prices found across any HTML block.")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class PriceAnalyzerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Ultimate Hybrid Price Analyzer")

        self.mode_var = tk.StringVar(value="url")

        mode_frame = tk.Frame(root)
        mode_frame.pack(fill="x", padx=5, pady=5)

        tk.Radiobutton(mode_frame, text="URL Mode", variable=self.mode_var, value="url").pack(side="left")
        tk.Radiobutton(mode_frame, text="HTML Mode", variable=self.mode_var, value="html").pack(side="left")

        self.input_text = scrolledtext.ScrolledText(root, height=10)
        self.input_text.pack(fill="both", expand=True, padx=5, pady=5)

        btn_frame = tk.Frame(root)
        btn_frame.pack(fill="x", padx=5, pady=5)

        tk.Button(btn_frame, text="Analyze", command=self.analyze).pack(side="left")
        tk.Button(btn_frame, text="Export JSON", command=self.export_json).pack(side="left")
        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side="left")

        self.output_text = scrolledtext.ScrolledText(root, height=15, state="normal")
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.last_results: Dict[str, Any] = {}

    def log(self, msg: str):
        self.output_text.insert("end", msg + "\n")
        self.output_text.see("end")

    def clear(self):
        self.input_text.delete("1.0", "end")
        self.output_text.delete("1.0", "end")
        self.last_results = {}

    def analyze(self):
        self.output_text.delete("1.0", "end")
        mode = self.mode_var.get()
        raw_input = self.input_text.get("1.0", "end").strip()

        if not raw_input:
            messagebox.showwarning("Input Required", "Please enter URLs or HTML.")
            return

        if mode == "url":
            urls = [u.strip() for u in raw_input.replace("\n", ",").split(",") if u.strip()]
            if not urls:
                messagebox.showwarning("Input Error", "No valid URLs found.")
                return
            self.log("[GUI] Analyzing URLs...")
            results, overall_lowest = analyze_urls_multithread(urls)
            self.last_results = {
                "mode": "url",
                "results": results,
                "overall_lowest": overall_lowest,
            }
            # Pretty print
            for r in results:
                url = r["url"]
                lowest = r["lowest"]
                if lowest is None:
                    self.log(f"URL: {url}\n  -> No prices found.\n")
                else:
                    self.log(f"URL: {url}")
                    self.log(f"  -> Lowest price: {lowest['value']:.2f} {lowest['currency']}")
                    self.log(f"  -> Confidence: {lowest['confidence']:.2f}")
                    self.log(f"  -> Context: {lowest['context'][:160]}\n")

            if overall_lowest is not None:
                self.log(">>> Overall lowest price across all URLs:")
                self.log(f"  Value: {overall_lowest['value']:.2f} {overall_lowest['currency']}")
                self.log(f"  From URL: {overall_lowest.get('url', 'N/A')}")
                self.log(f"  Confidence: {overall_lowest['confidence']:.2f}")
                self.log(f"  Context: {overall_lowest['context'][:200]}")
            else:
                self.log(">>> No prices found across any URL.")

        else:  # HTML mode
            # Split blocks by a delimiter line '---' or treat all as one block
            blocks_raw = [b.strip() for b in raw_input.split("\n---\n") if b.strip()]
            if not blocks_raw:
                messagebox.showwarning("Input Error", "No valid HTML blocks found.")
                return
            self.log("[GUI] Analyzing HTML blocks...")
            results, overall_lowest = analyze_html_blocks(blocks_raw)
            self.last_results = {
                "mode": "html",
                "results": results,
                "overall_lowest": overall_lowest,
            }

            for r in results:
                idx = r["block_index"]
                lowest = r["lowest"]
                if lowest is None:
                    self.log(f"HTML Block #{idx}:\n  -> No prices found.\n")
                else:
                    self.log(f"HTML Block #{idx}:")
                    self.log(f"  -> Lowest price: {lowest['value']:.2f} {lowest['currency']}")
                    self.log(f"  -> Confidence: {lowest['confidence']:.2f}")
                    self.log(f"  -> Context: {lowest['context'][:160]}\n")

            if overall_lowest is not None:
                self.log(">>> Overall lowest price across all HTML blocks:")
                self.log(f"  Value: {overall_lowest['value']:.2f} {overall_lowest['currency']}")
                self.log(f"  From block: {overall_lowest.get('block_index', 'N/A')}")
                self.log(f"  Confidence: {overall_lowest['confidence']:.2f}")
                self.log(f"  Context: {overall_lowest['context'][:200]}")
            else:
                self.log(">>> No prices found across any HTML block.")

    def export_json(self):
        if not self.last_results:
            messagebox.showinfo("No Data", "No analysis results to export.")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save JSON Results"
        )
        if not filename:
            return
        try:
            export_to_json(self.last_results, filename)
            messagebox.showinfo("Export Successful", f"Results exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main_cli():
    print("==========================================================")
    print(" ULTIMATE HYBRID PRICE ANALYZER")
    print(" (URL + HTML, Autoloader, Threads, JSON, GUI, Plugins)")
    print("==========================================================\n")

    while True:
        print("Select mode:")
        print("  1) URL-based price analysis (multi-threaded)")
        print("  2) HTML-based price analysis (paste HTML)")
        print("  3) Launch GUI")
        print("  4) Exit")
        choice = input("\nEnter choice (1/2/3/4): ").strip()

        if choice == "1":
            urls = prompt_urls()
            if not urls:
                print("[WARN] No URLs provided.\n")
                continue
            results, overall_lowest = analyze_urls_multithread(urls)
            print_cli_url_results(results, overall_lowest)

            export_choice = input("\nExport results to JSON? (y/n): ").strip().lower()
            if export_choice == "y":
                data = {
                    "mode": "url",
                    "results": results,
                    "overall_lowest": overall_lowest,
                }
                filename = export_to_json(data)
                print(f"[INFO] Results exported to: {filename}\n")

        elif choice == "2":
            blocks = prompt_html_blocks()
            if not blocks:
                print("[WARN] No HTML blocks provided.\n")
                continue
            results, overall_lowest = analyze_html_blocks(blocks)
            print_cli_html_results(results, overall_lowest)

            export_choice = input("\nExport results to JSON? (y/n): ").strip().lower()
            if export_choice == "y":
                data = {
                    "mode": "html",
                    "results": results,
                    "overall_lowest": overall_lowest,
                }
                filename = export_to_json(data)
                print(f"[INFO] Results exported to: {filename}\n")

        elif choice == "3":
            root = tk.Tk()
            app = PriceAnalyzerGUI(root)
            root.mainloop()

        elif choice == "4":
            print("Exiting Ultimate Hybrid Price Analyzer.")
            break
        else:
            print("[WARN] Invalid choice. Please enter 1, 2, 3, or 4.\n")


if __name__ == "__main__":
    main_cli()
