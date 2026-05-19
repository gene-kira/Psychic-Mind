#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
guardian_price_sentinel.py

Features:
- Hybrid price analysis:
    - URL-based (multi-threaded, normal HTTP fetch)
    - HTML-based (user-pasted HTML)
- Dark-mode GUI
- Hard autoloader (requests, beautifulsoup4)
- SQLite price-history database
- Monitoring engine:
    - Periodically re-checks configured URLs
    - Stores results in DB
    - Optional cloud sync
- Cloud sync:
    - POSTs JSON payloads to user-configured endpoint
- Universal price extraction:
    - Multi-currency detection ($, €, £, ¥, ₹)
    - Confidence scoring
    - HTML sanitization
    - Price map (context)
- JSON export
"""

import sys
import subprocess
import re
import json
import threading
import sqlite3
import time
from queue import Queue
from typing import List, Optional, Tuple, Dict, Any
from urllib.parse import urlparse
from datetime import datetime

# ---------------------------------------------------------------------------
# AUTOLOADER
# ---------------------------------------------------------------------------

REQUIRED_LIBS = {
    "requests": "requests",
    "bs4": "beautifulsoup4",
    "tkinter": None,
}


def ensure_libs():
    for module_name, pip_name in REQUIRED_LIBS.items():
        try:
            __import__(module_name)
        except ImportError:
            if pip_name is None:
                print(f"[AUTOLOADER] Missing stdlib module '{module_name}'.")
                sys.exit(1)
            print(f"[AUTOLOADER] Installing '{pip_name}'...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            except Exception as e:
                print(f"[AUTOLOADER] Failed to install '{pip_name}': {e}")
                sys.exit(1)


ensure_libs()

import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
import tkinter as tk  # type: ignore
from tkinter import scrolledtext, filedialog, messagebox  # type: ignore

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CURRENCY_SYMBOLS = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
}

DB_FILE = "price_sentinel_history.db"

# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------

class PriceHistoryDB:
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    source TEXT NOT NULL,
                    value REAL NOT NULL,
                    currency TEXT,
                    confidence REAL,
                    context TEXT
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS monitor_config (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    urls TEXT,
                    interval_sec INTEGER,
                    cloud_endpoint TEXT,
                    cloud_api_key TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def add_entry(self, mode: str, source: str, value: float,
                  currency: str, confidence: float, context: str):
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            ts = datetime.utcnow().isoformat()
            c.execute(
                """
                INSERT INTO price_history (timestamp, mode, source, value, currency, confidence, context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, mode, source, value, currency, confidence, context[:500]),
            )
            conn.commit()
        finally:
            conn.close()

    def get_recent_entries(self, limit: int = 20) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute(
                """
                SELECT timestamp, mode, source, value, currency, confidence, context
                FROM price_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = c.fetchall()
        finally:
            conn.close()

        entries = []
        for row in rows:
            ts, mode, source, value, currency, confidence, context = row
            entries.append({
                "timestamp": ts,
                "mode": mode,
                "source": source,
                "value": value,
                "currency": currency,
                "confidence": confidence,
                "context": context,
            })
        return entries

    def save_monitor_config(self, urls: List[str], interval_sec: int,
                            cloud_endpoint: str, cloud_api_key: str):
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            urls_json = json.dumps(urls)
            c.execute(
                """
                INSERT INTO monitor_config (id, urls, interval_sec, cloud_endpoint, cloud_api_key)
                VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    urls=excluded.urls,
                    interval_sec=excluded.interval_sec,
                    cloud_endpoint=excluded.cloud_endpoint,
                    cloud_api_key=excluded.cloud_api_key
                """,
                (urls_json, interval_sec, cloud_endpoint, cloud_api_key),
            )
            conn.commit()
        finally:
            conn.close()

    def load_monitor_config(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT urls, interval_sec, cloud_endpoint, cloud_api_key FROM monitor_config WHERE id=1")
            row = c.fetchone()
        finally:
            conn.close()

        if not row:
            return {
                "urls": [],
                "interval_sec": 600,
                "cloud_endpoint": "",
                "cloud_api_key": "",
            }

        urls_json, interval_sec, cloud_endpoint, cloud_api_key = row
        try:
            urls = json.loads(urls_json) if urls_json else []
        except Exception:
            urls = []
        return {
            "urls": urls,
            "interval_sec": interval_sec or 600,
            "cloud_endpoint": cloud_endpoint or "",
            "cloud_api_key": cloud_api_key or "",
        }


price_history_db = PriceHistoryDB()

# ---------------------------------------------------------------------------
# CORE EXTRACTION
# ---------------------------------------------------------------------------

def sanitize_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return str(soup)


def extract_price_and_currency(text: str) -> Tuple[Optional[float], Optional[str]]:
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
    sanitized = sanitize_html(html)
    soup = BeautifulSoup(sanitized, "html.parser")

    prices: List[Dict[str, Any]] = []

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
# URL ANALYZER
# ---------------------------------------------------------------------------

class URLAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

    def fetch_html(self, url: str, timeout: int = 15) -> Optional[str]:
        try:
            resp = self.session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"[ERROR] Failed to fetch URL '{url}': {e}")
            return None

    def analyze_single_url(self, url: str) -> Dict[str, Any]:
        html = self.fetch_html(url)
        if html is None:
            return {"url": url, "prices": [], "lowest": None}

        entries = extract_prices_with_map(html)
        lowest = get_lowest_price_entry(entries)
        if lowest is not None:
            price_history_db.add_entry(
                mode="url",
                source=url,
                value=lowest["value"],
                currency=lowest["currency"],
                confidence=lowest["confidence"],
                context=lowest["context"],
            )

        return {"url": url, "prices": entries, "lowest": lowest}

    def analyze_urls_multithread(self, urls: List[str], max_workers: int = 5) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
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
                res = self.analyze_single_url(url)
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


url_analyzer = URLAnalyzer()

# ---------------------------------------------------------------------------
# HTML ANALYSIS
# ---------------------------------------------------------------------------

def analyze_html_blocks(html_blocks: List[str]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
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
            price_history_db.add_entry(
                mode="html",
                source=f"block_{idx}",
                value=lowest["value"],
                currency=lowest["currency"],
                confidence=lowest["confidence"],
                context=lowest["context"],
            )

    overall_lowest = min(all_entries, key=lambda e: e["value"]) if all_entries else None
    return results, overall_lowest


# ---------------------------------------------------------------------------
# CLOUD SYNC CLIENT
# ---------------------------------------------------------------------------

class CloudSyncClient:
    def __init__(self):
        self.session = requests.Session()

    def sync_entry(self, endpoint: str, api_key: str, payload: Dict[str, Any]) -> bool:
        if not endpoint:
            return False
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            resp = self.session.post(endpoint, headers=headers, json=payload, timeout=10)
            return resp.status_code // 100 == 2
        except Exception as e:
            print(f"[CLOUD] Sync failed: {e}")
            return False


cloud_client = CloudSyncClient()

# ---------------------------------------------------------------------------
# MONITORING ENGINE
# ---------------------------------------------------------------------------

class MonitorEngine:
    def __init__(self, db: PriceHistoryDB, analyzer: URLAnalyzer, cloud: CloudSyncClient):
        self.db = db
        self.analyzer = analyzer
        self.cloud = cloud
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._status = "stopped"

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self._status = "running"

    def stop(self):
        with self._lock:
            self._stop_event.set()
            self._status = "stopped"

    def status(self) -> str:
        with self._lock:
            return self._status

    def _run_loop(self):
        while not self._stop_event.is_set():
            cfg = self.db.load_monitor_config()
            urls = cfg["urls"]
            interval = max(60, int(cfg["interval_sec"] or 600))
            endpoint = cfg["cloud_endpoint"]
            api_key = cfg["cloud_api_key"]

            if urls:
                print(f"[MONITOR] Checking {len(urls)} URLs...")
                results, overall_lowest = self.analyzer.analyze_urls_multithread(urls)
                if overall_lowest is not None and endpoint:
                    payload = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "mode": "monitor",
                        "overall_lowest": overall_lowest,
                        "urls": urls,
                    }
                    ok = self.cloud.sync_entry(endpoint, api_key, payload)
                    print(f"[MONITOR] Cloud sync: {'OK' if ok else 'FAILED'}")
            else:
                print("[MONITOR] No URLs configured.")

            for _ in range(interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)


monitor_engine = MonitorEngine(price_history_db, url_analyzer, cloud_client)

# ---------------------------------------------------------------------------
# JSON EXPORT
# ---------------------------------------------------------------------------

def export_to_json(data: Any, filename: Optional[str] = None) -> str:
    if filename is None:
        filename = "price_sentinel_results.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return filename


# ---------------------------------------------------------------------------
# CLI HELPERS
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


def print_cli_history(entries: List[Dict[str, Any]]):
    print("\n================ PRICE HISTORY (MOST RECENT) ================")
    if not entries:
        print("No history entries found.")
        return
    for e in entries:
        print(f"[{e['timestamp']}] mode={e['mode']} source={e['source']}")
        print(f"  value={e['value']:.2f} {e['currency']}  confidence={e['confidence']:.2f}")
        print(f"  context={e['context'][:120]}")
        print("")


# ---------------------------------------------------------------------------
# DARK-MODE GUI WITH MONITOR CONFIG + CLOUD SYNC
# ---------------------------------------------------------------------------

class PriceSentinelGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Guardian Price Sentinel (Dark Mode)")

        bg = "#1e1e1e"
        fg = "#f0f0f0"
        accent = "#3a3d41"

        self.root.configure(bg=bg)

        self.mode_var = tk.StringVar(value="url")

        mode_frame = tk.Frame(root, bg=bg)
        mode_frame.pack(fill="x", padx=5, pady=5)

        tk.Radiobutton(mode_frame, text="URL Mode", variable=self.mode_var, value="url",
                       bg=bg, fg=fg, selectcolor=accent).pack(side="left")
        tk.Radiobutton(mode_frame, text="HTML Mode", variable=self.mode_var, value="html",
                       bg=bg, fg=fg, selectcolor=accent).pack(side="left")

        self.input_text = scrolledtext.ScrolledText(root, height=6, bg="#252526", fg=fg,
                                                    insertbackground=fg)
        self.input_text.pack(fill="both", expand=True, padx=5, pady=5)

        btn_frame = tk.Frame(root, bg=bg)
        btn_frame.pack(fill="x", padx=5, pady=5)

        tk.Button(btn_frame, text="Analyze", command=self.analyze,
                  bg=accent, fg=fg).pack(side="left")
        tk.Button(btn_frame, text="Export JSON", command=self.export_json,
                  bg=accent, fg=fg).pack(side="left")
        tk.Button(btn_frame, text="Show History", command=self.show_history,
                  bg=accent, fg=fg).pack(side="left")
        tk.Button(btn_frame, text="Clear", command=self.clear,
                  bg=accent, fg=fg).pack(side="left")

        # Monitor config panel
        monitor_frame = tk.LabelFrame(root, text="Monitoring & Cloud Sync", bg=bg, fg=fg)
        monitor_frame.pack(fill="x", padx=5, pady=5)

        tk.Label(monitor_frame, text="Monitor URLs (comma-separated):", bg=bg, fg=fg).grid(row=0, column=0, sticky="w")
        self.monitor_urls_entry = tk.Entry(monitor_frame, bg="#252526", fg=fg, insertbackground=fg, width=80)
        self.monitor_urls_entry.grid(row=0, column=1, sticky="we", padx=5, pady=2)

        tk.Label(monitor_frame, text="Interval (sec):", bg=bg, fg=fg).grid(row=1, column=0, sticky="w")
        self.interval_entry = tk.Entry(monitor_frame, bg="#252526", fg=fg, insertbackground=fg, width=10)
        self.interval_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        tk.Label(monitor_frame, text="Cloud endpoint URL:", bg=bg, fg=fg).grid(row=2, column=0, sticky="w")
        self.cloud_endpoint_entry = tk.Entry(monitor_frame, bg="#252526", fg=fg, insertbackground=fg, width=80)
        self.cloud_endpoint_entry.grid(row=2, column=1, sticky="we", padx=5, pady=2)

        tk.Label(monitor_frame, text="Cloud API key (optional):", bg=bg, fg=fg).grid(row=3, column=0, sticky="w")
        self.cloud_api_key_entry = tk.Entry(monitor_frame, bg="#252526", fg=fg, insertbackground=fg, width=80, show="*")
        self.cloud_api_key_entry.grid(row=3, column=1, sticky="we", padx=5, pady=2)

        monitor_btn_frame = tk.Frame(monitor_frame, bg=bg)
        monitor_btn_frame.grid(row=4, column=0, columnspan=2, sticky="w", pady=4)

        tk.Button(monitor_btn_frame, text="Save Config", command=self.save_monitor_config,
                  bg=accent, fg=fg).pack(side="left", padx=2)
        tk.Button(monitor_btn_frame, text="Start Monitor", command=self.start_monitor,
                  bg=accent, fg=fg).pack(side="left", padx=2)
        tk.Button(monitor_btn_frame, text="Stop Monitor", command=self.stop_monitor,
                  bg=accent, fg=fg).pack(side="left", padx=2)

        monitor_frame.columnconfigure(1, weight=1)

        self.output_text = scrolledtext.ScrolledText(root, height=14, bg="#252526", fg=fg,
                                                     insertbackground=fg)
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.last_results: Dict[str, Any] = {}

        self._load_monitor_config_into_gui()
        self._status_updater()

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
            results, overall_lowest = url_analyzer.analyze_urls_multithread(urls)
            self.last_results = {
                "mode": "url",
                "results": results,
                "overall_lowest": overall_lowest,
            }
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

        else:
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

    def show_history(self):
        entries = price_history_db.get_recent_entries(30)
        if not entries:
            messagebox.showinfo("History", "No history entries found.")
            return
        self.output_text.insert("end", "\n=== PRICE HISTORY (MOST RECENT) ===\n")
        for e in entries:
            self.output_text.insert(
                "end",
                f"[{e['timestamp']}] mode={e['mode']} source={e['source']}\n"
                f"  value={e['value']:.2f} {e['currency']}  confidence={e['confidence']:.2f}\n"
                f"  context={e['context'][:160]}\n\n"
            )
        self.output_text.see("end")

    def _load_monitor_config_into_gui(self):
        cfg = price_history_db.load_monitor_config()
        self.monitor_urls_entry.delete(0, "end")
        self.monitor_urls_entry.insert(0, ", ".join(cfg["urls"]))
        self.interval_entry.delete(0, "end")
        self.interval_entry.insert(0, str(cfg["interval_sec"]))
        self.cloud_endpoint_entry.delete(0, "end")
        self.cloud_endpoint_entry.insert(0, cfg["cloud_endpoint"])
        self.cloud_api_key_entry.delete(0, "end")
        self.cloud_api_key_entry.insert(0, cfg["cloud_api_key"])

    def save_monitor_config(self):
        urls_raw = self.monitor_urls_entry.get().strip()
        urls = [u.strip() for u in urls_raw.split(",") if u.strip()]
        try:
            interval = int(self.interval_entry.get().strip() or "600")
        except ValueError:
            messagebox.showerror("Invalid Interval", "Interval must be an integer (seconds).")
            return
        endpoint = self.cloud_endpoint_entry.get().strip()
        api_key = self.cloud_api_key_entry.get().strip()
        price_history_db.save_monitor_config(urls, interval, endpoint, api_key)
        messagebox.showinfo("Config Saved", "Monitoring configuration saved.")

    def start_monitor(self):
        self.save_monitor_config()
        monitor_engine.start()
        self.log("[MONITOR] Started.")

    def stop_monitor(self):
        monitor_engine.stop()
        self.log("[MONITOR] Stopped.")

    def _status_updater(self):
        status = monitor_engine.status()
        self.root.title(f"Guardian Price Sentinel (Dark Mode) - Monitor: {status}")
        self.root.after(1000, self._status_updater)


# ---------------------------------------------------------------------------
# MAIN CLI
# ---------------------------------------------------------------------------

def main_cli():
    print("===================================================================")
    print(" GUARDIAN PRICE SENTINEL")
    print(" (Hybrid Analyzer + Dark GUI + DB + Cloud Sync + Monitoring)")
    print("===================================================================\n")

    while True:
        print("Select mode:")
        print("  1) URL-based price analysis (multi-threaded)")
        print("  2) HTML-based price analysis (paste HTML)")
        print("  3) Show recent price history")
        print("  4) Launch GUI")
        print("  5) Start monitoring (background)")
        print("  6) Stop monitoring")
        print("  7) Exit")
        choice = input("\nEnter choice (1-7): ").strip()

        if choice == "1":
            urls = prompt_urls()
            if not urls:
                print("[WARN] No URLs provided.\n")
                continue
            results, overall_lowest = url_analyzer.analyze_urls_multithread(urls)
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
            entries = price_history_db.get_recent_entries(30)
            print_cli_history(entries)

        elif choice == "4":
            root = tk.Tk()
            app = PriceSentinelGUI(root)
            root.mainloop()

        elif choice == "5":
            monitor_engine.start()
            print("[MONITOR] Started.\n")

        elif choice == "6":
            monitor_engine.stop()
            print("[MONITOR] Stopped.\n")

        elif choice == "7":
            print("Exiting Guardian Price Sentinel.")
            break
        else:
            print("[WARN] Invalid choice.\n")


if __name__ == "__main__":
    main_cli()
