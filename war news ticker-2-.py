import importlib
import subprocess
import sys
import pkgutil
import shutil
import threading
import queue
import datetime
import webbrowser

# ---------------- AUTOLOADER ---------------- #

class AutoLoader:
    def __init__(
        self,
        required_libs,
        versions=None,
        silent=False,
        log_file="autoloader.log",
    ):
        """
        required_libs = {"module_name": "pip_name"}
        versions = {"pip_name": "1.2.3"}  # optional version pinning
        """
        self.required_libs = required_libs
        self.versions = versions or {}
        self.silent = silent
        self.log_file = log_file
        self.log_queue = queue.Queue()

        self._start_logger()
        self._detect_environment()

    def _start_logger(self):
        def logger_thread():
            with open(self.log_file, "a", encoding="utf-8") as f:
                while True:
                    msg = self.log_queue.get()
                    if msg == "__STOP__":
                        break
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] {msg}\n")

        t = threading.Thread(target=logger_thread, daemon=True)
        t.start()
        self.logger_thread = t

    def log(self, msg):
        if not self.silent:
            print(msg)
        self.log_queue.put(msg)

    def _detect_environment(self):
        self.in_venv = (
            hasattr(sys, "real_prefix")
            or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        )
        if self.in_venv:
            self.log("[AUTOLOADER] Running inside a virtual environment.")
        else:
            self.log("[AUTOLOADER] WARNING: Not inside a virtual environment.")

    def _internet_available(self):
        # Very simple check: if we have 'ping', assume some network stack exists.
        return shutil.which("ping") is not None

    def _install_package(self, pip_name):
        version = self.versions.get(pip_name)
        pkg = f"{pip_name}=={version}" if version else pip_name

        for attempt in range(3):
            try:
                self.log(f"[AUTOLOADER] Installing {pkg} (attempt {attempt+1}/3)...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                return True
            except Exception as e:
                self.log(f"[AUTOLOADER] Install failed: {e}")

        return False

    def _verify_import(self, module_name):
        try:
            importlib.import_module(module_name)
            return True
        except Exception:
            return False

    def _warn_dependency_conflicts(self):
        try:
            output = subprocess.check_output(
                [sys.executable, "-m", "pip", "check"],
                stderr=subprocess.STDOUT
            ).decode()
            if output.strip():
                self.log("[AUTOLOADER] DEPENDENCY CONFLICT WARNING:")
                self.log(output)
        except Exception:
            pass

    def load(self):
        if not self._internet_available():
            self.log("[AUTOLOADER] WARNING: No internet detected. Install may fail.")

        threads = []

        for module_name, pip_name in self.required_libs.items():
            if pkgutil.find_loader(module_name) is None:
                t = threading.Thread(target=self._install_package, args=(pip_name,))
                threads.append(t)
                t.start()
            else:
                self.log(f"[AUTOLOADER] Found: {module_name}")

        for t in threads:
            t.join()

        imported = {}
        for module_name in self.required_libs.keys():
            if self._verify_import(module_name):
                imported[module_name] = importlib.import_module(module_name)
                self.log(f"[AUTOLOADER] Imported: {module_name}")
            else:
                self.log(f"[AUTOLOADER] ERROR importing {module_name}")

        self._warn_dependency_conflicts()

        self.log_queue.put("__STOP__")
        return imported


# ---------------- LOAD LIBS ---------------- #

loader = AutoLoader(
    required_libs={
        "pandas": "pandas",
        "yfinance": "yfinance",
        "requests": "requests",
        "feedparser": "feedparser",
        "tkinter": "tk",  # tkinter is stdlib; pip 'tk' only if needed
    },
    versions={},
    silent=False,
)

libs = loader.load()

pd = libs.get("pandas")
yf = libs.get("yfinance")
requests = libs.get("requests")
feedparser = libs.get("feedparser")

# tkinter import (stdlib)
try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    print("Tkinter not available. Install it via your OS package manager.")
    sys.exit(1)

# Optional: sound alert on Windows
try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False


# ---------------- DATA + CONSTANTS ---------------- #

CLOSURE_DATE = datetime.date(2026, 2, 28)  # adjust if needed

TICKERS = {
    "Brent": "BZ=F",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "VIX": "^VIX",
}

# Public RSS feeds (examples; you can add/remove)
RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/worldNews",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
]

WAR_KEYWORDS = [
    "iran", "tehran", "hormuz", "strait of hormuz",
    "missile", "strike", "attack", "gulf", "middle east",
    "israel", "us ", "u.s.", "tanker", "naval", "drone"
]


# ---------------- MARKET DATA ---------------- #

def fetch_latest_prices():
    end = datetime.date.today()
    start = end - datetime.timedelta(days=10)

    try:
        data = yf.download(
            list(TICKERS.values()),
            start=start.isoformat(),
            end=(end + datetime.timedelta(days=1)).isoformat(),
            progress=False,
        )["Adj Close"]
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return {}

    if isinstance(data, pd.Series):
        data = data.to_frame()

    latest = {}
    for name, ticker in TICKERS.items():
        if ticker in data.columns:
            series = data[ticker].dropna()
            if not series.empty:
                latest[name] = float(series.iloc[-1])
            else:
                latest[name] = None
        else:
            latest[name] = None

    return latest


def compute_hormuz_stress(prices):
    """
    Multi-asset geopolitical stress index.
    Output: 0–100
    """

    brent = prices.get("Brent")
    vix = prices.get("VIX")
    gold = prices.get("Gold")
    silver = prices.get("Silver")
    btc = prices.get("BTC")
    eth = prices.get("ETH")

    if None in [brent, vix, gold, silver, btc, eth]:
        return 0.0

    # --- Oil spike (Brent) ---
    oil_component = max(0, min((brent - 80) * 0.6, 25))  # 80 → baseline, 120 → max

    # --- Volatility spike (VIX) ---
    vix_component = max(0, min((vix - 15) * 1.5, 25))    # 15 → calm, 30 → panic

    # --- Gold surge (flight to safety) ---
    gold_component = max(0, min((gold - 1900) * 0.02, 15))

    # --- Silver surge ---
    silver_component = max(0, min((silver - 24) * 0.5, 10))

    # --- BTC crash ---
    btc_component = max(0, min((35000 - btc) / 800, 15))  # 35k baseline → 20k panic

    # --- ETH crash ---
    eth_component = max(0, min((2500 - eth) / 150, 10))

    # --- Cross-asset panic correlation ---
    panic_assets = 0
    if brent > 100:
        panic_assets += 1
    if vix > 25:
        panic_assets += 1
    if gold > 2000:
        panic_assets += 1
    if btc < 30000:
        panic_assets += 1

    correlation_component = panic_assets * 3  # up to +12

    stress = (
        oil_component +
        vix_component +
        gold_component +
        silver_component +
        btc_component +
        eth_component +
        correlation_component
    )

    return max(0.0, min(stress, 100.0))


# ---------------- WAR NEWS (RSS + API) ---------------- #

def headline_matches_war(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(k in lower for k in WAR_KEYWORDS)


def fetch_war_news_rss(max_items=50):
    """
    Fetch headlines from RSS feeds and filter for Iran/Middle East war-related items.
    Returns a list of strings (headline + source).
    """
    headlines = []
    seen = set()

    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue

        source = feed.feed.get("title", "Unknown Source")
        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            text = f"{title} - {summary}"
            if not headline_matches_war(text):
                continue

            key = title.strip()
            if key in seen:
                continue
            seen.add(key)

            headlines.append(f"{title} [{source}]")
            if len(headlines) >= max_items:
                return headlines

    return headlines


def fetch_war_news_api(max_items=50):
    """
    Fetch Iran/Middle East war-related headlines using GNews API.
    Returns: list of headline strings.
    NOTE: replace 'demo' with your own API key for real usage.
    """
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": "Iran OR Hormuz OR Middle East OR missile OR tanker OR strike",
        "lang": "en",
        "max": max_items,
        "token": "demo",  # <-- replace with your API key
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
    except Exception:
        return []

    headlines = []
    for item in data.get("articles", []):
        title = item.get("title", "")
        source = item.get("source", {}).get("name", "Unknown")
        if not title:
            continue
        if not headline_matches_war(title):
            continue
        headlines.append(f"{title} [{source}]")

    return headlines


def fetch_all_war_news():
    rss_news = fetch_war_news_rss(max_items=50)
    api_news = fetch_war_news_api(max_items=50)

    combined = rss_news + api_news
    # remove duplicates while preserving order
    combined = list(dict.fromkeys(combined))

    if not combined:
        return ["No war-related headlines found."]

    return combined[:80]


# ---------------- GUI COCKPIT ---------------- #

class HormuzCockpit(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Strait of Hormuz Impact Cockpit")
        self.geometry("1200x720")
        self.configure(bg="#101018")

        self.alert_active = False
        self.alert_flash_state = False

        self.news_headlines = []
        self.ticker_index = 0

        self._build_ui()
        self._schedule_market_refresh()
        self._schedule_news_refresh()
        self._schedule_ticker_step()

    def _build_ui(self):
        # Top frame: title + status
        top = tk.Frame(self, bg="#101018")
        top.pack(fill="x", pady=5)

        self.title_label = tk.Label(
            top,
            text="Strait of Hormuz Impact Monitor",
            fg="#00ffcc",
            bg="#101018",
            font=("Segoe UI", 18, "bold"),
        )
        self.title_label.pack(side="left", padx=10)

        self.status_label = tk.Label(
            top,
            text="Status: Initializing...",
            fg="#cccccc",
            bg="#101018",
            font=("Segoe UI", 10),
        )
        self.status_label.pack(side="right", padx=10)

        # Main area: left (markets + impact), right (news feed)
        main = tk.Frame(self, bg="#101018")
        main.pack(fill="both", expand=True, padx=10, pady=(5, 0))

        left = tk.Frame(main, bg="#101018")
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))

        right = tk.Frame(main, bg="#101018")
        right.pack(side="right", fill="both", expand=True, padx=(5, 0))

        # Market panel
        market_frame = tk.LabelFrame(
            left,
            text="Markets",
            fg="#00ffcc",
            bg="#101018",
            bd=2,
            font=("Segoe UI", 11, "bold"),
            labelanchor="n",
        )
        market_frame.pack(fill="x", pady=5)

        self.market_labels = {}
        for name in ["Brent", "BTC", "ETH", "Gold", "Silver", "VIX"]:
            row = tk.Frame(market_frame, bg="#101018")
            row.pack(fill="x", pady=2)
            lbl_name = tk.Label(
                row,
                text=f"{name}:",
                fg="#cccccc",
                bg="#101018",
                font=("Segoe UI", 10),
                width=10,
                anchor="w",
            )
            lbl_name.pack(side="left")
            lbl_val = tk.Label(
                row,
                text="--",
                fg="#ffffff",
                bg="#101018",
                font=("Segoe UI", 10, "bold"),
                anchor="w",
            )
            lbl_val.pack(side="left")
            self.market_labels[name] = lbl_val

        # Impact panel
        impact_frame = tk.LabelFrame(
            left,
            text="Strait of Hormuz Impact",
            fg="#ff6666",
            bg="#101018",
            bd=2,
            font=("Segoe UI", 11, "bold"),
            labelanchor="n",
        )
        impact_frame.pack(fill="both", expand=True, pady=5)

        self.stress_label = tk.Label(
            impact_frame,
            text="Stress Index: --",
            fg="#ff6666",
            bg="#101018",
            font=("Segoe UI", 14, "bold"),
        )
        self.stress_label.pack(pady=10)

        self.closure_label = tk.Label(
            impact_frame,
            text=f"Closure date (scenario): {CLOSURE_DATE}",
            fg="#cccccc",
            bg="#101018",
            font=("Segoe UI", 10),
        )
        self.closure_label.pack(pady=5)

        self.alert_label = tk.Label(
            impact_frame,
            text="ALERT: None",
            fg="#ffffff",
            bg="#101018",
            font=("Segoe UI", 12, "bold"),
        )
        self.alert_label.pack(pady=10)

        # Map panel
        map_frame = tk.LabelFrame(
            left,
            text="Tanker Traffic Map",
            fg="#ffcc00",
            bg="#101018",
            bd=2,
            font=("Segoe UI", 11, "bold"),
            labelanchor="n",
        )
        map_frame.pack(fill="x", pady=5)

        map_label = tk.Label(
            map_frame,
            text="Open live tanker map in browser (external site).",
            fg="#cccccc",
            bg="#101018",
            font=("Segoe UI", 10),
        )
        map_label.pack(pady=5)

        map_button = tk.Button(
            map_frame,
            text="Open Tanker Map",
            command=self.open_tanker_map,
            bg="#202040",
            fg="#00ffcc",
            font=("Segoe UI", 10, "bold"),
        )
        map_button.pack(pady=5)

        # Right side: vertical war news feed
        news_frame = tk.LabelFrame(
            right,
            text="War News Feed (Iran / Region)",
            fg="#00ffcc",
            bg="#101018",
            bd=2,
            font=("Segoe UI", 11, "bold"),
            labelanchor="n",
        )
        news_frame.pack(fill="both", expand=True, pady=5)

        self.news_list = tk.Listbox(
            news_frame,
            bg="#181828",
            fg="#ffffff",
            font=("Segoe UI", 10),
            selectbackground="#00ffcc",
            activestyle="none",
        )
        self.news_list.pack(fill="both", expand=True, padx=5, pady=5)

        # Bottom: horizontal ticker
        ticker_frame = tk.Frame(self, bg="#080810")
        ticker_frame.pack(fill="x", side="bottom")

        ticker_label_title = tk.Label(
            ticker_frame,
            text="IRAN WAR TICKER:",
            fg="#ffcc00",
            bg="#080810",
            font=("Segoe UI", 10, "bold"),
        )
        ticker_label_title.pack(side="left", padx=5)

        self.ticker_label = tk.Label(
            ticker_frame,
            text="Loading headlines...",
            fg="#ffffff",
            bg="#080810",
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.ticker_label.pack(side="left", fill="x", expand=True, padx=5)

    # ---------------- MAP ---------------- #

    def open_tanker_map(self):
        url = "https://www.marinetraffic.com/"
        webbrowser.open(url)

    # ---------------- SCHEDULING ---------------- #

    def _schedule_market_refresh(self):
        self.refresh_markets()
        # every 5 minutes
        self.after(300000, self._schedule_market_refresh)

    def _schedule_news_refresh(self):
        self.refresh_news()
        # every 3 minutes
        self.after(180000, self._schedule_news_refresh)

    def _schedule_ticker_step(self):
        self.update_ticker()
        # every 500 ms
        self.after(500, self._schedule_ticker_step)

    # ---------------- MARKET + IMPACT ---------------- #

    def refresh_markets(self):
        self.status_label.config(text="Status: Updating markets...")
        self.update_idletasks()

        prices = fetch_latest_prices()
        for name, lbl in self.market_labels.items():
            val = prices.get(name)
            if val is None:
                lbl.config(text="--")
            else:
                if name in ["BTC", "ETH"]:
                    lbl.config(text=f"{val:,.0f} USD")
                elif name in ["Gold", "Silver", "Brent"]:
                    lbl.config(text=f"{val:,.2f}")
                elif name == "VIX":
                    lbl.config(text=f"{val:,.2f}")
                else:
                    lbl.config(text=str(val))

        stress = compute_hormuz_stress(prices)
        self.stress_label.config(text=f"Stress Index: {stress:,.1f} / 100")

        if stress >= 60:
            self.activate_alert("HIGH STRESS: Markets reacting strongly.")
        elif stress >= 40:
            self.activate_alert("ELEVATED STRESS: Watch closely.")
        else:
            self.deactivate_alert()

        self.status_label.config(text="Status: Live")

    def activate_alert(self, message):
        if not self.alert_active:
            self.alert_active = True
            self.alert_label.config(text=f"ALERT: {message}")
            self._start_alert_flash()
            if HAS_SOUND:
                try:
                    winsound.Beep(1000, 500)
                except Exception:
                    pass
        else:
            self.alert_label.config(text=f"ALERT: {message}")

    def deactivate_alert(self):
        self.alert_active = False
        self.alert_label.config(
            text="ALERT: None",
            bg="#101018",
            fg="#ffffff",
        )

    def _start_alert_flash(self):
        if not self.alert_active:
            self.alert_label.config(bg="#101018", fg="#ffffff")
            return

        self.alert_flash_state = not self.alert_flash_state
        if self.alert_flash_state:
            self.alert_label.config(bg="#ff0000", fg="#ffffff")
        else:
            self.alert_label.config(bg="#101018", fg="#ff6666")

        self.after(500, self._start_alert_flash)

    # ---------------- NEWS (VERTICAL + TICKER) ---------------- #

    def refresh_news(self):
        self.status_label.config(text="Status: Updating war news...")
        self.update_idletasks()

        headlines = fetch_all_war_news()
        self.news_headlines = headlines if headlines else ["No war-related headlines found."]

        self.news_list.delete(0, tk.END)
        for h in self.news_headlines:
            self.news_list.insert(tk.END, h)

        self.ticker_index = 0
        self.status_label.config(text="Status: Live")

    def update_ticker(self):
        if not self.news_headlines:
            self.ticker_label.config(text="No headlines.")
            return

        headline = self.news_headlines[self.ticker_index % len(self.news_headlines)]
        self.ticker_label.config(text=headline)
        self.ticker_index += 1


# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    app = HormuzCockpit()
    app.mainloop()

