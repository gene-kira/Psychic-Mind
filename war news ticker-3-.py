import importlib
import subprocess
import sys
import pkgutil
import shutil
import threading
import queue
import datetime
import webbrowser
import statistics
import os
import json

# ---------------- SIMPLE LOGGER ---------------- #

LOG_FILE = "cockpit.log"

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------- AUTOLOADER ---------------- #

class AutoLoader:
    def __init__(
        self,
        required_libs,
        versions=None,
        silent=False,
        log_file="autoloader.log",
    ):
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
        # simple check: ping presence only; real connectivity checked later
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
        "matplotlib": "matplotlib",
        "vaderSentiment": "vaderSentiment",
        "tkinter": "tk",
    },
    versions={},
    silent=False,
)

libs = loader.load()

pd = libs.get("pandas")
yf = libs.get("yfinance")
requests = libs.get("requests")
feedparser = libs.get("feedparser")
matplotlib = libs.get("matplotlib")
vaderSentiment = libs.get("vaderSentiment")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    print("Tkinter not available. Install it via your OS package manager.")
    sys.exit(1)

try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------- DATA + CONSTANTS ---------------- #

CLOSURE_DATE = datetime.date(2026, 2, 28)

TICKERS = {
    "Brent": "BZ=F",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "VIX": "^VIX",
}

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

analyzer = SentimentIntensityAnalyzer()

OFFLINE_MODE = False
CACHE_MARKETS_FILE = "market_cache.json"
CACHE_NEWS_FILE = "news_cache.json"


# ---------------- CONNECTIVITY CHECK ---------------- #

def check_internet():
    global OFFLINE_MODE
    try:
        requests.get("https://www.google.com", timeout=3)
        OFFLINE_MODE = False
        log("[NET] Online")
    except Exception:
        OFFLINE_MODE = True
        log("[NET] Offline detected")


# ---------------- MARKET DATA + CACHE + RETRY ---------------- #

def fetch_price_history(days=30, retries=3):
    if OFFLINE_MODE:
        log("[MARKET] Offline mode: cannot fetch history.")
        return pd.DataFrame()

    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)

    for attempt in range(1, retries + 1):
        try:
            log(f"[MARKET] Fetching history (attempt {attempt}/{retries})")
            data = yf.download(
                list(TICKERS.values()),
                start=start.isoformat(),
                end=(end + datetime.timedelta(days=1)).isoformat(),
                progress=False,
            )["Adj Close"]
            if isinstance(data, pd.Series):
                data = data.to_frame()
            if not data.empty:
                log("[MARKET] History fetch OK")
                return data
        except Exception as e:
            log(f"[MARKET] Error fetching market data: {e}")
    log("[MARKET] All retries failed for history.")
    return pd.DataFrame()


def save_market_cache(prices, hist):
    try:
        cache = {
            "prices": prices,
            "hist": hist.to_dict() if isinstance(hist, pd.DataFrame) else {},
        }
        with open(CACHE_MARKETS_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        log("[CACHE] Market cache saved.")
    except Exception as e:
        log(f"[CACHE] Error saving market cache: {e}")


def load_market_cache():
    if not os.path.exists(CACHE_MARKETS_FILE):
        return {}, pd.DataFrame()
    try:
        with open(CACHE_MARKETS_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        prices = cache.get("prices", {})
        hist_dict = cache.get("hist", {})
        hist = pd.DataFrame(hist_dict)
        log("[CACHE] Market cache loaded.")
        return prices, hist
    except Exception as e:
        log(f"[CACHE] Error loading market cache: {e}")
        return {}, pd.DataFrame()


def fetch_latest_prices():
    """
    Always returns (prices_dict, hist_df).
    Uses retries, offline mode, and cache fallback.
    """
    check_internet()

    if OFFLINE_MODE:
        log("[MARKET] Offline mode: using cached market data if available.")
        prices, hist = load_market_cache()
        return prices, hist

    hist = fetch_price_history(days=30, retries=3)

    if hist is None or hist.empty:
        log("[MARKET] History empty, falling back to cache.")
        prices, hist = load_market_cache()
        return prices, hist

    latest = {}
    for name, ticker in TICKERS.items():
        if ticker in hist.columns:
            series = hist[ticker].dropna()
            latest[name] = float(series.iloc[-1]) if not series.empty else None
        else:
            latest[name] = None

    save_market_cache(latest, hist)
    return latest, hist


def compute_hormuz_stress(prices):
    brent = prices.get("Brent")
    vix = prices.get("VIX")
    gold = prices.get("Gold")
    silver = prices.get("Silver")
    btc = prices.get("BTC")
    eth = prices.get("ETH")

    if None in [brent, vix, gold, silver, btc, eth]:
        return 0.0

    oil_component = max(0, min((brent - 80) * 0.6, 25))
    vix_component = max(0, min((vix - 15) * 1.5, 25))
    gold_component = max(0, min((gold - 1900) * 0.02, 15))
    silver_component = max(0, min((silver - 24) * 0.5, 10))
    btc_component = max(0, min((35000 - btc) / 800, 15))
    eth_component = max(0, min((2500 - eth) / 150, 10))

    panic_assets = 0
    if brent > 100:
        panic_assets += 1
    if vix > 25:
        panic_assets += 1
    if gold > 2000:
        panic_assets += 1
    if btc < 30000:
        panic_assets += 1

    correlation_component = panic_assets * 3

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


# ---------------- WAR NEWS + SENTIMENT + CACHE + RETRY ---------------- #

def headline_matches_war(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(k in lower for k in WAR_KEYWORDS)


def fetch_war_news_rss(max_items=50, retries=2):
    headlines = []
    seen = set()

    if OFFLINE_MODE:
        log("[NEWS] Offline mode: RSS not available.")
        return []

    for url in RSS_FEEDS:
        for attempt in range(1, retries + 1):
            try:
                log(f"[NEWS] Fetching RSS {url} (attempt {attempt}/{retries})")
                feed = feedparser.parse(url)
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
                    headlines.append((title, source))
                    if len(headlines) >= max_items:
                        return headlines
                break
            except Exception as e:
                log(f"[NEWS] RSS error: {e}")
                continue

    return headlines


def fetch_war_news_api(max_items=50, retries=2):
    if OFFLINE_MODE:
        log("[NEWS] Offline mode: API not available.")
        return []

    url = "https://gnews.io/api/v4/search"
    params = {
        "q": "Iran OR Hormuz OR Middle East OR missile OR tanker OR strike",
        "lang": "en",
        "max": max_items,
        "token": "demo",  # replace with your key
    }

    for attempt in range(1, retries + 1):
        try:
            log(f"[NEWS] Fetching GNews (attempt {attempt}/{retries})")
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            headlines = []
            for item in data.get("articles", []):
                title = item.get("title", "")
                source = item.get("source", {}).get("name", "Unknown")
                if not title:
                    continue
                if not headline_matches_war(title):
                    continue
                headlines.append((title, source))
            return headlines
        except Exception as e:
            log(f"[NEWS] API error: {e}")
            continue

    return []


def save_news_cache(headlines):
    try:
        cache = [{"title": t, "source": s} for t, s in headlines]
        with open(CACHE_NEWS_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        log("[CACHE] News cache saved.")
    except Exception as e:
        log(f"[CACHE] Error saving news cache: {e}")


def load_news_cache():
    if not os.path.exists(CACHE_NEWS_FILE):
        return []
    try:
        with open(CACHE_NEWS_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        headlines = [(item["title"], item["source"]) for item in cache]
        log("[CACHE] News cache loaded.")
        return headlines
    except Exception as e:
        log(f"[CACHE] Error loading news cache: {e}")
        return []


def fetch_all_war_news():
    check_internet()

    if OFFLINE_MODE:
        log("[NEWS] Offline mode: using cached news if available.")
        cached = load_news_cache()
        if cached:
            return cached
        return [("No war-related headlines (offline, no cache).", "System")]

    rss_news = fetch_war_news_rss(max_items=50)
    api_news = fetch_war_news_api(max_items=50)

    combined = rss_news + api_news
    seen = set()
    deduped = []
    for title, source in combined:
        if title not in seen:
            seen.add(title)
            deduped.append((title, source))

    if not deduped:
        log("[NEWS] No headlines from live sources, falling back to cache.")
        cached = load_news_cache()
        if cached:
            return cached
        return [("No war-related headlines found.", "System")]

    save_news_cache(deduped)
    return deduped[:80]


def analyze_sentiment(headlines):
    scores = []
    for title, _ in headlines:
        vs = analyzer.polarity_scores(title)
        scores.append(vs["compound"])
    if not scores:
        return 0.0, "Neutral"

    avg = statistics.mean(scores)
    if avg > 0.2:
        label = "Positive / De-escalatory"
    elif avg < -0.2:
        label = "Negative / Escalatory"
    else:
        label = "Neutral / Mixed"
    return avg, label


def summarize_headlines(headlines, max_len=3):
    if not headlines:
        return ["No headlines to summarize."]

    texts = [t for t, _ in headlines]
    all_words = []
    for t in texts:
        for w in t.lower().split():
            if len(w) > 4:
                all_words.append(w.strip(".,!?;:()[]\"'"))

    if not all_words:
        return texts[:max_len]

    freq = {}
    for w in all_words:
        freq[w] = freq.get(w, 0) + 1

    key_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
    key_terms = [k for k, _ in key_terms]

    scored = []
    for t, _ in headlines:
        score = sum(1 for k in key_terms if k in t.lower())
        scored.append((score, t))

    scored.sort(reverse=True, key=lambda x: x[0])
    summary = [t for s, t in scored[:max_len] if s > 0]
    if not summary:
        summary = [t for t, _ in headlines[:max_len]]

    return summary


def compute_risk_forecast(stress, sentiment_score, headline_count):
    base = stress
    if sentiment_score < -0.2:
        base += 10
    elif sentiment_score > 0.2:
        base -= 5

    if headline_count > 40:
        base += 5
    elif headline_count < 10:
        base -= 5

    base = max(0.0, min(base, 100.0))

    if base >= 70:
        label = "High Escalation Risk"
    elif base >= 45:
        label = "Elevated Risk"
    elif base >= 25:
        label = "Watchful"
    else:
        label = "Low / Stable"

    return base, label


# ---------------- GUI COCKPIT ---------------- #

class HormuzCockpit(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Strait of Hormuz Impact Cockpit")
        self.geometry("1400x800")
        self.configure(bg="#101018")

        self.alert_active = False
        self.alert_flash_state = False

        self.news_headlines = []
        self.ticker_index = 0

        self.stress_history = []
        self.time_history = []
        self.alert_log = []

        self.second_screen = None

        self._build_ui()
        self._schedule_market_refresh()
        self._schedule_news_refresh()
        self._schedule_ticker_step()

    def _build_ui(self):
        # Error banner
        self.error_banner = tk.Label(
            self,
            text="",
            fg="#ffffff",
            bg="#aa0000",
            font=("Segoe UI", 9, "bold"),
        )
        self.error_banner.pack(fill="x")
        self.hide_error_banner()

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

        main = tk.Frame(self, bg="#101018")
        main.pack(fill="both", expand=True, padx=10, pady=(5, 0))

        left = tk.Frame(main, bg="#101018")
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))

        right = tk.Frame(main, bg="#101018")
        right.pack(side="right", fill="both", expand=True, padx=(5, 0))

        # LEFT: markets + impact + charts + timeline
        upper_left = tk.Frame(left, bg="#101018")
        upper_left.pack(fill="x")

        market_frame = tk.LabelFrame(
            upper_left,
            text="Markets",
            fg="#00ffcc",
            bg="#101018",
            bd=2,
            font=("Segoe UI", 11, "bold"),
            labelanchor="n",
        )
        market_frame.pack(side="left", fill="both", expand=True, pady=5, padx=(0, 5))

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

        impact_frame = tk.LabelFrame(
            upper_left,
            text="Strait of Hormuz Impact",
            fg="#ff6666",
            bg="#101018",
            bd=2,
            font=("Segoe UI", 11, "bold"),
            labelanchor="n",
        )
        impact_frame.pack(side="right", fill="both", expand=True, pady=5, padx=(5, 0))

        self.stress_label = tk.Label(
            impact_frame,
            text="Stress Index: --",
            fg="#ff6666",
            bg="#101018",
            font=("Segoe UI", 14, "bold"),
        )
        self.stress_label.pack(pady=5)

        self.closure_label = tk.Label(
            impact_frame,
            text=f"Closure date (scenario): {CLOSURE_DATE}",
            fg="#cccccc",
            bg="#101018",
            font=("Segoe UI", 10),
        )
        self.closure_label.pack(pady=2)

        self.sentiment_label = tk.Label(
            impact_frame,
            text="Sentiment: --",
            fg="#cccccc",
            bg="#101018",
            font=("Segoe UI", 10),
        )
        self.sentiment_label.pack(pady=2)

        self.risk_label = tk.Label(
            impact_frame,
            text="Risk Forecast: --",
            fg="#ffcc00",
            bg="#101018",
            font=("Segoe UI", 11, "bold"),
        )
        self.risk_label.pack(pady=5)

        self.alert_label = tk.Label(
            impact_frame,
            text="ALERT: None",
            fg="#ffffff",
            bg="#101018",
            font=("Segoe UI", 12, "bold"),
        )
        self.alert_label.pack(pady=5)

        second_screen_btn = tk.Button(
            impact_frame,
            text="Open Second Screen",
            command=self.open_second_screen,
            bg="#202040",
            fg="#00ffcc",
            font=("Segoe UI", 9, "bold"),
        )
        second_screen_btn.pack(pady=5)

        # Charts panel
        charts_frame = tk.LabelFrame(
            left,
            text="Charts",
            fg="#00ffcc",
            bg="#101018",
            bd=2,
            font=("Segoe UI", 11, "bold"),
            labelanchor="n",
        )
        charts_frame.pack(fill="both", expand=True, pady=5)

        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax_stress = self.fig.add_subplot(2, 1, 1)
        self.ax_price = self.fig.add_subplot(2, 1, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=charts_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Timeline + alert log
        bottom_left = tk.Frame(left, bg="#101018")
        bottom_left.pack(fill="both", expand=True, pady=5)

        timeline_frame = tk.LabelFrame(
            bottom_left,
            text="Timeline",
            fg="#00ffcc",
            bg="#101018",
            bd=2,
            font=("Segoe UI", 11, "bold"),
            labelanchor="n",
        )
        timeline_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.timeline_list = tk.Listbox(
            timeline_frame,
            bg="#181828",
            fg="#ffffff",
            font=("Segoe UI", 9),
            selectbackground="#00ffcc",
            activestyle="none",
        )
        self.timeline_list.pack(fill="both", expand=True, padx=5, pady=5)

        alertlog_frame = tk.LabelFrame(
            bottom_left,
            text="Alert Log",
            fg="#ff6666",
            bg="#101018",
            bd=2,
            font=("Segoe UI", 11, "bold"),
            labelanchor="n",
        )
        alertlog_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))

        self.alert_list = tk.Listbox(
            alertlog_frame,
            bg="#181828",
            fg="#ffffff",
            font=("Segoe UI", 9),
            selectbackground="#ff6666",
            activestyle="none",
        )
        self.alert_list.pack(fill="both", expand=True, padx=5, pady=5)

        # RIGHT: news feed + summaries
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

        summary_frame = tk.LabelFrame(
            right,
            text="AI-style Summary & Forecast",
            fg="#ffcc00",
            bg="#101018",
            bd=2,
            font=("Segoe UI", 11, "bold"),
            labelanchor="n",
        )
        summary_frame.pack(fill="x", pady=5)

        self.summary_text = tk.Text(
            summary_frame,
            bg="#181828",
            fg="#ffffff",
            font=("Segoe UI", 9),
            height=6,
            wrap="word",
        )
        self.summary_text.pack(fill="both", expand=True, padx=5, pady=5)

        map_frame = tk.LabelFrame(
            right,
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

        # Bottom ticker
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

    # ---------------- ERROR BANNER ---------------- #

    def show_error_banner(self, message):
        self.error_banner.config(text=message)
        self.error_banner.pack(fill="x")
        log(f"[UI] ERROR BANNER: {message}")

    def hide_error_banner(self):
        self.error_banner.config(text="")
        self.error_banner.pack_forget()

    # ---------------- MAP ---------------- #

    def open_tanker_map(self):
        url = "https://www.marinetraffic.com/"
        webbrowser.open(url)

    # ---------------- SECOND SCREEN ---------------- #

    def open_second_screen(self):
        if self.second_screen and tk.Toplevel.winfo_exists(self.second_screen):
            self.second_screen.lift()
            return

        self.second_screen = tk.Toplevel(self)
        self.second_screen.title("Hormuz Second Screen")
        self.second_screen.geometry("800x200")
        self.second_screen.configure(bg="#101018")

        stress_label = tk.Label(
            self.second_screen,
            text=self.stress_label.cget("text"),
            fg="#ff6666",
            bg="#101018",
            font=("Segoe UI", 14, "bold"),
        )
        stress_label.pack(pady=5)

        alert_label = tk.Label(
            self.second_screen,
            text=self.alert_label.cget("text"),
            fg="#ffffff",
            bg="#101018",
            font=("Segoe UI", 12, "bold"),
        )
        alert_label.pack(pady=5)

        ticker_clone = tk.Label(
            self.second_screen,
            text=self.ticker_label.cget("text"),
            fg="#ffffff",
            bg="#080810",
            font=("Segoe UI", 10),
            anchor="w",
        )
        ticker_clone.pack(fill="x", padx=5, pady=5)

        def sync_second_screen():
            if not tk.Toplevel.winfo_exists(self.second_screen):
                return
            stress_label.config(text=self.stress_label.cget("text"))
            alert_label.config(text=self.alert_label.cget("text"))
            ticker_clone.config(text=self.ticker_label.cget("text"))
            self.after(1000, sync_second_screen)

        sync_second_screen()

    # ---------------- SCHEDULING ---------------- #

    def _schedule_market_refresh(self):
        self.refresh_markets()
        self.after(300000, self._schedule_market_refresh)

    def _schedule_news_refresh(self):
        self.refresh_news()
        self.after(180000, self._schedule_news_refresh)

    def _schedule_ticker_step(self):
        self.update_ticker()
        self.after(500, self._schedule_ticker_step)

    # ---------------- TIMELINE + ALERT LOG ---------------- #

    def log_timeline(self, message):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        self.timeline_list.insert(tk.END, entry)
        self.timeline_list.see(tk.END)
        log(f"[TIMELINE] {entry}")

    def log_alert(self, message, stress):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{ts} | Stress={stress:.1f} | {message}"
        self.alert_list.insert(tk.END, entry)
        self.alert_list.see(tk.END)
        self.alert_log.append(entry)
        log(f"[ALERT] {entry}")

    # ---------------- MARKET + IMPACT ---------------- #

    def refresh_markets(self):
        self.status_label.config(text="Status: Updating markets...")
        self.update_idletasks()
        self.hide_error_banner()

        try:
            prices, hist = fetch_latest_prices()
        except Exception as e:
            self.status_label.config(text="Status: Market fetch error")
            self.show_error_banner(f"Market error: {e}")
            log(f"[ERROR] refresh_markets exception: {e}")
            return

        if not prices:
            self.status_label.config(text="Status: Market data unavailable")
            self.show_error_banner("Market data unavailable (offline or no cache).")
            return

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

        now = datetime.datetime.now()
        self.stress_history.append(stress)
        self.time_history.append(now)

        if len(self.stress_history) > 200:
            self.stress_history = self.stress_history[-200:]
            self.time_history = self.time_history[-200:]

        if stress >= 60:
            self.activate_alert("HIGH STRESS: Markets reacting strongly.", stress)
        elif stress >= 40:
            self.activate_alert("ELEVATED STRESS: Watch closely.", stress)
        else:
            self.deactivate_alert()

        self.update_charts(hist)
        self.status_label.config(text=f"Status: Live ({'OFFLINE' if OFFLINE_MODE else 'ONLINE'})")
        self.log_timeline("Markets + stress updated.")

    def activate_alert(self, message, stress):
        if not self.alert_active:
            self.alert_active = True
            self.alert_label.config(text=f"ALERT: {message}")
            self._start_alert_flash()
            self.log_alert(message, stress)
            if HAS_SOUND:
                try:
                    winsound.Beep(1000, 500)
                except Exception:
                    pass
        else:
            self.alert_label.config(text=f"ALERT: {message}")

    def deactivate_alert(self):
        if self.alert_active:
            self.log_alert("Alert cleared", self.stress_history[-1] if self.stress_history else 0)
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

    # ---------------- CHARTS ---------------- #

    def update_charts(self, hist):
        self.ax_stress.clear()
        self.ax_price.clear()

        if self.time_history and self.stress_history:
            times = [t.strftime("%H:%M") for t in self.time_history]
            self.ax_stress.plot(times, self.stress_history, color="#ff6666", label="Stress")
            self.ax_stress.set_title("Stress Index Over Time", color="#ffffff")
            self.ax_stress.tick_params(axis="x", rotation=45, labelsize=6)
            self.ax_stress.set_ylim(0, 100)
            self.ax_stress.grid(True, alpha=0.2)
            self.ax_stress.legend(facecolor="#101018", edgecolor="#ffffff")
            self.ax_stress.set_facecolor("#181828")

        if isinstance(hist, pd.DataFrame) and not hist.empty:
            for name, ticker, color in [
                ("Brent", "BZ=F", "#ffcc00"),
                ("BTC", "BTC-USD", "#00ffcc"),
                ("Gold", "GC=F", "#ffffff"),
            ]:
                if ticker in hist.columns:
                    series = hist[ticker].dropna()
                    if not series.empty:
                        self.ax_price.plot(series.index, series.values, label=name, color=color)

            self.ax_price.set_title("Key Prices (Last 30 Days)", color="#ffffff")
            self.ax_price.tick_params(axis="x", rotation=45, labelsize=6)
            self.ax_price.grid(True, alpha=0.2)
            self.ax_price.legend(facecolor="#101018", edgecolor="#ffffff")
            self.ax_price.set_facecolor("#181828")

        self.fig.tight_layout()
        self.fig.patch.set_facecolor("#101018")
        self.canvas.draw()

    # ---------------- NEWS + SUMMARY + RISK ---------------- #

    def refresh_news(self):
        self.status_label.config(text="Status: Updating war news...")
        self.update_idletasks()
        self.hide_error_banner()

        try:
            headlines = fetch_all_war_news()
        except Exception as e:
            self.status_label.config(text="Status: News fetch error")
            self.show_error_banner(f"News error: {e}")
            log(f"[ERROR] refresh_news exception: {e}")
            return

        self.news_headlines = headlines

        self.news_list.delete(0, tk.END)
        for title, source in self.news_headlines:
            self.news_list.insert(tk.END, f"{title} [{source}]")

        sentiment_score, sentiment_label = analyze_sentiment(self.news_headlines)
        self.sentiment_label.config(
            text=f"Sentiment: {sentiment_label} (score={sentiment_score:.2f})"
        )

        stress = self.stress_history[-1] if self.stress_history else 0
        risk_score, risk_label = compute_risk_forecast(
            stress, sentiment_score, len(self.news_headlines)
        )
        self.risk_label.config(text=f"Risk Forecast: {risk_label} ({risk_score:.1f}/100)")

        summary_lines = summarize_headlines(self.news_headlines, max_len=3)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, "Summary of current situation:\n\n")
        for line in summary_lines:
            self.summary_text.insert(tk.END, f"• {line}\n")

        self.log_timeline("News + sentiment + risk updated.")
        self.ticker_index = 0
        self.status_label.config(text=f"Status: Live ({'OFFLINE' if OFFLINE_MODE else 'ONLINE'})")

    def update_ticker(self):
        if not self.news_headlines:
            self.ticker_label.config(text="No headlines.")
            return

        title, source = self.news_headlines[self.ticker_index % len(self.news_headlines)]
        self.ticker_label.config(text=f"{title} [{source}]")
        self.ticker_index += 1


# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    log("=== Starting Hormuz Cockpit ===")
    app = HormuzCockpit()
    app.mainloop()

