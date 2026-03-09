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
import time
import math

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


class AutoLoader:
    def __init__(self, required_libs, versions=None, silent=False, log_file="autoloader.log"):
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
                    f.write(f"[timestamp={timestamp}] {msg}\n")
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


loader = AutoLoader(
    required_libs={
        "pandas": "pandas",
        "yfinance": "yfinance",
        "requests": "requests",
        "feedparser": "feedparser",
        "matplotlib": "matplotlib",
        "vaderSentiment": "vaderSentiment",
        "tkinter": "tk",
        "uiautomation": "uiautomation",
        "sklearn": "scikit-learn",
        "pyttsx3": "pyttsx3",
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
uiautomation = libs.get("uiautomation")
sklearn = libs.get("sklearn")
pyttsx3 = libs.get("pyttsx3")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    print("Tkinter not available.")
    sys.exit(1)

try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.neural_network import MLPRegressor

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

REGION_PROFILES = {
    "Iran / Hormuz": {
        "keywords": [
            "iran", "tehran", "hormuz", "strait of hormuz",
            "missile", "strike", "attack", "gulf", "middle east",
            "israel", "us ", "u.s.", "tanker", "naval", "drone"
        ],
        "query": "Iran OR Hormuz OR Middle East OR missile OR tanker OR strike",
        "map_url": "https://www.marinetraffic.com/en/ais/home/centerx:56.5/centery:26.5/zoom:7",
    },
    "Taiwan": {
        "keywords": [
            "taiwan", "taipei", "strait", "pla", "chinese jets",
            "taiwan strait", "beijing", "naval drills"
        ],
        "query": "Taiwan OR Taipei OR PLA OR 'Taiwan Strait' OR Beijing",
        "map_url": "https://www.marinetraffic.com/en/ais/home/centerx:122.5/centery:23.5/zoom:7",
    },
    "Red Sea": {
        "keywords": [
            "red sea", "houthi", "yemen", "bab el-mandeb",
            "shipping lane", "tanker", "missile", "drone"
        ],
        "query": "\"Red Sea\" OR Houthi OR Yemen OR tanker OR shipping",
        "map_url": "https://www.marinetraffic.com/en/ais/home/centerx:42.5/centery:16.0/zoom:7",
    },
    "Ukraine": {
        "keywords": [
            "ukraine", "kyiv", "kiev", "donbas", "crimea",
            "russia", "missile", "strike", "frontline"
        ],
        "query": "Ukraine OR Kyiv OR Donbas OR Crimea OR missile OR strike",
        "map_url": "https://www.marinetraffic.com/en/ais/home/centerx:32.0/centery:45.0/zoom:6",
    },
}

CURRENT_REGION = "Iran / Hormuz"

analyzer = SentimentIntensityAnalyzer()

OFFLINE_MODE = False
CACHE_MARKETS_FILE = "market_cache.json"
CACHE_NEWS_FILE = "news_cache.json"

DEFAULT_ELEVATED = 40.0
DEFAULT_HIGH = 60.0


def check_internet():
    global OFFLINE_MODE
    try:
        requests.get("https://www.google.com", timeout=3)
        OFFLINE_MODE = False
        log("[NET] Online")
    except Exception:
        OFFLINE_MODE = True
        log("[NET] Offline detected")


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


def predict_future_stress_linear(stress_history, steps_ahead=6):
    if len(stress_history) < 3:
        return stress_history[-1] if stress_history else 0.0
    n = min(10, len(stress_history))
    y = stress_history[-n:]
    x = list(range(n))
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den = sum((xi - mean_x) ** 2 for xi in x) or 1.0
    slope = num / den
    intercept = mean_y - slope * mean_x
    future_x = n - 1 + steps_ahead
    pred = slope * future_x + intercept
    return max(0.0, min(pred, 100.0))


def headline_matches_region(text: str, region: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    profile = REGION_PROFILES.get(region, REGION_PROFILES["Iran / Hormuz"])
    for k in profile["keywords"]:
        if k.lower() in lower:
            return True
    return False


def fetch_war_news_rss(region, max_items=50, retries=2):
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
                    if not headline_matches_region(text, region):
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


def fetch_war_news_api(region, max_items=50, retries=2):
    if OFFLINE_MODE:
        log("[NEWS] Offline mode: API not available.")
        return []
    profile = REGION_PROFILES.get(region, REGION_PROFILES["Iran / Hormuz"])
    query = profile["query"]
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": "en",
        "max": max_items,
        "token": "demo",  # replace with your key
    }
    for attempt in range(1, retries + 1):
        try:
            log(f"[NEWS] Fetching GNews (attempt {attempt}/{retries}) for region={region}")
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            headlines = []
            for item in data.get("articles", []):
                title = item.get("title", "")
                source = item.get("source", {}).get("name", "Unknown")
                if not title:
                    continue
                if not headline_matches_region(title, region):
                    continue
                headlines.append((title, source))
            return headlines
        except Exception as e:
            log(f"[NEWS] API error: {e}")
            continue
    return []


def save_news_cache(headlines, region):
    try:
        cache = {
            "region": region,
            "items": [{"title": t, "source": s} for t, s in headlines],
        }
        with open(CACHE_NEWS_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        log(f"[CACHE] News cache saved for region={region}.")
    except Exception as e:
        log(f"[CACHE] Error saving news cache: {e}")


def load_news_cache(region):
    if not os.path.exists(CACHE_NEWS_FILE):
        return []
    try:
        with open(CACHE_NEWS_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("region") != region:
            return []
        headlines = [(item["title"], item["source"]) for item in cache.get("items", [])]
        log(f"[CACHE] News cache loaded for region={region}.")
        return headlines
    except Exception as e:
        log(f"[CACHE] Error loading news cache: {e}")
        return []


def fetch_all_war_news(region):
    check_internet()
    if OFFLINE_MODE:
        log("[NEWS] Offline mode: using cached news if available.")
        cached = load_news_cache(region)
        if cached:
            return cached
        return [(f"No war-related headlines (offline, no cache) for {region}.", "System")]
    rss_news = fetch_war_news_rss(region, max_items=50)
    api_news = fetch_war_news_api(region, max_items=50)
    combined = rss_news + api_news
    seen = set()
    deduped = []
    for title, source in combined:
        if title not in seen:
            seen.add(title)
            deduped.append((title, source))
    if not deduped:
        log("[NEWS] No headlines from live sources, falling back to cache.")
        cached = load_news_cache(region)
        if cached:
            return cached
        return [(f"No war-related headlines found for {region}.", "System")]
    save_news_cache(deduped, region)
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


def generate_risk_explanation(stress, sentiment_score, headline_count, risk_label, region):
    parts = []
    parts.append(f"Region: {region}.")
    parts.append(f"Current stress index is {stress:.1f}/100.")
    if sentiment_score < -0.2:
        parts.append("Headline sentiment is negative, indicating escalation or fear.")
    elif sentiment_score > 0.2:
        parts.append("Headline sentiment is relatively positive or de-escalatory.")
    else:
        parts.append("Headline sentiment is mixed or neutral.")
    if headline_count > 40:
        parts.append("News volume is high, suggesting intense coverage and heightened attention.")
    elif headline_count < 10:
        parts.append("News volume is low, suggesting limited coverage or a quieter phase.")
    else:
        parts.append("News volume is moderate.")
    parts.append(f"Overall, this maps to a risk state classified as: {risk_label}.")
    return " ".join(parts)


def generate_scenario_forecast(stress, predicted_stress, prob, sentiment_score, region):
    lines = []
    lines.append(f"Scenario forecast for {region}:")
    if prob >= 0.75:
        lines.append("• High likelihood of escalation in the near term.")
    elif prob >= 0.5:
        lines.append("• Conditions suggest a meaningful chance of escalation.")
    elif prob >= 0.25:
        lines.append("• Escalation risk exists but remains limited.")
    else:
        lines.append("• Escalation appears unlikely in the immediate term.")
    if predicted_stress > stress + 10:
        lines.append("• Stress is projected to rise from current levels.")
    elif predicted_stress < stress - 10:
        lines.append("• Stress is projected to ease from current levels.")
    else:
        lines.append("• Stress is projected to remain broadly similar.")
    if sentiment_score < -0.2:
        lines.append("• Headlines are skewed negative, reinforcing escalation risk.")
    elif sentiment_score > 0.2:
        lines.append("• Headlines lean positive or de-escalatory, tempering risk.")
    else:
        lines.append("• Headlines are mixed, leaving the trajectory uncertain.")
    lines.append("• Operators should monitor for sudden shifts in news volume or market dislocations.")
    return "\n".join(lines)


class UIAutomationOrgan(threading.Thread):
    def __init__(self, poll_interval=2.0):
        super().__init__(daemon=True)
        self.poll_interval = poll_interval
        self.active_app = None
        self.active_title = ""
        self.intent = "idle"
        self.intent_confidence = 0.0
        self._stop_flag = threading.Event()

    def stop(self):
        self._stop_flag.set()

    def run(self):
        if uiautomation is None:
            log("[UIAUTO] uiautomation not available.")
            return
        log("[UIAUTO] UIAutomation organ started.")
        while not self._stop_flag.is_set():
            try:
                ctrl = uiautomation.GetForegroundControl()
                if ctrl:
                    name = ctrl.Name or ""
                    process_name = ctrl.ProcessName or ""
                    title = name.strip()
                    self.active_app = process_name.lower()
                    self.active_title = title
                    self._infer_intent()
            except Exception as e:
                log(f"[UIAUTO] Error: {e}")
            time.sleep(self.poll_interval)

    def _infer_intent(self):
        title = (self.active_title or "").lower()
        app = (self.active_app or "").lower()
        intent = "idle"
        conf = 0.0
        if any(k in title for k in ["tradingview", "chart", "brent", "btc", "eth", "gold", "vix"]):
            intent = "markets"
            conf = 0.8
        elif any(k in app for k in ["tradingview", "thinkorswim", "tws"]):
            intent = "markets"
            conf = 0.7
        if any(k in title for k in ["reuters", "bbc", "al jazeera", "cnn", "news", "breaking"]):
            intent = "news"
            conf = max(conf, 0.8)
        elif any(k in app for k in ["chrome", "msedge", "firefox"]):
            if "news" in title or "live" in title:
                intent = "news"
                conf = max(conf, 0.6)
        if any(k in title for k in ["marinetraffic", "worldview", "sentinel", "map", "satellite"]):
            intent = "maps"
            conf = max(conf, 0.8)
        if conf == 0.0:
            intent = "other"
            conf = 0.3
        self.intent = intent
        self.intent_confidence = conf

    def snapshot(self):
        return {
            "active_app": self.active_app,
            "active_title": self.active_title,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
        }


class AnomalyOrgan:
    def __init__(self, window=30):
        self.window = window
        self.stress_window = []
        self.sentiment_window = []
        self.headline_window = []
        self.last_anomaly = None

    def update(self, stress, sentiment, headlines_count):
        self.stress_window.append(stress)
        self.sentiment_window.append(sentiment)
        self.headline_window.append(headlines_count)
        if len(self.stress_window) > self.window:
            self.stress_window = self.stress_window[-self.window:]
            self.sentiment_window = self.sentiment_window[-self.window:]
            self.headline_window = self.headline_window[-self.window:]
        anomaly_msgs = []
        if len(self.stress_window) >= 10:
            mean_s = statistics.mean(self.stress_window)
            std_s = statistics.pstdev(self.stress_window) or 1.0
            if stress > mean_s + 2 * std_s:
                anomaly_msgs.append("Stress spike anomaly detected.")
        if len(self.headline_window) >= 10:
            mean_h = statistics.mean(self.headline_window)
            std_h = statistics.pstdev(self.headline_window) or 1.0
            if headlines_count > mean_h + 2 * std_h:
                anomaly_msgs.append("News volume surge anomaly detected.")
        if len(self.sentiment_window) >= 10:
            mean_sent = statistics.mean(self.sentiment_window)
            std_sent = statistics.pstdev(self.sentiment_window) or 1.0
            if abs(sentiment - mean_sent) > 2 * std_sent:
                anomaly_msgs.append("Sentiment shock anomaly detected.")
        if anomaly_msgs:
            self.last_anomaly = " ".join(anomaly_msgs)
        else:
            self.last_anomaly = None
        return self.last_anomaly


class CorrelationEngine:
    def __init__(self):
        self.region_stress_history = {r: [] for r in REGION_PROFILES.keys()}
        self.region_news_history = {r: [] for r in REGION_PROFILES.keys()}

    def update_region(self, region, stress, news_count):
        self.region_stress_history[region].append(stress)
        self.region_news_history[region].append(news_count)
        if len(self.region_stress_history[region]) > 200:
            self.region_stress_history[region] = self.region_stress_history[region][-200:]
            self.region_news_history[region] = self.region_news_history[region][-200:]

    def compute_cross_region_correlations(self):
        regions = list(REGION_PROFILES.keys())
        corr_map = {}
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                r1, r2 = regions[i], regions[j]
                s1 = self.region_stress_history[r1]
                s2 = self.region_stress_history[r2]
                if len(s1) >= 10 and len(s2) >= 10:
                    n = min(len(s1), len(s2))
                    a = s1[-n:]
                    b = s2[-n:]
                    corr = self._pearson(a, b)
                    corr_map[(r1, r2)] = corr
        return corr_map

    def _pearson(self, a, b):
        n = len(a)
        if n == 0:
            return 0.0
        mean_a = sum(a) / n
        mean_b = sum(b) / n
        num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
        den_a = math.sqrt(sum((x - mean_a) ** 2 for x in a)) or 1.0
        den_b = math.sqrt(sum((y - mean_b) ** 2 for y in b)) or 1.0
        return num / (den_a * den_b)


class ScenarioEngine:
    def __init__(self):
        pass

    def simulate(self, stress, predicted_stress, prob, sentiment, news_count):
        scenarios = []
        base = stress
        # Best case
        best = max(0.0, base - 15 - (sentiment * 10))
        scenarios.append(("Best Case", best))
        # Base case
        base_case = (base + predicted_stress) / 2.0
        scenarios.append(("Base Case", base_case))
        # Worst case
        worst = min(100.0, base + 20 + (1 - prob) * 10)
        scenarios.append(("Worst Case", worst))
        # Tail risk
        tail = min(100.0, base + 35 + (news_count / 10.0))
        scenarios.append(("Tail Risk", tail))
        return scenarios


class PredictiveOrgan:
    def __init__(self):
        self.stress_history = []
        self.sentiment_history = []
        self.headline_history = []
        self.ui_intent_history = []
        self.market_micro_history = []
        self.last_prediction = {
            "predicted_stress": 0.0,
            "escalation_probability": 0.0,
            "label": "Unknown",
        }
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=800,
            random_state=42,
        )
        self.mlp_trained = False

    def _encode_intent(self, intent):
        return {
            "idle": 0.0,
            "other": 0.1,
            "news": 0.5,
            "markets": 0.7,
            "maps": 0.9,
        }.get(intent, 0.0)

    def _market_micro_features(self, prices):
        brent = prices.get("Brent") or 0.0
        vix = prices.get("VIX") or 0.0
        gold = prices.get("Gold") or 0.0
        btc = prices.get("BTC") or 0.0
        # crude micro features
        oil_vol = max(0.0, (brent - 80) / 40.0)
        vix_risk = max(0.0, (vix - 15) / 20.0)
        gold_panic = max(0.0, (gold - 1900) / 200.0)
        btc_crash = max(0.0, (35000 - btc) / 15000.0)
        return [oil_vol, vix_risk, gold_panic, btc_crash]

    def update(self, stress, sentiment_score, headline_count, ui_snapshot, prices):
        self.stress_history.append(stress)
        self.sentiment_history.append(sentiment_score)
        self.headline_history.append(headline_count)
        self.ui_intent_history.append(ui_snapshot.get("intent", "idle"))
        micro = self._market_micro_features(prices)
        self.market_micro_history.append(micro)
        if len(self.stress_history) > 400:
            self.stress_history = self.stress_history[-400:]
            self.sentiment_history = self.sentiment_history[-400:]
            self.headline_history = self.headline_history[-400:]
            self.ui_intent_history = self.ui_intent_history[-400:]
            self.market_micro_history = self.market_micro_history[-400:]
        predicted_stress_lin = predict_future_stress_linear(self.stress_history, steps_ahead=6)
        base = predicted_stress_lin / 100.0
        if sentiment_score < -0.2:
            base += 0.18
        elif sentiment_score > 0.2:
            base -= 0.12
        if headline_count > 40:
            base += 0.12
        elif headline_count < 10:
            base -= 0.06
        intent = ui_snapshot.get("intent", "idle")
        conf = ui_snapshot.get("intent_confidence", 0.0)
        if intent in ["markets", "news", "maps"]:
            base += 0.06 * conf
        oil_vol, vix_risk, gold_panic, btc_crash = micro
        base += 0.05 * (oil_vol + vix_risk + gold_panic + btc_crash)
        prob = max(0.0, min(base, 1.0))
        if len(self.stress_history) >= 25:
            X = []
            y = []
            for i in range(5, len(self.stress_history)):
                s = self.stress_history[i - 1]
                ds = self.stress_history[i - 1] - self.stress_history[i - 2]
                sent = self.sentiment_history[i - 1]
                hc = self.headline_history[i - 1]
                inten = self._encode_intent(self.ui_intent_history[i - 1])
                oil_vol, vix_risk, gold_panic, btc_crash = self.market_micro_history[i - 1]
                X.append([s, ds, sent, hc, inten, oil_vol, vix_risk, gold_panic, btc_crash])
                y.append(self.stress_history[i])
            try:
                self.mlp.fit(X, y)
                self.mlp_trained = True
            except Exception as e:
                log(f"[MLP] Training error: {e}")
        predicted_stress_mlp = predicted_stress_lin
        if self.mlp_trained:
            try:
                s = self.stress_history[-1]
                ds = self.stress_history[-1] - self.stress_history[-2] if len(self.stress_history) > 1 else 0.0
                sent = self.sentiment_history[-1]
                hc = self.headline_history[-1]
                inten = self._encode_intent(self.ui_intent_history[-1])
                oil_vol, vix_risk, gold_panic, btc_crash = self.market_micro_history[-1]
                predicted_stress_mlp = float(self.mlp.predict([[s, ds, sent, hc, inten, oil_vol, vix_risk, gold_panic, btc_crash]])[0])
                predicted_stress_mlp = max(0.0, min(predicted_stress_mlp, 100.0))
            except Exception as e:
                log(f"[MLP] Prediction error: {e}")
        combined_pred = 0.5 * predicted_stress_lin + 0.5 * predicted_stress_mlp
        combined_pred = max(0.0, min(combined_pred, 100.0))
        if prob >= 0.8:
            label = "Likely Escalation"
        elif prob >= 0.55:
            label = "Possible Escalation"
        elif prob >= 0.3:
            label = "Low Escalation Risk"
        else:
            label = "Unlikely Escalation"
        self.last_prediction = {
            "predicted_stress": combined_pred,
            "escalation_probability": prob,
            "label": label,
        }
        return self.last_prediction

    def snapshot(self):
        return self.last_prediction.copy()


class VoiceAlert:
    def __init__(self):
        self.engine = None
        try:
            if pyttsx3 is not None:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 180)
                log("[VOICE] pyttsx3 initialized.")
        except Exception as e:
            log(f"[VOICE] Initialization error: {e}")
            self.engine = None

    def speak(self, text):
        if self.engine is None:
            return
        def run():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                log(f"[VOICE] Speak error: {e}")
        threading.Thread(target=run, daemon=True).start()


class AdaptiveScheduler:
    def __init__(self):
        self.market_interval_ms = 300000
        self.news_interval_ms = 180000
        self.predictive_interval_ms = 10000

    def update_intervals(self, stress, news_count):
        if stress < 25 and news_count < 10:
            self.market_interval_ms = 600000
            self.news_interval_ms = 300000
            self.predictive_interval_ms = 15000
        elif stress < 45:
            self.market_interval_ms = 300000
            self.news_interval_ms = 180000
            self.predictive_interval_ms = 10000
        elif stress < 70:
            self.market_interval_ms = 120000
            self.news_interval_ms = 90000
            self.predictive_interval_ms = 7000
        else:
            self.market_interval_ms = 60000
            self.news_interval_ms = 60000
            self.predictive_interval_ms = 5000


class HormuzCockpit(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Global Chokepoint Impact Cockpit (Next-Gen Predictive)")
        self.geometry("1700x950")
        self.configure(bg="#101018")
        self.alert_active = False
        self.alert_flash_state = False
        self.news_headlines = []
        self.ticker_index = 0
        self.stress_history = []
        self.time_history = []
        self.alert_log = []
        self.second_screen = None
        self.alert_threshold_elevated = DEFAULT_ELEVATED
        self.alert_threshold_high = DEFAULT_HIGH
        self.current_region = CURRENT_REGION
        self.region_news_counts = {r: 0 for r in REGION_PROFILES.keys()}
        self.ui_organ = UIAutomationOrgan(poll_interval=2.0)
        self.predictive_organ = PredictiveOrgan()
        self.voice = VoiceAlert()
        self.anomaly_organ = AnomalyOrgan(window=30)
        self.correlation_engine = CorrelationEngine()
        self.scenario_engine = ScenarioEngine()
        self.scheduler = AdaptiveScheduler()
        self.ui_organ.start()
        self._build_ui()
        self._schedule_market_refresh()
        self._schedule_news_refresh()
        self._schedule_ticker_step()
        self._schedule_predictive_refresh()
        self._schedule_auto_region_switch()
        self._schedule_correlation_update()

    def _build_ui(self):
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
            text="Global Chokepoint Impact Monitor (Next-Gen Predictive + Multi-Region + Anomaly)",
            fg="#00ffcc",
            bg="#101018",
            font=("Segoe UI", 18, "bold"),
        )
        self.title_label.pack(side="left", padx=10)
        region_frame = tk.Frame(top, bg="#101018")
        region_frame.pack(side="left", padx=20)
        tk.Label(
            region_frame,
            text="Region:",
            fg="#cccccc",
            bg="#101018",
            font=("Segoe UI", 10),
        ).pack(side="left", padx=(0, 5))
        self.region_var = tk.StringVar(value=self.current_region)
        region_menu = ttk.Combobox(
            region_frame,
            textvariable=self.region_var,
            values=list(REGION_PROFILES.keys()),
            state="readonly",
            width=20,
        )
        region_menu.pack(side="left")
        region_menu.bind("<<ComboboxSelected>>", self.on_region_change)
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
            tk.Label(
                row,
                text=f"{name}:",
                fg="#cccccc",
                bg="#101018",
                font=("Segoe UI", 10),
                width=10,
                anchor="w",
            ).pack(side="left")
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
            text="Impact & Risk",
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
        self.prediction_label = tk.Label(
            impact_frame,
            text="Predicted (6 steps): --",
            fg="#ffcc00",
            bg="#101018",
            font=("Segoe UI", 10, "bold"),
        )
        self.prediction_label.pack(pady=2)
        self.ml_prob_label = tk.Label(
            impact_frame,
            text="Escalation Probability: --",
            fg="#ffcc00",
            bg="#101018",
            font=("Segoe UI", 10, "bold"),
        )
        self.ml_prob_label.pack(pady=2)
        self.ml_label_label = tk.Label(
            impact_frame,
            text="ML Assessment: --",
            fg="#ffcc00",
            bg="#101018",
            font=("Segoe UI", 10, "bold"),
        )
        self.ml_label_label.pack(pady=2)
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
        self.anomaly_label = tk.Label(
            impact_frame,
            text="Anomaly: None",
            fg="#ffaaaa",
            bg="#101018",
            font=("Segoe UI", 9, "bold"),
        )
        self.anomaly_label.pack(pady=3)
        rules_frame = tk.LabelFrame(
            impact_frame,
            text="Custom Alert Rules",
            fg="#ffcc00",
            bg="#101018",
            bd=1,
            font=("Segoe UI", 9, "bold"),
            labelanchor="n",
        )
        rules_frame.pack(fill="x", padx=5, pady=5)
        row1 = tk.Frame(rules_frame, bg="#101018")
        row1.pack(fill="x", pady=2)
        tk.Label(
            row1,
            text="Elevated ≥",
            fg="#cccccc",
            bg="#101018",
            font=("Segoe UI", 9),
        ).pack(side="left")
        self.elevated_entry = tk.Entry(
            row1,
            width=5,
            bg="#181828",
            fg="#ffffff",
            insertbackground="#ffffff",
        )
        self.elevated_entry.insert(0, str(DEFAULT_ELEVATED))
        self.elevated_entry.pack(side="left", padx=3)
        tk.Label(
            row1,
            text="High ≥",
            fg="#cccccc",
            bg="#101018",
            font=("Segoe UI", 9),
        ).pack(side="left", padx=(10, 0))
        self.high_entry = tk.Entry(
            row1,
            width=5,
            bg="#181828",
            fg="#ffffff",
            insertbackground="#ffffff",
        )
        self.high_entry.insert(0, str(DEFAULT_HIGH))
        self.high_entry.pack(side="left", padx=3)
        tk.Button(
            rules_frame,
            text="Apply",
            command=self.apply_custom_rules,
            bg="#202040",
            fg="#00ffcc",
            font=("Segoe UI", 8, "bold"),
        ).pack(pady=3)
        tk.Button(
            impact_frame,
            text="Export Report",
            command=self.export_report,
            bg="#204020",
            fg="#aaffaa",
            font=("Segoe UI", 9, "bold"),
        ).pack(pady=3)
        tk.Button(
            impact_frame,
            text="Generate Daily Briefing",
            command=self.export_daily_briefing,
            bg="#402020",
            fg="#ffaaaa",
            font=("Segoe UI", 9, "bold"),
        ).pack(pady=3)
        tk.Button(
            impact_frame,
            text="Open Second Screen",
            command=self.open_second_screen,
            bg="#202040",
            fg="#00ffcc",
            font=("Segoe UI", 9, "bold"),
        ).pack(pady=5)
        tk.Button(
            impact_frame,
            text="Scenario Forecast",
            command=self.show_scenario_forecast,
            bg="#404020",
            fg="#ffffaa",
            font=("Segoe UI", 9, "bold"),
        ).pack(pady=3)
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
        alertlog_frame.pack(side="left", fill="both", expand=True, padx=(5, 5))
        self.alert_list = tk.Listbox(
            alertlog_frame,
            bg="#181828",
            fg="#ffffff",
            font=("Segoe UI", 9),
            selectbackground="#ff6666",
            activestyle="none",
        )
        self.alert_list.pack(fill="both", expand=True, padx=5, pady=5)
        replay_frame = tk.LabelFrame(
            bottom_left,
            text="Historical Replay",
            fg="#ffcc00",
            bg="#101018",
            bd=2,
            font=("Segoe UI", 11, "bold"),
            labelanchor="n",
        )
        replay_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        self.replay_scale = tk.Scale(
            replay_frame,
            from_=0,
            to=0,
            orient="horizontal",
            bg="#101018",
            fg="#ffffff",
            troughcolor="#181828",
            highlightthickness=0,
            command=self.update_replay_view,
        )
        self.replay_scale.pack(fill="x", padx=5, pady=5)
        self.replay_label = tk.Label(
            replay_frame,
            text="Replay: --",
            fg="#ffffff",
            bg="#101018",
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.replay_label.pack(fill="x", padx=5, pady=5)
        news_frame = tk.LabelFrame(
            right,
            text="War / Crisis News Feed",
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
            text="AI-style Summary & Risk Explanation",
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
            height=10,
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
        tk.Label(
            map_frame,
            text="Open live tanker map in browser (external site).",
            fg="#cccccc",
            bg="#101018",
            font=("Segoe UI", 10),
        ).pack(pady=5)
        tk.Button(
            map_frame,
            text="Open Tanker Map",
            command=self.open_tanker_map,
            bg="#202040",
            fg="#00ffcc",
            font=("Segoe UI", 10, "bold"),
        ).pack(pady=5)
        overlay_frame = tk.LabelFrame(
            right,
            text="Map Overlay Panel",
            fg="#00ffcc",
            bg="#101018",
            bd=2,
            font=("Segoe UI", 11, "bold"),
            labelanchor="n",
        )
        overlay_frame.pack(fill="x", pady=5)
        tk.Label(
            overlay_frame,
            text="Quick access overlay for key chokepoints.\nOpens focused maritime maps.",
            fg="#cccccc",
            bg="#101018",
            font=("Segoe UI", 9),
            justify="left",
        ).pack(pady=5, padx=5, anchor="w")
        tk.Button(
            overlay_frame,
            text="Open Map Overlay",
            command=self.open_map_overlay,
            bg="#202040",
            fg="#00ffcc",
            font=("Segoe UI", 9, "bold"),
        ).pack(pady=5)
        sat_frame = tk.LabelFrame(
            right,
            text="Satellite Imagery Feeds",
            fg="#ffcc00",
            bg="#101018",
            bd=2,
            font=("Segoe UI", 11, "bold"),
            labelanchor="n",
        )
        sat_frame.pack(fill="x", pady=5)
        tk.Label(
            sat_frame,
            text="Open external satellite viewers (NASA Worldview, Sentinel).",
            fg="#cccccc",
            bg="#101018",
            font=("Segoe UI", 9),
            justify="left",
        ).pack(pady=5, padx=5, anchor="w")
        tk.Button(
            sat_frame,
            text="NASA Worldview",
            command=lambda: self.open_url("https://worldview.earthdata.nasa.gov/"),
            bg="#202040",
            fg="#00ffcc",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=5, pady=5)
        tk.Button(
            sat_frame,
            text="Sentinel Hub EO Browser",
            command=lambda: self.open_url("https://apps.sentinel-hub.com/eo-browser/"),
            bg="#202040",
            fg="#00ffcc",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=5, pady=5)
        ticker_frame = tk.Frame(self, bg="#080810")
        ticker_frame.pack(fill="x", side="bottom")
        tk.Label(
            ticker_frame,
            text="CRISIS TICKER:",
            fg="#ffcc00",
            bg="#080810",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left", padx=5)
        self.ticker_label = tk.Label(
            ticker_frame,
            text="Loading headlines...",
            fg="#ffffff",
            bg="#080810",
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.ticker_label.pack(side="left", fill="x", expand=True, padx=5)
        ui_frame = tk.Frame(self, bg="#101018")
        ui_frame.pack(fill="x", side="bottom")
        self.ui_status_label = tk.Label(
            ui_frame,
            text="UI Intent: --",
            fg="#cccccc",
            bg="#101018",
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.ui_status_label.pack(side="left", padx=10, pady=2)
        self.corr_label = tk.Label(
            ui_frame,
            text="Cross-Region Correlation: --",
            fg="#cccccc",
            bg="#101018",
            font=("Segoe UI", 9),
            anchor="e",
        )
        self.corr_label.pack(side="right", padx=10, pady=2)

    def show_error_banner(self, message):
        self.error_banner.config(text=message)
        self.error_banner.pack(fill="x")
        log(f"[UI] ERROR BANNER: {message}")

    def hide_error_banner(self):
        self.error_banner.config(text="")
        self.error_banner.pack_forget()

    def on_region_change(self, event=None):
        self.current_region = self.region_var.get()
        self.log_timeline(f"Region changed to {self.current_region}")
        self.refresh_news()

    def open_url(self, url):
        webbrowser.open(url)

    def open_tanker_map(self):
        url = "https://www.marinetraffic.com/"
        webbrowser.open(url)

    def open_map_overlay(self):
        win = tk.Toplevel(self)
        win.title("Chokepoint Map Overlay")
        win.geometry("600x260")
        win.configure(bg="#101018")
        tk.Label(
            win,
            text=(
                "Map Overlay: key maritime chokepoints.\n"
                "- Strait of Hormuz\n"
                "- Bab el-Mandeb / Red Sea\n"
                "- Taiwan Strait\n"
                "- Black Sea / Ukraine access\n\n"
                "Use the buttons below to open external maritime maps."
            ),
            fg="#ffffff",
            bg="#101018",
            font=("Segoe UI", 10),
            justify="left",
        ).pack(padx=10, pady=10, anchor="w")
        btn_frame = tk.Frame(win, bg="#101018")
        btn_frame.pack(pady=5)
        for region in REGION_PROFILES.keys():
            tk.Button(
                btn_frame,
                text=region,
                command=lambda r=region: self.open_url(REGION_PROFILES[r]["map_url"]),
                bg="#202040",
                fg="#00ffcc",
                font=("Segoe UI", 9, "bold"),
            ).pack(side="left", padx=5)

    def open_second_screen(self):
        if self.second_screen and tk.Toplevel.winfo_exists(self.second_screen):
            self.second_screen.lift()
            return
        self.second_screen = tk.Toplevel(self)
        self.second_screen.title("Chokepoint Second Screen")
        self.second_screen.geometry("800x260")
        self.second_screen.configure(bg="#101018")
        stress_label = tk.Label(
            self.second_screen,
            text=self.stress_label.cget("text"),
            fg="#ff6666",
            bg="#101018",
            font=("Segoe UI", 14, "bold"),
        )
        stress_label.pack(pady=5)
        pred_label = tk.Label(
            self.second_screen,
            text=self.prediction_label.cget("text"),
            fg="#ffcc00",
            bg="#101018",
            font=("Segoe UI", 10, "bold"),
        )
        pred_label.pack(pady=2)
        ml_prob_label = tk.Label(
            self.second_screen,
            text=self.ml_prob_label.cget("text"),
            fg="#ffcc00",
            bg="#101018",
            font=("Segoe UI", 10, "bold"),
        )
        ml_prob_label.pack(pady=2)
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
            pred_label.config(text=self.prediction_label.cget("text"))
            ml_prob_label.config(text=self.ml_prob_label.cget("text"))
            alert_label.config(text=self.alert_label.cget("text"))
            ticker_clone.config(text=self.ticker_label.cget("text"))
            self.after(1000, sync_second_screen)
        sync_second_screen()

    def _schedule_market_refresh(self):
        self.refresh_markets()
        self.after(self.scheduler.market_interval_ms, self._schedule_market_refresh)

    def _schedule_news_refresh(self):
        self.refresh_news()
        self.after(self.scheduler.news_interval_ms, self._schedule_news_refresh)

    def _schedule_ticker_step(self):
        self.update_ticker()
        self.after(500, self._schedule_ticker_step)

    def _schedule_predictive_refresh(self):
        self.refresh_predictive()
        self.after(self.scheduler.predictive_interval_ms, self._schedule_predictive_refresh)

    def _schedule_auto_region_switch(self):
        self.auto_switch_region()
        self.after(300000, self._schedule_auto_region_switch)

    def _schedule_correlation_update(self):
        self.update_correlation_label()
        self.after(600000, self._schedule_correlation_update)

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

    def apply_custom_rules(self):
        try:
            elev = float(self.elevated_entry.get())
            high = float(self.high_entry.get())
            if high <= elev:
                raise ValueError("High must be > Elevated")
            self.alert_threshold_elevated = elev
            self.alert_threshold_high = high
            self.show_error_banner(f"Custom alert rules applied: Elevated ≥ {elev}, High ≥ {high}")
        except Exception as e:
            self.show_error_banner(f"Invalid alert rules, using defaults. ({e})")
            self.alert_threshold_elevated = DEFAULT_ELEVATED
            self.alert_threshold_high = DEFAULT_HIGH

    def export_report(self):
        now = datetime.datetime.now()
        filename = f"report_{now.strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("Global Chokepoint Impact Report\n")
                f.write(f"Generated: {now}\n")
                f.write(f"Region: {self.current_region}\n\n")
                stress = self.stress_history[-1] if self.stress_history else 0
                f.write(f"Stress Index: {stress:.1f}/100\n")
                f.write(f"{self.prediction_label.cget('text')}\n")
                f.write(f"{self.ml_prob_label.cget('text')}\n")
                f.write(f"{self.ml_label_label.cget('text')}\n")
                f.write(f"Risk Label: {self.risk_label.cget('text')}\n")
                f.write(f"Sentiment: {self.sentiment_label.cget('text')}\n")
                f.write(f"Status: {self.status_label.cget('text')}\n\n")
                f.write("Markets:\n")
                for name, lbl in self.market_labels.items():
                    f.write(f"  {name}: {lbl.cget('text')}\n")
                f.write("\nTop Headlines:\n")
                for i, (title, source) in enumerate(self.news_headlines[:10]):
                    f.write(f"  {i+1}. {title} [{source}]\n")
                f.write("\nAlert Log (last 10):\n")
                for entry in self.alert_log[-10:]:
                    f.write(f"  {entry}\n")
            self.show_error_banner(f"Report exported: {filename}")
            log(f"[REPORT] Exported to {filename}")
        except Exception as e:
            self.show_error_banner(f"Failed to export report: {e}")
            log(f"[REPORT] Error exporting report: {e}")

    def export_daily_briefing(self):
        now = datetime.datetime.now()
        filename = f"briefing_{now.strftime('%Y%m%d')}.txt"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("Daily Geopolitical Briefing\n")
                f.write(f"Date: {now.date()}\n")
                f.write(f"Region: {self.current_region}\n\n")
                stress = self.stress_history[-1] if self.stress_history else 0
                f.write(f"Current Stress Index: {stress:.1f}/100\n")
                f.write(f"{self.prediction_label.cget('text')}\n")
                f.write(f"{self.ml_prob_label.cget('text')}\n")
                f.write(f"{self.ml_label_label.cget('text')}\n")
                f.write(f"{self.risk_label.cget('text')}\n")
                f.write(f"{self.sentiment_label.cget('text')}\n\n")
                f.write("Key Headlines:\n")
                for i, (title, source) in enumerate(self.news_headlines[:15]):
                    f.write(f"  {i+1}. {title} [{source}]\n")
                f.write("\nRecent Alerts:\n")
                for entry in self.alert_log[-15:]:
                    f.write(f"  {entry}\n")
                f.write("\nOperator Notes:\n(Add your own observations here.)\n")
            self.show_error_banner(f"Daily briefing exported: {filename}")
            log(f"[BRIEFING] Exported to {filename}")
        except Exception as e:
            self.show_error_banner(f"Failed to export briefing: {e}")
            log(f"[BRIEFING] Error exporting briefing: {e}")

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
        if self.stress_history:
            self.replay_scale.config(to=len(self.stress_history) - 1)
            self.replay_scale.set(len(self.stress_history) - 1)
            self.update_replay_view(len(self.stress_history) - 1)
        elev = self.alert_threshold_elevated
        high = self.alert_threshold_high
        if stress >= high:
            self.activate_alert("HIGH STRESS: Markets reacting strongly.", stress)
        elif stress >= elev:
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
            self.voice.speak(message)
        else:
            self.alert_label.config(text=f"ALERT: {message}")

    def deactivate_alert(self):
        if self.alert_active:
            self.log_alert("Alert cleared", self.stress_history[-1] if self.stress_history else 0)
        self.alert_active = False
        self.alert_label.config(text="ALERT: None", bg="#101018", fg="#ffffff")

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

    def update_replay_view(self, idx):
        try:
            idx = int(idx)
        except Exception:
            return
        if not self.stress_history or idx < 0 or idx >= len(self.stress_history):
            self.replay_label.config(text="Replay: --")
            return
        t = self.time_history[idx]
        s = self.stress_history[idx]
        self.replay_label.config(text=f"Replay: {t.strftime('%Y-%m-%d %H:%M:%S')} | Stress={s:.1f}")

    def refresh_news(self):
        self.status_label.config(text="Status: Updating news...")
        self.update_idletasks()
        self.hide_error_banner()
        try:
            headlines = fetch_all_war_news(self.current_region)
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
        self.sentiment_label.config(text=f"Sentiment: {sentiment_label} (score={sentiment_score:.2f})")
        stress = self.stress_history[-1] if self.stress_history else 0
        risk_score, risk_label = compute_risk_forecast(
            stress, sentiment_score, len(self.news_headlines)
        )
        self.risk_label.config(text=f"Risk Forecast: {risk_label} ({risk_score:.1f}/100)")
        summary_lines = summarize_headlines(self.news_headlines, max_len=3)
        explanation = generate_risk_explanation(
            stress, sentiment_score, len(self.news_headlines), risk_label, self.current_region
        )
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, "Summary of current situation:\n\n")
        for line in summary_lines:
            self.summary_text.insert(tk.END, f"• {line}\n")
        self.summary_text.insert(tk.END, "\nRisk Explanation:\n\n")
        self.summary_text.insert(tk.END, explanation + "\n")
        self.log_timeline("News + sentiment + risk updated.")
        self.ticker_index = 0
        self.status_label.config(text=f"Status: Live ({'OFFLINE' if OFFLINE_MODE else 'ONLINE'})")
        self.region_news_counts[self.current_region] = len(self.news_headlines)
        # anomaly update
        anomaly = self.anomaly_organ.update(stress, sentiment_score, len(self.news_headlines))
        if anomaly:
            self.anomaly_label.config(text=f"Anomaly: {anomaly}")
            self.voice.speak(anomaly)
            self.log_timeline(anomaly)
        else:
            self.anomaly_label.config(text="Anomaly: None")
        # adaptive scheduler update
        self.scheduler.update_intervals(stress, len(self.news_headlines))

    def update_ticker(self):
        if not self.news_headlines:
            self.ticker_label.config(text="No headlines.")
            return
        title, source = self.news_headlines[self.ticker_index % len(self.news_headlines)]
        self.ticker_label.config(text=f"{title} [{source}]")
        self.ticker_index += 1

    def refresh_predictive(self):
        if not self.stress_history:
            return
        stress = self.stress_history[-1]
        sentiment_score = 0.0
        if self.news_headlines:
            sentiment_score, _ = analyze_sentiment(self.news_headlines)
        headline_count = len(self.news_headlines)
        ui_snapshot = self.ui_organ.snapshot()
        ui_text = f"UI Intent: {ui_snapshot.get('intent', 'idle')} (conf={ui_snapshot.get('intent_confidence', 0.0):.2f})"
        self.ui_status_label.config(text=ui_text)
        # approximate latest prices from labels
        prices = {}
        for name in ["Brent", "BTC", "ETH", "Gold", "Silver", "VIX"]:
            txt = self.market_labels[name].cget("text")
            try:
                val = float(txt.split()[0].replace(",", ""))
            except Exception:
                val = None
            prices[name] = val
        pred = self.predictive_organ.update(
            stress,
            sentiment_score,
            headline_count,
            ui_snapshot,
            prices,
        )
        self.prediction_label.config(
            text=f"Predicted (6 steps): {pred['predicted_stress']:.1f} / 100"
        )
        self.ml_prob_label.config(
            text=f"Escalation Probability: {pred['escalation_probability']*100:.1f}%"
        )
        self.ml_label_label.config(text=f"ML Assessment: {pred['label']}")
        # adaptive scheduler update
        self.scheduler.update_intervals(stress, headline_count)

    def auto_switch_region(self):
        if OFFLINE_MODE:
            return
        if not any(self.region_news_counts.values()):
            return
        best_region = max(self.region_news_counts, key=lambda r: self.region_news_counts[r])
        if best_region != self.current_region and self.region_news_counts[best_region] > 0:
            self.current_region = best_region
            self.region_var.set(best_region)
            self.log_timeline(f"Auto-switched region to {best_region} (highest news volume).")
            self.refresh_news()

    def update_correlation_label(self):
        # update correlation engine with current region stress
        if self.stress_history:
            self.correlation_engine.update_region(self.current_region, self.stress_history[-1], len(self.news_headlines))
        corr_map = self.correlation_engine.compute_cross_region_correlations()
        if not corr_map:
            self.corr_label.config(text="Cross-Region Correlation: --")
            return
        # show strongest pair
        best_pair, best_val = max(corr_map.items(), key=lambda kv: abs(kv[1]))
        r1, r2 = best_pair
        self.corr_label.config(text=f"Cross-Region Correlation: {r1} ↔ {r2} ({best_val:.2f})")

    def show_scenario_forecast(self):
        if not self.stress_history:
            return
        stress = self.stress_history[-1]
        pred = self.predictive_organ.snapshot()
        predicted_stress = pred["predicted_stress"]
        prob = pred["escalation_probability"]
        sentiment_score = 0.0
        if self.news_headlines:
            sentiment_score, _ = analyze_sentiment(self.news_headlines)
        scenarios = self.scenario_engine.simulate(
            stress, predicted_stress, prob, sentiment_score, len(self.news_headlines)
        )
        text = f"Scenario forecast for {self.current_region}:\n\n"
        for name, val in scenarios:
            text += f"{name}: {val:.1f}/100\n"
        text += "\n"
        text += generate_scenario_forecast(
            stress, predicted_stress, prob, sentiment_score, self.current_region
        )
        win = tk.Toplevel(self)
        win.title("Scenario Forecast")
        win.geometry("600x400")
        win.configure(bg="#101018")
        txt = tk.Text(
            win,
            bg="#181828",
            fg="#ffffff",
            font=("Segoe UI", 9),
            wrap="word",
        )
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        txt.insert("1.0", text)
        txt.config(state="disabled")

    def destroy(self):
        try:
            self.ui_organ.stop()
        except Exception:
            pass
        super().destroy()


if __name__ == "__main__":
    log("=== Starting Global Chokepoint Cockpit (Next-Gen Predictive) ===")
    app = HormuzCockpit()
    app.mainloop()

