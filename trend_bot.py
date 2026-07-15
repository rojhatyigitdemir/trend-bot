import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import yfinance as yf
import warnings
import time
import os
import datetime as dt
import requests
import json
from google import genai
from google.genai import types

warnings.filterwarnings('ignore')

# --- SETTINGS ---
API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

client = genai.Client(api_key=API_KEY)
MODEL_ID = 'gemini-2.5-flash'

# ====================================================================
# ÇEKİRDEK PORTFÖY (CORE ASSETS)
# ====================================================================
CORE_ASSETS = ["O", "BNDW", "BTC-USD", "ZGLD.SW", "SHEL", "NVDA", "ENA-USD", "ACN", "ZSIL.SW", "BCHE.SW"]

# --- SINYAL GECMISI AYARLARI ---
HISTORY_FILE = "signals_history.csv"
HISTORY_COLUMNS = [
    "run_date", "symbol", "category", "price", "trend",
    "momentum_3m", "ai_signal",
    "eval_date_1m", "realized_return_1m",
    "eval_date_3m", "realized_return_3m",
]
EVAL_DAYS_1M = 30
EVAL_DAYS_3M = 90

def read_portfolio(file_name="portfolio.csv"):
    try:
        df = pd.read_csv(file_name)
        if 'Symbol' in df.columns:
            symbols = df['Symbol'].dropna().tolist()
        else:
            symbols = df.iloc[:, 0].dropna().tolist()
        return [str(s).strip().upper() for s in symbols if str(s).strip()]
    except Exception as e:
        print(f"Error: Could not read {file_name}. Detail: {e}")
        return []

def secure_ai_query(prompt, is_json=False, max_retries=3):
    for attempt in range(max_retries):
        try:
            config_args = {"temperature": 0.2}
            if is_json:
                config_args["response_mime_type"] = "application/json"
                
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(**config_args)
            )
            return response.text.strip()
        except Exception as e:
            print(f"   [API] Deneme {attempt+1} basarisiz. (Google Sunucusu Mesgul). Hata: {e}")
            if attempt < max_retries - 1:
                time.sleep(12) # Google'a nefes almasi icin 12 saniye ver ve tekrar dene
            else:
                return "{}" if is_json else "AI Limit/Connection Error"

def global_macro_intelligence():
    print("🌍 Analyzing Global Macroeconomics, Geopolitics, and Crypto Agenda...")
    macro_symbols = ["^GSPC", "CL=F", "^TNX", "BTC-USD"]
    macro_text = ""
    
    for ms in macro_symbols:
        try:
            ticker = yf.Ticker(ms)
            news = ticker.news
            if news:
                for h in news[:2]:
                    title = h.get('title') or (h.get('content', {}).get('title') if isinstance(h.get('content'), dict) else None)
                    if title:
                        macro_text += f"- {title}\n"
        except Exception:
            pass
            
    if not macro_text:
        return "Currently, there is a calm trend or limited data flow in global markets."
        
    prompt = f"""
    You are a chief economist and macro strategist. Read the following global macro, geopolitical, and blockchain/crypto news headlines:
    {macro_text}
    Based on this information, write a clear, striking, and directive 'GLOBAL STRATEGIC TACTICAL NOTE' for the investor today.
    Start the sentence directly with an action recommendation. Maximum 70 words.
    """
    return secure_ai_query(prompt, is_json=False).replace('\n', ' ')

def batch_download_data(symbols, period="2y", interval="1mo"):
    try:
        data = yf.download(
            tickers=symbols, period=period, interval=interval,
            group_by='ticker', threads=True, progress=False
        )
        if data is None or data.empty:
            return None
        return data
    except Exception as e:
        print(f"   [Uyari] Toplu indirme basarisiz oldu. Detay: {e}")
        return None

def analyze_asset_data(symbol, batch_data=None):
    try:
        data = None
        if batch_data is not None:
            try:
                if isinstance(batch_data.columns, pd.MultiIndex):
                    top_level = batch_data.columns.get_level_values(0)
                    if symbol in top_level:
                        candidate = batch_data[symbol].dropna(how='all')
                        if not candidate.empty:
                            data = candidate
                else:
                    candidate = batch_data.dropna(how='all')
                    if not candidate.empty:
                        data = candidate
            except Exception:
                data = None

        if data is None:
            data = yf.download(symbol, period="2y", interval="1mo", progress=False)

        if data.empty or len(data) < 10:
            return None
        
        close_price = data['Close'][symbol] if isinstance(data.columns, pd.MultiIndex) else data['Close']
        volume = data['Volume'][symbol] if isinstance(data.columns, pd.MultiIndex) else data['Volume']
        
        current_price = close_price.iloc[-1]
        sma_10 = close_price.rolling(window=10).mean().iloc[-1]
        trend = "UPTREND 🟢" if current_price > sma_10 else "DOWNTREND 🔴"
        
        price_3m_ago = close_price.iloc[-4] 
        momentum_3m = ((current_price - price_3m_ago) / price_3m_ago) * 100
        
        try:
            last_full_month_vol = volume.iloc[-2]
            prev_3m_vol_avg = volume.iloc[-5:-2].mean()
            volume_change = ((last_full_month_vol - prev_3m_vol_avg) / prev_3m_vol_avg) * 100 if prev_3m_vol_avg > 0 else 0
            
            if volume_change > 15:
                volume_comment = "Increasing (Strong)"
            elif volume_change < -15:
                volume_comment = "Decreasing (Weak)"
            else:
                volume_comment = "Stable"
        except Exception:
            volume_comment = "No Data"

        return {
            "Asset": symbol, "Price ($)": round(current_price, 2), "Absolute Trend": trend,
            "3M Momentum (%)": round(momentum_3m, 2), "Volume Status": volume_comment,
            "AI Action & Risk Warning": "---"
        }
    except Exception:
        return None

def dual_momentum_and_risk_analysis(symbols):
    results = []
    print(f"AlphaGuard AI Initiating...\nStage 1: Calculating Data...\n")

    all_symbols = list(dict.fromkeys(list(symbols) + CORE_ASSETS))
    batch_data = batch_download_data(all_symbols)
    
    for symbol in symbols:
        asset_data = analyze_asset_data(symbol, batch_data=batch_data)
        if asset_data:
            results.append(asset_data)
            
    df = pd.DataFrame(results)
    
    # 15 Varlik Secimi Güncellendi
    if not df.empty:
        uptrend_assets = df[df["Absolute Trend"] == "UPTREND 🟢"]
        top_15_leaders = uptrend_assets.sort_values(by="3M Momentum (%)", ascending=False).head(15).copy()
    else:
        top_15_leaders = pd.DataFrame()
        
    top_15_leaders['Category'] = 'Dynamic Top 15'
    
    core_results = []
    for symbol in CORE_ASSETS:
        asset_data = analyze_asset_data(symbol, batch_data=batch_data)
        if asset_data:
            asset_data['Category'] = 'Core Foundation'
            core_results.append(asset_data)
        else:
            core_results.append({
                "Asset": symbol, "Price ($)": 0.0, "Absolute Trend": "UNKNOWN", 
                "3M Momentum (%)": 0.0, "Volume Status": "No Data", 
                "AI Action & Risk Warning": "Data Error", "Category": 'Core Foundation'
            })
            
    df_core = pd.DataFrame(core_results)
    final_analysis_list = pd.concat([df_core, top_15_leaders], ignore_index=True)
    
    macro_note = global_macro_intelligence()
    
    print(f"\nStage 2: Packaging Assets for Batch JSON AI Analysis...")
    
    batch_serialized_data = ""
    
    # --- HABER OKUMA DENETİMİ (AUDIT LOG) EKLENDİ ---
    print("\n" + "="*50)
    print("📰 HABER OKUMA DENETIMI (Yapay Zekaya Giden Veri)")
    print("="*50)
    
    for index, row in final_analysis_list.iterrows():
        symbol = row["Asset"]
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            news_titles = [h.get('title') or (h.get('content', {}).get('title') if isinstance(h.get('content'), dict) else "") for h in news[:2]] if news else []
            news_text = " | ".join([t for t in news_titles if t]) if news_titles else "No news."
        except Exception:
            news_text = "No news."
            
        # Hisselerin o an cekilen haberlerini log ekranina yazdiriyoruz
        print(f"[{symbol}] Haberleri: {news_text}")
            
        batch_serialized_data += f"- Asset: {symbol}, Category: {row['Category']}, Trend: {row['Absolute Trend']}, 3M Return: {row['3M Momentum (%)']}%, Volume: {row['Volume Status']}, News: {news_text}\n"

    print("="*50 + "\n")

    # Prompt icerisindeki "15 assets" kismini dinamik hale getirdik (Cunku cekirdek + uydu varlik sayisi degisebilir)
    batch_prompt = f"""
    You are an elite hedge fund manager. Analyze the following {len(final_analysis_list)} assets simultaneously.
    Assets Dataset:
    {batch_serialized_data}

    CRITICAL INSTRUCTION: Generate a short risk/action warning for EACH asset (max 30 words per asset).
    Rule 1: If return is high (15%+) BUT news is bad OR volume is 'Decreasing', output a 'TAKE PROFIT / SELL WARNING ⚠️'.
    Rule 2: If it is a Core Foundation asset, lean towards 'HOLD 🟢' unless news is catastrophic.
    You MUST respond ONLY with a valid JSON object. Keys must be Asset symbols, values the analysis string.
    """
    
    raw_json_response = secure_ai_query(batch_prompt, is_json=True)
    
    try:
        analysis_dict = json.loads(raw_json_response)
        for index, row in final_analysis_list.iterrows():
            symbol = row["Asset"]
            ai_comment = analysis_dict.get(symbol, "Hold and monitor technical levels.")
            final_analysis_list.at[index, "AI Action & Risk Warning"] = ai_comment.replace('\n', ' ')
    except Exception as e:
        print(f"JSON Parse Error: {e}")
        for index, row in final_analysis_list.iterrows():
            final_analysis_list.at[index, "AI Action & Risk Warning"] = "Technical Review Required"

    df_display = final_analysis_list.drop(columns=["Volume Status"])
    return df_display, macro_note

def load_signal_history():
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE, parse_dates=["run_date", "eval_date_1m", "eval_date_3m"])
            for col in HISTORY_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.NA
            return df[HISTORY_COLUMNS]
        except Exception as e:
            print(f"[Uyari] signals_history okunamadi. Detay: {e}")
    return pd.DataFrame(columns=HISTORY_COLUMNS)

def get_latest_price(symbol):
    try:
        data = yf.Ticker(symbol).history(period="5d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except Exception:
        pass
    return None

def update_realized_returns(history_df):
    if history_df.empty:
        return history_df

    today = pd.Timestamp(dt.date.today())
    price_cache = {}

    for idx, row in history_df.iterrows():
        run_date = row["run_date"]
        if pd.isna(run_date): continue
        
        days_passed = (today - run_date).days
        entry_price = row["price"]

        needs_1m = days_passed >= EVAL_DAYS_1M and pd.isna(row.get("realized_return_1m"))
        needs_3m = days_passed >= EVAL_DAYS_3M and pd.isna(row.get("realized_return_3m"))

        if not (needs_1m or needs_3m): continue
