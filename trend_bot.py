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

# ====================================================================
# --- SETTINGS (AYARLAR) ---
# ====================================================================
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    client = None

MODEL_ID = 'gemini-2.5-flash'

CORE_ASSETS = ["O", "BNDW", "BTC-USD", "ZGLD.SW", "SHEL", "ALL", "MU", "KLAC", "ZSIL.SW", "BCHE.SW"]

# --------------------------------------------------------------------
# Strateji Parametreleri
# --------------------------------------------------------------------
DATA_PERIOD = "2y"
DATA_INTERVAL = "1wk"
SMA_WINDOW = 8
MOMENTUM_WEEKS = 4
MOMENTUM_COL = f"{MOMENTUM_WEEKS}W Mom (%)"
RETURN_1W_COL = "1W Ret (%)"

TRIM_MOMENTUM_THRESHOLD = 10.0
REBALANCE_WEEKDAYS = {0, 4} # Pazartesi (0) ve Cuma (4)
SHOCK_LOOKBACK_DAYS = 3
SHOCK_THRESHOLD_PCT = 6.0

HISTORY_FILE = "signals_history.csv"
HISTORY_COLUMNS = [
    "run_date", "symbol", "category", "price", "trend",
    f"momentum_{MOMENTUM_WEEKS}w", "ai_signal",
    "eval_date_1w", "realized_return_1w",
    "eval_date_4w", "realized_return_4w",
]
EVAL_DAYS_1W = 7
EVAL_DAYS_4W = 28


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

def secure_ai_query(prompt, max_retries=3):
    if not client:
        return "API Key Eksik."

    for attempt in range(max_retries):
        try:
            config_args = {
                "temperature": 0.1,
                "max_output_tokens": 8192
            } 

            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(**config_args)
            )
            
            return response.text.strip()
        except Exception as e:
            print(f"   [API Hatası] Deneme {attempt+1}: {e}")
            if attempt < max_retries - 1: time.sleep(12)
            else: return "AI Sunucu Hatası"

def global_macro_intelligence():
    macro_symbols = ["^GSPC", "CL=F", "^TNX", "BTC-USD"]
    macro_text = ""
    for ms in macro_symbols:
        try:
            ticker = yf.Ticker(ms)
            news = ticker.news
            if news:
                for h in news[:2]:
                    title = h.get('title') or (h.get('content', {}).get('title') if isinstance(h.get('content'), dict) else None)
                    if title: macro_text += f"- {title}\n"
        except Exception:
            pass

    if not macro_text: return "Calm trend in global markets."
    prompt = f"You are a chief economist. Read these headlines: {macro_text}. Write a 70-word GLOBAL STRATEGIC TACTICAL NOTE."
    return secure_ai_query(prompt).replace('\n', ' ')

def defcon_shock_monitor(symbols, macro_note):
    print("\n🔍 DEFCON Protokolü Başlatıldı (Matematiksel ATR & Yapay Zeka Haber Analizi)...")
    alerts = []
    news_dataset = ""
    
    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(period="1mo", interval="1d")
            if hist.empty or len(hist) < 15: continue
            
            hist['H-L'] = hist['High'] - hist['Low']
            hist['H-PC'] = abs(hist['High'] - hist['Close'].shift(1))
            hist['L-PC'] = abs(hist['Low'] - hist['Close'].shift(1))
            hist['TR'] = hist[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            hist['ATR'] = hist['TR'].rolling(window=14).mean()
            
            yesterday_atr = hist['ATR'].iloc[-2]
            today_high = hist['High'].iloc[-1]
            today_low = hist['Low'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            
            if pd.isna(yesterday_atr) or yesterday_atr == 0: continue
            
            gap_up = today_high - prev_close
            gap_down = prev_close - today_low
            
            if gap_up > (2.5 * yesterday_atr):
                alerts.append(f"📈 [MATEMATİK ŞOK] {sym}: +{(gap_up/prev_close)*100:.1f}% (2.5x ATR aşıldı) - Yukarı Yönlü Hacimli Sıçrama!")
            elif gap_down > (2.5 * yesterday_atr):
                alerts.append(f"📉 [MATEMATİK ŞOK] {sym}: -{(gap_down/prev_close)*100:.1f}% (2.5x ATR aşıldı) - Panik Satışı (Stop-Loss) Kırılımı!")

            news = yf.Ticker(sym).news
            titles = [h.get('title') for h in news[:2]] if news else []
            if titles:
                news_text = " | ".join(titles)
                news_dataset += f"- Varlık: {sym}, Haber: {news_text}\n"
        except Exception:
            continue

    if news_dataset:
        prompt = f"""
        Sen bir Acil Durum (Kriz) Yöneticisisin. Küresel durum: {macro_note}
        Aşağıdaki varlıklara ait son dakika haberlerini oku:
        {news_dataset}
        
        GÖREVİN: Bu haberlerin varlığın trendini değiştirip değiştirmeyeceğini bulmak.
        1. "🚀 YÜKSELİŞ ŞOKU" (Oyun değiştirici, muazzam iyi haber)
        2. "🚨 DÜŞÜŞ ŞOKU" (Trendi öldüren felaket haberi)
        3. "⚪ GÜRÜLTÜ" (Sıradan haber)
        
        KURAL: Çoğu haber "GÜRÜLTÜ"dür. Sadece gerçekten yıkıcı veya patlayıcı haberlere ŞOK etiketi ver.
        
        FORMAT ZORUNLULUĞU (ÇOK ÖNEMLİ):
        JSON KULLANMA. Her varlık için sadece ŞU FORMATTA tek bir satır yaz. Araya dik çizgi (|) koy:
        SEMBOL | ŞOK TÜRÜ | Gerekçe Cümlesi
        
        Örnek Çıktı:
        CVX | 🚀 YÜKSELİŞ ŞOKU | İran gerilimi petrol arzını tehdit ediyor, acil ralli katalizörü.
        AAPL | ⚪ GÜRÜLTÜ | Yeni telefon sızıntıları fiyata etki etmez.
        """
        raw_response = secure_ai_query(prompt)
        try:
            for line in raw_response.split('\n'):
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        sym = parts[0].strip().replace('*', '') 
                        analysis = f"{parts[1].strip()} - {parts[2].strip()}"
                        if "YÜKSELİŞ ŞOKU" in analysis or "DÜŞÜŞ ŞOKU" in analysis:
                            alerts.append(f"🧠 [AI HABER İSTİHBARATI] {sym}: {analysis}")
        except Exception as e:
            print(f"DEFCON AI Ayrıştırma Hatası: {e}")
            
    return alerts

# ====================================================================
# --- DOSYA KONTROLÜ KALDIRILDI - HER TETİKLENMEDE ÇALIŞIR ---
# ====================================================================
def is_rebalance_day():
    today = dt.date.today()
    return today.weekday() in REBALANCE_WEEKDAYS

def batch_download_data(symbols, period=DATA_PERIOD, interval=DATA_INTERVAL):
    try:
        data = yf.download(tickers=symbols, period=period, interval=interval, group_by='ticker', threads=True, progress=False)
        return None if data is None or data.empty else data
    except Exception: return None

def analyze_asset_data(symbol, batch_data=None):
    try:
        data = None
        if batch_data is not None:
            if isinstance(batch_data.columns, pd.MultiIndex):
                if symbol in batch_data.columns.get_level_values(0): data = batch_data[symbol].dropna(how='all')
            else:
                data = batch_data.dropna(how='all')

        if data is None: data = yf.download(symbol, period=DATA_PERIOD, interval=DATA_INTERVAL, progress=False)
        if data.empty or len(data) < SMA_WINDOW + MOMENTUM_WEEKS + 3: return None

        close_series = (data['Close'][symbol] if isinstance(data.columns, pd.MultiIndex) else data['Close']).ffill()
        volume_series = (data['Volume'][symbol] if isinstance(data.columns, pd.MultiIndex) else data['Volume']).ffill()

        if close_series.isna().all(): return None

        last_bar_date = pd.Timestamp(close_series.index[-1]).tz_localize(None)
        today = pd.Timestamp(dt.date.today())
        current_week_open = today - pd.Timedelta(days=today.weekday())
        is_last_bar_open_week = last_bar_date >= current_week_open

        current_price = float(close_series.iloc[-1])
        
        if is_last_bar_open_week and len(close_series) > 1:
            closed_close = close_series.iloc[:-1]
            closed_volume = volume_series.iloc[:-1]
        else:
            closed_close = close_series
            closed_volume = volume_series

        if len(closed_close) < SMA_WINDOW + MOMENTUM_WEEKS + 1: return None

        sma_ref = closed_close.rolling(window=SMA_WINDOW).mean().dropna().iloc[-1]
        trend = "UPTREND 🟢" if current_price > sma_ref else "DOWNTREND 🔴"

        momentum_ref_price = closed_close.iloc[-(MOMENTUM_WEEKS)] 
        momentum_pct = ((current_price - momentum_ref_price) / momentum_ref_price) * 100
        
        one_week_ref_price = closed_close.iloc[-1]
        one_week_pct = ((current_price - one_week_ref_price) / one_week_ref_price) * 100

        volume_change = 0
        try:
            last_full_week_vol = closed_volume.iloc[-1]
            prev_avg_vol = closed_volume.iloc[-(MOMENTUM_WEEKS + 1):-1].mean()
            if prev_avg_vol > 0:
                volume_change = ((last_full_week_vol - prev_avg_vol) / prev_avg_vol) * 100

            if volume_change > 15: volume_comment = "Increasing (Strong)"
            elif volume_change < -15: volume_comment = "Decreasing (Weak)"
            else: volume_comment = "Stable"
        except Exception:
            volume_comment = "No Data"

        return {
            "Asset": symbol,
            "Price ($)": round(current_price, 2),
            "Absolute Trend": trend,
            RETURN_1W_COL: round(one_week_pct, 2),
            MOMENTUM_COL: round(momentum_pct, 2),
            "Volume_Num": volume_change, 
            "Volume Status": volume_comment,
            "AI Action & Risk Warning": "---"
        }
    except Exception: return None

def dual_momentum_and_risk_analysis(symbols, macro_note):
    results = []
    print(f"AlphaGuard AI Initiating Rebalance...\nStage 1: Calculating Data...\n")

    all_symbols = list(dict.fromkeys(list(symbols) + CORE_ASSETS))
    batch_data = batch_download_data(all_symbols)

    for symbol in symbols:
        asset_data = analyze_asset_data(symbol, batch_data=batch_data)
        if asset_data: results.append(asset_data)

    df = pd.DataFrame(results)

    if not df.empty:
        uptrend_assets = df[df["Absolute Trend"] == "UPTREND 🟢"].copy()
        uptrend_assets = uptrend_assets.sort_values(by=MOMENTUM_COL, ascending=False)
        if not uptrend_assets.empty: uptrend_assets['Category'] = 'Dynamic Uptrend'

        downtrend_assets = df[df["Absolute Trend"] == "DOWNTREND 🔴"].copy()
        downtrend_assets = downtrend_assets.sort_values(by=RETURN_1W_COL, ascending=False)
        if not downtrend_assets.empty: downtrend_assets['Category'] = 'Downtrend / Reversal'
        
        all_portfolio_assets = pd.concat([uptrend_assets, downtrend_assets], ignore_index=True)
    else:
        all_portfolio_assets = pd.DataFrame()

    core_results = []
    for symbol in CORE_ASSETS:
        asset_data = analyze_asset_data(symbol, batch_data=batch_data)
        if asset_data:
            asset_data['Category'] = 'Core Foundation'
            core_results.append(asset_data)
        else:
            core_results.append({
                "Asset": symbol, "Price ($)": 0.0, "Absolute Trend": "UNKNOWN", RETURN_1W_COL: 0.0,
                MOMENTUM_COL: 0.0, "Volume_Num": 0.0, "Volume Status": "No Data",
                "AI Action & Risk Warning": "Data Error", "Category": 'Core Foundation'
            })

    final_analysis_list = pd.concat([pd.DataFrame(core_results), all_portfolio_assets], ignore_index=True)

    print(f"\nStage 2: Packaging Assets for Bulletproof Text AI Analysis...")
    batch_serialized_data = ""
    for index, row in final_analysis_list.iterrows():
        symbol = row["Asset"]
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            news_titles = [h.get('title') or (h.get('content', {}).get('title') if isinstance(h.get('content'), dict) else "") for h in news[:2]] if news else []
            news_text = " | ".join([t for t in news_titles if t]) if news_titles else "No news."
        except Exception: news_text = "No news."
        batch_serialized_data += f"- Asset: {symbol}, Category: {row['Category']}, Trend: {row['Absolute Trend']}, 1W Ret: {row[RETURN_1W_COL]}%, {MOMENTUM_WEEKS}W Ret: {row[MOMENTUM_COL]}%, Volume: {row['Volume Status']}, News: {news_text}\n"

    batch_prompt = f"""
    You are an elite hedge fund manager. 
    Global context: {macro_note}

    Analyze the following {len(final_analysis_list)} assets:
    {batch_serialized_data}

    CRITICAL INSTRUCTIONS:
    1. Evaluate EVERY asset.
    2. Choose ONE tag: [STRONG BUY 🚀], [ACCUMULATE 🟢], [HOLD 🟡], [TRIM 🟠], or [SELL 🔴].
    3. Provide a max 15-word justification.
    4. RULE for TRIM: If {MOMENTUM_WEEKS}W Ret > {TRIM_MOMENTUM_THRESHOLD}% BUT news is bad OR volume is 'Decreasing', use [TRIM 🟠].
    5. RULE for DOWNTREND: If Category is 'Downtrend / Reversal', evaluate if Volume is 'Increasing' and 1W Ret is positive (bottom-fishing opportunity), otherwise strongly lean towards [SELL 🔴] or [HOLD 🟡].
    
    FORMAT ZORUNLULUĞU (ÇOK ÖNEMLİ):
    DO NOT RETURN JSON. JSON KULLANMA. Return plain text in EXACTLY this format, one line per asset. Araya dik çizgi (|) koy.
    
    Örnek Çıktı:
    O | [HOLD 🟡] | Dividend is safe but technicals are weak.
    CVX | [TRIM 🟠] | Taking profits due to geopolitical risks.
    """

    raw_response = secure_ai_query(batch_prompt)

    analysis_dict = {}
    for line in raw_response.split('\n'):
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 3:
                sym = parts[0].strip().replace('*', '') 
                tag = parts[1].strip()
                justification = parts[2].strip()
                analysis_dict[sym] = f"{tag} {justification}"

    for index, row in final_analysis_list.iterrows():
        final_analysis_list.at[index, "AI Action & Risk Warning"] = analysis_dict.get(row["Asset"], "Hold and monitor (AI Parse Issue).")

    df_display = final_analysis_list.drop(columns=["Volume_Num", "Volume Status"])
    return df_display

def load_signal_history():
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            for col in ["run_date", "eval_date_1w", "eval_date_4w"]:
                if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
            for col in HISTORY_COLUMNS:
                if col not in df.columns: df[col] = pd.NA
            return df
        except Exception: pass
    return pd.DataFrame(columns=HISTORY_COLUMNS)

def get_latest_price(symbol):
    try:
        data = yf.Ticker(symbol).history(period="5d")
        if not data.empty: return float(data['Close'].ffill().iloc[-1])
    except Exception: pass
    return None

def update_realized_returns(history_df):
    if history_df.empty: return history_df
    today = pd.Timestamp(dt.date.today())
    price_cache = {}

    for idx, row in history_df.iterrows():
        run_date = row.get("run_date")
        if pd.isna(run_date): continue
        days_passed = (today - run_date).days
        entry_price = row.get("price")

        needs_1w = days_passed >= EVAL_DAYS_1W and pd.isna(row.get("realized_return_1w"))
        needs_4w = days_passed >= EVAL_DAYS_4W and pd.isna(row.get("realized_return_4w"))
        if not (needs_1w or needs_4w): continue

        if row["symbol"] not in price_cache: price_cache[row["symbol"]] = get_latest_price(row["symbol"])
        current_price = price_cache[row["symbol"]]
        
        # SIFIRA BÖLME (ZeroDivisionError) HATASI ÇÖZÜMÜ: entry_price kontrolü
        if current_price is None or pd.isna(entry_price) or float(entry_price) == 0.0: continue

        realized_return = round(((current_price - entry_price) / entry_price) * 100, 2)
        if needs_1w:
            history_df.at[idx, "realized_return_1w"] = realized_return
            history_df.at[idx, "eval_date_1w"] = today
        if needs_4w:
            history_df.at[idx, "realized_return_4w"] = realized_return
            history_df.at[idx, "eval_date_4w"] = today
    return history_df

def append_new_signals(history_df, final_analysis_list):
    today = pd.Timestamp(dt.date.today())
    new_rows = []
    for _, row in final_analysis_list.iterrows():
        new_rows.append({
            "run_date": today, "symbol": row["Asset"], "category": row["Category"],
            "price": row["Price ($)"], "trend": row["Absolute Trend"], f"momentum_{MOMENTUM_WEEKS}w": row[MOMENTUM_COL],
            "ai_signal": row["AI Action & Risk Warning"], "eval_date_1w": pd.NaT,
            "realized_return_1w": pd.NA, "eval_date_4w": pd.NaT, "realized_return_4w": pd.NA,
        })
    return pd.concat([history_df, pd.DataFrame(new_rows)], ignore_index=True)

def generate_accuracy_summary(history_df):
    def is_hit(row, col):
        ret = row.get(col)
        if pd.isna(ret): return None
        if any(tag in str(row.get("ai_signal", "")) for tag in ["SELL", "TRIM"]): return ret < 0
        return ret > 0

    if "realized_return_1w" not in history_df.columns: return "Veri hazir degil."
    evaluated_1w = history_df.dropna(subset=["realized_return_1w"]).copy()
    if evaluated_1w.empty: return "Henuz 1 haftasi dolmus sinyal yok."

    evaluated_1w["hit"] = evaluated_1w.apply(lambda r: is_hit(r, "realized_return_1w"), axis=1)
    summary = f"[1 Hafta] Sinyal: {len(evaluated_1w)} | Ort. Getiri: {evaluated_1w['realized_return_1w'].mean():.2f}% | Isabet: {evaluated_1w['hit'].mean() * 100:.1f}%"

    if "realized_return_4w" in history_df.columns:
        evaluated_4w = history_df.dropna(subset=["realized_return_4w"]).copy()
        if not evaluated_4w.empty:
            evaluated_4w["hit"] = evaluated_4w.apply(lambda r: is_hit(r, "realized_return_4w"), axis=1)
            summary += f"\n[4 Hafta] Sinyal: {len(evaluated_4w)} | Ort. Getiri: {evaluated_4w['realized_return_4w'].mean():.2f}% | Isabet: {evaluated_4w['hit'].mean() * 100:.1f}%"
    return summary

def send_telegram_message(messages):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: 
        print("❌ HATA: Token veya Chat ID boş!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    
    if isinstance(messages, str): messages = [messages[i:i+3900] for i in range(0, len(messages), 3900)]
        
    for i, msg in enumerate(messages):
        try: 
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
        except Exception: pass
        time.sleep(1)

def build_full_report_messages(macro_note, final_report_df, accuracy_summary, shock_alerts):
    messages = []
    current_time = dt.datetime.now().strftime("%H:%M")
    rebalance_label = f"Rebalance ({current_time} Raporu)"
    
    msg1 = "=" * 50 + f"\n🌍 GLOBAL STRATEJIK NOT\n" + "=" * 50 + f"\n{macro_note}\n\n"
    if shock_alerts: 
        msg1 += "=" * 50 + "\n🚨 DEFCON ŞOK UYARILARI 🚨\n" + "=" * 50 + "\n" + "\n".join(shock_alerts)
    else:
        msg1 += "✅ DEFCON: Fiyat ve Haber Şoku Tespit Edilmedi.\n"
    messages.append(msg1.strip())
    
    chunk_size = 15
    for i in range(0, len(final_report_df), chunk_size):
        df_chunk = final_report_df.iloc[i:i+chunk_size]
        msg_df = "=" * 50 + f"\n🏛️ PORTFOY RAPORU {rebalance_label} - Parça {i//chunk_size + 1}\n" + "=" * 50 + "\n"
        msg_df += df_chunk.to_string(index=False)
        messages.append(msg_df)
        
    msg_perf = "=" * 50 + "\n📊 GECMIS PERFORMANS\n" + "=" * 50 + "\n" + accuracy_summary
    messages.append(msg_perf)
    
    return messages

if __name__ == "__main__":
    try:
        print("\n🚀 ALPHAGUARD SİSTEMİ BAŞLATILIYOR...")
        watchlist = read_portfolio("portfolio.csv")
        watchlist = [s for s in watchlist if s not in CORE_ASSETS] if watchlist else []
        
        all_monitored_symbols = list(set(CORE_ASSETS + watchlist))

        # 1. MAKRO & DEFCON ŞOK İZLEMESİ (HER TETİKLENMEDE ÇALIŞIR)
        macro_note = global_macro_intelligence()
        shock_alerts = defcon_shock_monitor(all_monitored_symbols, macro_note)

        # 2. REBALANCE (Pzt ve Cuma günde 2 kez tüm varlıkları gönderir)
        if is_rebalance_day():
            final_report = dual_momentum_and_risk_analysis(watchlist, macro_note)
            pd.set_option('display.max_colwidth', None)
            
            history_df = update_realized_returns(load_signal_history())
            history_df = append_new_signals(history_df, final_report)
            cols = [c for c in history_df.columns if c in HISTORY_COLUMNS or "1m" in c or "3m" in c]
            history_df[cols].to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
            
            report_messages = build_full_report_messages(macro_note, final_report, generate_accuracy_summary(history_df), shock_alerts)
            print("\n\n".join(report_messages))
            send_telegram_message(report_messages)
        else:
            # SADECE İZLEME (Diğer günler)
            report_text = f"🌍 GÜNLÜK İZLEME & DEFCON\n{macro_note}\n\n" 
            if shock_alerts:
                report_text += "🚨 ŞOK TESPİT EDİLDİ 🚨\n" + "\n".join(shock_alerts)
            else:
                report_text += "✅ DEFCON: Fiyat kırılımı veya oyun değiştirici haber tespit edilmedi."
            
            print(report_text)
            send_telegram_message(report_text)
            
        print("\n🏁 SİSTEM BAŞARIYLA TAMAMLANDI!")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR (Sistem Çöktü): {e}")
        raise e
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR (Sistem Çöktü): {e}")
        raise e
