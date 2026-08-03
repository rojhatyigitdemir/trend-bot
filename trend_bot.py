import sys
# Windows sunucularinda emoji ve ozel karakterlerin (Unicode) cokmemesi icin zorunlu UTF-8 ayari
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
API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    client = None

MODEL_ID = 'gemini-2.5-flash'

CORE_ASSETS = ["O", "BNDW", "BTC-USD", "ZGLD.SW", "SHEL", "ALL", "MU", "KLAC", "ZSIL.SW", "BCHE.SW"]

# --------------------------------------------------------------------
# Strateji Parametreleri
# --------------------------------------------------------------------
DATA_PERIOD = "2y"              # veri cekim penceresi
DATA_INTERVAL = "1wk"           # haftalik bar
SMA_WINDOW = 8                  # haftalik SMA penceresi (~2 ay)
MOMENTUM_WEEKS = 4               # haftalik momentum penceresi (~1 ay)
MOMENTUM_COL = f"{MOMENTUM_WEEKS}W Momentum (%)"

# "Haftada %5 hedefi" bir ADAY SECIM FILTRESIDIR.
CANDIDATE_MOMENTUM_THRESHOLD = 5.0
SATELLITE_TOP_N = 20            # YENI: Top 15 -> Top 20

# TRIM kurali icin esik
TRIM_MOMENTUM_THRESHOLD = 10.0

# YENI: Pazartesi (0) ve Cuma (4) günleri rebalance çalışır
REBALANCE_WEEKDAYS = {0, 4}
LAST_REBALANCE_FILE = "last_rebalance.txt"

# Gunluk "sok" kontrolu
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


def secure_ai_query(prompt, is_json=False, max_retries=3):
    if not client:
        return "{}" if is_json else "API Key Eksik. AI Degerlendirmesi Yapilamadi."

    for attempt in range(max_retries):
        try:
            config_args = {"temperature": 0.2}
            if is_json:
                config_args["response_mime_type"] = "application/json"
                config_args["max_output_tokens"] = 8192  # JSON kesilmesini onlemek icin artirildi

            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(**config_args)
            )
            
            result_text = response.text.strip()
            
            # YENI: JSON format temizligi (LLM markdown block icinde dondururse)
            if is_json:
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                result_text = result_text.strip()
                
            return result_text
        except Exception as e:
            print(f"   [API] Deneme {attempt+1} basarisiz. Hata: {e}")
            if attempt < max_retries - 1:
                time.sleep(12)
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


def daily_shock_check(symbols, lookback_days=SHOCK_LOOKBACK_DAYS, threshold=SHOCK_THRESHOLD_PCT):
    alerts = []
    for symbol in symbols:
        try:
            hist = yf.Ticker(symbol).history(period="10d", interval="1d")
            if hist.empty or len(hist) < lookback_days + 1:
                continue
            # YENI: NaN degerleri ileri dogru doldur
            close = hist['Close'].ffill()
            change_pct = ((close.iloc[-1] - close.iloc[-(lookback_days + 1)]) / close.iloc[-(lookback_days + 1)]) * 100
            if abs(change_pct) >= threshold:
                direction = "sert dustu 🔻" if change_pct < 0 else "sert yukseldi 🔺"
                alerts.append(f"⚠️ {symbol}: son {lookback_days} gunde {direction} ({change_pct:+.2f}%)")
        except Exception:
            continue
    return alerts


def is_rebalance_day():
    today = dt.date.today()
    if today.weekday() not in REBALANCE_WEEKDAYS:
        return False
    if os.path.exists(LAST_REBALANCE_FILE):
        try:
            with open(LAST_REBALANCE_FILE, "r") as f:
                last = f.read().strip()
            if last == str(today):
                return False  # ayni gun icinde birden fazla tetiklenmeyi onle
        except Exception:
            pass
    return True


def mark_rebalance_done():
    with open(LAST_REBALANCE_FILE, "w") as f:
        f.write(str(dt.date.today()))


def batch_download_data(symbols, period=DATA_PERIOD, interval=DATA_INTERVAL):
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
            data = yf.download(symbol, period=DATA_PERIOD, interval=DATA_INTERVAL, progress=False)

        if data.empty or len(data) < SMA_WINDOW + MOMENTUM_WEEKS + 3:
            return None

        # YENI: İsviçre hisselerindeki NaN hatalarını çözmek için .ffill() kullanıyoruz
        close_series = (data['Close'][symbol] if isinstance(data.columns, pd.MultiIndex) else data['Close']).ffill()
        volume_series = (data['Volume'][symbol] if isinstance(data.columns, pd.MultiIndex) else data['Volume']).ffill()

        if close_series.isna().all():
            return None

        # --- LOOK-AHEAD BIAS & CUMA/PAZARTESİ FIX ---
        last_bar_date = pd.Timestamp(close_series.index[-1]).tz_localize(None)
        today = pd.Timestamp(dt.date.today())
        current_week_open = today - pd.Timedelta(days=today.weekday())
        is_last_bar_open_week = last_bar_date >= current_week_open

        # Güncel Fiyat (Pzt veya Cuma, o günkü en son geçerli fiyat)
        current_price = float(close_series.iloc[-1])
        
        # SMA sadece 'kapanmış' haftalar üzerinden hesaplanacak ki look-ahead bias olmasın
        if is_last_bar_open_week and len(close_series) > 1:
            closed_close = close_series.iloc[:-1]
            closed_volume = volume_series.iloc[:-1]
        else:
            closed_close = close_series
            closed_volume = volume_series

        if len(closed_close) < SMA_WINDOW + MOMENTUM_WEEKS + 1:
            return None

        sma_ref = closed_close.rolling(window=SMA_WINDOW).mean().dropna().iloc[-1]
        
        # YENI: Trend ve momentum doğrudan "current_price" ile ölçülür, böylece hafta içi fiyat hareketi (Cuma'ya kadar olan) hesaplamaya girer.
        trend = "UPTREND 🟢" if current_price > sma_ref else "DOWNTREND 🔴"

        momentum_ref_price = closed_close.iloc[-(MOMENTUM_WEEKS)] 
        momentum_pct = ((current_price - momentum_ref_price) / momentum_ref_price) * 100

        try:
            last_full_week_vol = closed_volume.iloc[-1]
            prev_avg_vol = closed_volume.iloc[-(MOMENTUM_WEEKS + 1):-1].mean()
            volume_change = ((last_full_week_vol - prev_avg_vol) / prev_avg_vol) * 100 if prev_avg_vol > 0 else 0

            if volume_change > 15:
                volume_comment = "Increasing (Strong)"
            elif volume_change < -15:
                volume_comment = "Decreasing (Weak)"
            else:
                volume_comment = "Stable"
        except Exception:
            volume_comment = "No Data"

        return {
            "Asset": symbol,
            "Price ($)": round(current_price, 2),
            "Absolute Trend": trend,
            MOMENTUM_COL: round(momentum_pct, 2),
            "Volume Status": volume_comment,
            "AI Action & Risk Warning": "---"
        }
    except Exception as e:
        print(f"   [Uyari] {symbol} analiz edilemedi: {e}")
        return None


def dual_momentum_and_risk_analysis(symbols, macro_note):
    results = []
    
    today_name = "Pazartesi" if dt.date.today().weekday() == 0 else "Cuma"
    print(f"AlphaGuard AI Initiating ({today_name} Rebalance)...\nStage 1: Calculating Data...\n")

    all_symbols = list(dict.fromkeys(list(symbols) + CORE_ASSETS))
    batch_data = batch_download_data(all_symbols)

    for symbol in symbols:
        asset_data = analyze_asset_data(symbol, batch_data=batch_data)
        if asset_data:
            results.append(asset_data)

    df = pd.DataFrame(results)

    if not df.empty:
        uptrend_assets = df[df["Absolute Trend"] == "UPTREND 🟢"]
        strong_candidates = uptrend_assets[uptrend_assets[MOMENTUM_COL] >= CANDIDATE_MOMENTUM_THRESHOLD]
        pool = strong_candidates if not strong_candidates.empty else uptrend_assets
        top_leaders = pool.sort_values(by=MOMENTUM_COL, ascending=False).head(SATELLITE_TOP_N).copy()
    else:
        top_leaders = pd.DataFrame()

    top_leaders['Category'] = 'Dynamic Top Candidates'

    core_results = []
    for symbol in CORE_ASSETS:
        asset_data = analyze_asset_data(symbol, batch_data=batch_data)
        if asset_data:
            asset_data['Category'] = 'Core Foundation'
            core_results.append(asset_data)
        else:
            core_results.append({
                "Asset": symbol, "Price ($)": 0.0, "Absolute Trend": "UNKNOWN",
                MOMENTUM_COL: 0.0, "Volume Status": "No Data",
                "AI Action & Risk Warning": "Data Error", "Category": 'Core Foundation'
            })

    df_core = pd.DataFrame(core_results)
    final_analysis_list = pd.concat([df_core, top_leaders], ignore_index=True)

    print(f"\nStage 2: Packaging Assets for Batch JSON AI Analysis...")

    batch_serialized_data = ""

    print("\n" + "=" * 50)
    print("📰 HABER OKUMA DENETIMI (Yapay Zekaya Giden Veri)")
    print("=" * 50)

    for index, row in final_analysis_list.iterrows():
        symbol = row["Asset"]
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            news_titles = [h.get('title') or (h.get('content', {}).get('title') if isinstance(h.get('content'), dict) else "") for h in news[:2]] if news else []
            news_text = " | ".join([t for t in news_titles if t]) if news_titles else "No news."
        except Exception:
            news_text = "No news."

        print(f"[{symbol}] Haberleri: {news_text}")

        batch_serialized_data += f"- Asset: {symbol}, Category: {row['Category']}, Trend: {row['Absolute Trend']}, {MOMENTUM_WEEKS}W Return: {row[MOMENTUM_COL]}%, Volume: {row['Volume Status']}, News: {news_text}\n"

    print("=" * 50 + "\n")

    batch_prompt = f"""
    You are an elite hedge fund manager running a dynamic strategy (positions reviewed Monday and Friday).
    Global context for today: {macro_note}

    Analyze the following {len(final_analysis_list)} assets simultaneously.
    Assets Dataset:
    {batch_serialized_data}

    CRITICAL INSTRUCTIONS (FOLLOW EXACTLY):
    1. You MUST start your response for EVERY SINGLE asset with exactly one of these five tags:
       [STRONG BUY 🚀], [ACCUMULATE 🟢], [HOLD 🟡], [TRIM 🟠], or [SELL 🔴].
    2. After the tag, provide a maximum 15-word justification. 
    3. RULE for TRIM: If {MOMENTUM_WEEKS}W Return is high ({TRIM_MOMENTUM_THRESHOLD}%+) BUT news is mixed/bad OR volume is 'Decreasing', you MUST use [TRIM 🟠] to lock in profits.
    4. RULE for CORE ASSETS: If the Category is 'Core Foundation', strongly lean towards [ACCUMULATE 🟢] or [HOLD 🟡]. Never use [SELL 🔴] unless macro news is catastrophic; use [TRIM 🟠] if risk is elevated.
    
    CRITICAL JSON RULE: DO NOT use ANY quotation marks (single ' or double ") anywhere inside the justification text. Write the text cleanly without quotes.
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
        print(f"JSON Parse Error: {e}\nRaw Response (Truncated): {raw_json_response[:200]}")
        for index, row in final_analysis_list.iterrows():
            final_analysis_list.at[index, "AI Action & Risk Warning"] = "Technical Review Required"

    df_display = final_analysis_list.drop(columns=["Volume Status"])
    return df_display


def load_signal_history():
    # YENI: Çökme önleyici esnek CSV okuyucu
    if os.path.exists(HISTORY_FILE):
        try:
            # Önce parse_dates kullanmadan düz okuyoruz
            df = pd.read_csv(HISTORY_FILE)
            
            # Tarih kolonlarını güvenli bir şekilde datetime'a çevir (hata yapanları pas geç)
            date_columns_to_try = ["run_date", "eval_date_1w", "eval_date_4w", "eval_date_1m", "eval_date_3m"]
            for col in date_columns_to_try:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Kodun gerektirdiği kolonlar eksikse NA ile doldur ki tablo şablonu bozulmasın
            for col in HISTORY_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.NA
                    
            return df
        except Exception as e:
            print(f"[Uyari] signals_history bozuk veya okunamadi. Hata: {e}")
            
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def get_latest_price(symbol):
    try:
        data = yf.Ticker(symbol).history(period="5d")
        if not data.empty:
            # YENI: NaN korumasi
            close_series = data['Close'].ffill()
            return float(close_series.iloc[-1])
    except Exception:
        pass
    return None


def update_realized_returns(history_df):
    if history_df.empty:
        return history_df

    today = pd.Timestamp(dt.date.today())
    price_cache = {}

    for idx, row in history_df.iterrows():
        run_date = row.get("run_date")
        if pd.isna(run_date):
            continue

        days_passed = (today - run_date).days
        entry_price = row.get("price")

        needs_1w = days_passed >= EVAL_DAYS_1W and pd.isna(row.get("realized_return_1w"))
        needs_4w = days_passed >= EVAL_DAYS_4W and pd.isna(row.get("realized_return_4w"))

        if not (needs_1w or needs_4w):
            continue

        if row["symbol"] not in price_cache:
            price_cache[row["symbol"]] = get_latest_price(row["symbol"])

        current_price = price_cache[row["symbol"]]
        if current_price is None or not entry_price or pd.isna(entry_price):
            continue

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
    new_df = pd.DataFrame(new_rows)
    return pd.concat([history_df, new_df], ignore_index=True)


def generate_accuracy_summary(history_df):
    def is_hit(row, col):
        ret = row.get(col)
        if pd.isna(ret):
            return None
        
        signal = str(row.get("ai_signal", ""))
        if any(tag in signal for tag in ["SELL", "TRIM"]):
            return ret < 0
        return ret > 0

    if "realized_return_1w" not in history_df.columns:
        return "Gecmis veri formati henuz hazir degil."

    evaluated_1w = history_df.dropna(subset=["realized_return_1w"])
    if evaluated_1w.empty:
        return "Henuz 1 haftasi dolmus/degerlendirilmis sinyal yok."

    evaluated_1w = evaluated_1w.copy()
    evaluated_1w["hit"] = evaluated_1w.apply(lambda r: is_hit(r, "realized_return_1w"), axis=1)
    avg_return_1w = evaluated_1w["realized_return_1w"].mean()
    hit_rate_1w = evaluated_1w["hit"].mean() * 100
    n_1w = len(evaluated_1w)

    summary = f"[1 Hafta] Degerlendirilen Sinyal: {n_1w} | Ort. Getiri: {avg_return_1w:.2f}% | Yon-Dogru Oran: {hit_rate_1w:.1f}%"

    if "realized_return_4w" in history_df.columns:
        evaluated_4w = history_df.dropna(subset=["realized_return_4w"])
        if not evaluated_4w.empty:
            evaluated_4w = evaluated_4w.copy()
            evaluated_4w["hit"] = evaluated_4w.apply(lambda r: is_hit(r, "realized_return_4w"), axis=1)
            avg_return_4w = evaluated_4w["realized_return_4w"].mean()
            hit_rate_4w = evaluated_4w["hit"].mean() * 100
            n_4w = len(evaluated_4w)
            summary += f"\n[4 Hafta] Degerlendirilen Sinyal: {n_4w} | Ort. Getiri: {avg_return_4w:.2f}% | Yon-Dogru Oran: {hit_rate_4w:.1f}%"

    return summary


def send_telegram_message(message):
    print("\n[Telegram] Mesaj gonderimi baslatiliyor...")

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ HATA: Telegram Token veya Chat ID bulunamadi!")
        return

    url = f"[https://api.telegram.org/bot](https://api.telegram.org/bot){TELEGRAM_TOKEN}/sendMessage"

    max_length = 3900
    message_chunks = [message[i:i + max_length] for i in range(0, len(message), max_length)]

    for i, chunk in enumerate(message_chunks):
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk
        }

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                print(f"✅ Rapor Telegram'a basariyla gonderildi! (Parca {i+1}/{len(message_chunks)})")
            else:
                print(f"❌ Telegram API Hatasi (Parca {i+1}): {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Telegram baglanti hatasi: {e}")


def build_full_report(macro_note, final_report_df, accuracy_summary, shock_alerts):
    today_name = "Pazartesi" if dt.date.today().weekday() == 0 else "Cuma"
    rebalance_label = f"{today_name} Rebalance"
    if today_name == "Cuma":
        rebalance_label += " (Hafta Sonu Oncesi Tasfiye Degerlendirmesi)"

    text = "=" * 65 + "\n🌍 ALPHAGUARD GLOBAL STRATEJIK NOT (Gunluk)\n" + "=" * 65 + f"\n{macro_note}\n\n"
    if shock_alerts:
        text += "=" * 65 + "\n⚡ ANLIK SOK UYARILARI\n" + "=" * 65 + "\n" + "\n".join(shock_alerts) + "\n\n"
    text += "=" * 65 + f"\n🏛️ ALPHAGUARD PORTFOY RAPORU ({rebalance_label})\n" + "=" * 65 + "\n" + final_report_df.to_string(index=False)
    text += "\n\n" + "=" * 65 + "\n📊 GECMIS SINYAL PERFORMANSI\n" + "=" * 65 + "\n" + accuracy_summary
    return text


def build_daily_monitor_report(macro_note, shock_alerts):
    text = "=" * 65 + "\n🌍 ALPHAGUARD GUNLUK IZLEME NOTU\n" + "=" * 65 + f"\n{macro_note}\n\n"
    if shock_alerts:
        text += "=" * 65 + "\n⚡ ANLIK SOK UYARILARI\n" + "=" * 65 + "\n" + "\n".join(shock_alerts) + "\n\n"
    else:
        text += "Bugun icin esik-asan bir sok hareket tespit edilmedi.\n\n"
    text += "ℹ️ Not: Portfoy sinyalleri Pazartesi ve Cuma gunleri yeniden hesaplanir; bugun sadece piyasa izlemesi yapildi."
    return text


if __name__ == "__main__":
    try:
        print("\n🚀 ALPHAGUARD SISTEMI BASLATILIYOR (Dinamik Strateji / Gunluk Izleme)...")

        if not API_KEY or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            raise ValueError("KRITIK HATA: API_KEY, TELEGRAM_TOKEN veya TELEGRAM_CHAT_ID bulunamadi!")

        macro_note = global_macro_intelligence()
        shock_alerts = daily_shock_check(CORE_ASSETS)

        if is_rebalance_day():
            today_name = "Pazartesi" if dt.date.today().weekday() == 0 else "Cuma"
            print(f"\n📅 Bugun {today_name} REBALANCE gunu (sinyal yeniden hesaplaniyor)...")

            watchlist = read_portfolio("portfolio.csv")
            watchlist = [s for s in watchlist if s not in CORE_ASSETS] if watchlist else []

            final_report = dual_momentum_and_risk_analysis(watchlist, macro_note)
            pd.set_option('display.max_colwidth', None)

            print("\nStage 3: Sinyal gecmisi guncelleniyor...\n")
            history_df = load_signal_history()
            history_df = update_realized_returns(history_df)
            history_df = append_new_signals(history_df, final_report)
            
            # Guncellenmis yapiyi guvenli sekilde kaydet (eski/gereksiz kolonlari eleyerek)
            cols_to_save = [c for c in history_df.columns if c in HISTORY_COLUMNS or "1m" in c or "3m" in c]
            history_df[cols_to_save].to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
            
            accuracy_summary = generate_accuracy_summary(history_df)

            report_text = build_full_report(macro_note, final_report, accuracy_summary, shock_alerts)
            mark_rebalance_done()
        else:
            print("\n📡 Bugun izleme gunu (rebalance yok, sadece makro + sok taramasi)...")
            report_text = build_daily_monitor_report(macro_note, shock_alerts)

        print(report_text)
        send_telegram_message(report_text)

        print("🏁 SISTEM BASARIYLA TAMAMLANDI!")

    except Exception as e:
        print(f"\n❌ FATAL ERROR (Sistem Coktu): {e}")
        raise e
