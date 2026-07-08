#!/usr/bin/env python
# coding: utf-8

# In[9]:




# In[10]:


import pandas as pd
import yfinance as yf
import warnings
import time
import os
import datetime as dt
import requests
from google import genai
from google.genai import types

warnings.filterwarnings('ignore')

# --- SETTINGS ---
API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Yeni Google GenAI SDK'sı ile istemci başlatılıyor
client = genai.Client(api_key=API_KEY)
MODEL_ID = 'gemini-3.5-flash'

# Sabit Çekirdek Varlıklar (Emlak, Tahvil, Kripto, Altın, Petrol)
CORE_ASSETS = ["O", "BNDW", "BTC-USD", "ZGLD.SW", "SHEL"]

# --- SINYAL GECMISI AYARLARI (Madde 2) ---
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

def secure_ai_query(prompt, max_retries=4):
    """Yeni nesil SDK ile çalışan, hız limitlerine karşı güçlendirilmiş sorgu motoru."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2, # Daha analitik ve tutarlı cevaplar için
                )
            )
            return response.text.strip().replace('\n', ' ')
        except Exception as e:
            error_message = str(e).lower()
            if "429" in error_message or "quota" in error_message or "exhausted" in error_message:
                sleep_time = 15 * (attempt + 1)
                print(f"      [Rate Limit Hit] Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                time.sleep(5)
                
            if attempt == max_retries - 1:
                return "AI Limit/Connection Error"

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
    return secure_ai_query(prompt)

def ai_risk_and_action_analysis(symbol, trend, momentum_3m, volume_comment, is_core=False):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        news_titles = []
        if news:
            for h in news[:3]:
                if isinstance(h, dict):
                    title = h.get('title') or (h.get('content', {}).get('title') if isinstance(h.get('content'), dict) else None)
                    if title:
                        news_titles.append(f"- {title}")
        
        news_text = "\n".join(news_titles) if news_titles else "No fresh news flow."
        
        core_context = "This is a CORE HOLDING (macro foundation) of our portfolio, meant for stability." if is_core else "This is a MOMENTUM ASSET meant for high growth."
        
        prompt = f"""
        You are an elite hedge fund manager. The status of the asset {symbol} in our portfolio is:
        - Portfolio Role: {core_context}
        - Long-Term Trend: {trend}
        - Last 3 Months Return: {momentum_3m:.2f}%
        - Volume (Strength) Status: {volume_comment}
        - Fresh News: {news_text}
        
        Blend all this data. 
        Rule 1: If the return is high (15%+) BUT the news is bad OR the volume status says 'Decreasing', give a 'TAKE PROFIT / SELL WARNING ⚠️'.
        Rule 2: If it's a Core Asset, lean towards 'HOLD 🟢' unless news is catastrophic.
        Be short, clear, and use a maximum of 30 words.
        """
        return secure_ai_query(prompt)
    except Exception:
        return "Monitor Technical Support Level"

def batch_download_data(symbols, period="2y", interval="1mo"):
    """
    Tum sembolleri TEK bir yfinance istegiyle ceker (rate-limit dostu, hizli).
    Basarisiz olursa None doner; cagiran taraf bunu tekil indirmeye (eski yontem) dusurur.
    Bu fonksiyon SADECE veri cekme seklini degistirir; hesaplama mantigina dokunmaz.
    """
    try:
        data = yf.download(
            tickers=symbols,
            period=period,
            interval=interval,
            group_by='ticker',
            threads=True,
            progress=False,
        )
        if data is None or data.empty:
            return None
        return data
    except Exception as e:
        print(f"   [Uyari] Toplu indirme basarisiz oldu, tekil indirmeye dusuluyor. Detay: {e}")
        return None


def analyze_asset_data(symbol, batch_data=None):
    """Tekil bir varlığın momentum ve hacim hesaplamalarını yapar.

    batch_data verilirse (toplu indirmeden gelen tablo), onceden indirilmis veriyi kullanir.
    batch_data icinde sembol bulunamazsa veya bos gelirse, GUVENLIK AGI olarak
    orijinal tekil yf.download cagrisina otomatik olarak geri doner.
    """
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
                    # Tek sembol icin toplu indirme de duz kolonlu donebilir
                    candidate = batch_data.dropna(how='all')
                    if not candidate.empty:
                        data = candidate
            except Exception:
                data = None

        if data is None:
            # GUVENLIK AGI: toplu veride sembol yoksa/bozuksa eski yonteme (tekil indirme) don
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
            
            if prev_3m_vol_avg > 0:
                volume_change = ((last_full_month_vol - prev_3m_vol_avg) / prev_3m_vol_avg) * 100
            else:
                volume_change = 0
            
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
            "3M Momentum (%)": round(momentum_3m, 2),
            "Volume Status": volume_comment,
            "AI Action & Risk Warning": "---"
        }
    except Exception:
        return None

def dual_momentum_and_risk_analysis(symbols):
    results = []
    print(f"AlphaGuard AI Initiating...\nStage 1: Calculating Data for {len(symbols)} Watchlist Assets and Core Portfolio...\n")

    # --- Performans: watchlist + core varliklar icin veriyi TEK istekte cek ---
    all_symbols = list(dict.fromkeys(list(symbols) + CORE_ASSETS))  # sirali, tekrarsiz birlesim
    print(f"   [Toplu Indirme] {len(all_symbols)} sembol icin veri tek istekte cekiliyor...")
    batch_data = batch_download_data(all_symbols)
    
    # 1. Tüm izleme listesini işle
    for symbol in symbols:
        asset_data = analyze_asset_data(symbol, batch_data=batch_data)
        if asset_data:
            results.append(asset_data)
            
    df = pd.DataFrame(results)
    
    # 2. Uydu Varlıkları (Dynamic Top 10) Belirle
    uptrend_assets = df[df["Absolute Trend"] == "UPTREND 🟢"]
    top_10_leaders = uptrend_assets.sort_values(by="3M Momentum (%)", ascending=False).head(10).copy()
    top_10_leaders['Category'] = 'Dynamic Top 10'
    
    # 3. Çekirdek (Core) Varlıkları Hesapla
    core_results = []
    for symbol in CORE_ASSETS:
        asset_data = analyze_asset_data(symbol, batch_data=batch_data)
        if asset_data:
            asset_data['Category'] = 'Core Foundation'
            core_results.append(asset_data)
        else:
            # Veri çekilemezse bile tabloda görünmesi için varsayılan değer
            core_results.append({
                "Asset": symbol, "Price ($)": 0.0, "Absolute Trend": "UNKNOWN", 
                "3M Momentum (%)": 0.0, "Volume Status": "No Data", 
                "AI Action & Risk Warning": "Data Error", "Category": 'Core Foundation'
            })
            
    df_core = pd.DataFrame(core_results)
    
    # 4. Final Listeyi Birleştir
    final_analysis_list = pd.concat([df_core, top_10_leaders], ignore_index=True)
    
    macro_note = global_macro_intelligence()
    
    print(f"\nStage 2: 5 Core + 10 Dynamic Assets Selected. Initiating AI Analysis...\n")
    for index, row in final_analysis_list.iterrows():
        symbol = row["Asset"]
        trend = row["Absolute Trend"]
        momentum = row["3M Momentum (%)"]
        vol = row["Volume Status"]
        is_core = row["Category"] == 'Core Foundation'
        
        print(f">> Sending {symbol} to AI ({row['Category']})...")
        ai_signal = ai_risk_and_action_analysis(symbol, trend, momentum, vol, is_core)
        final_analysis_list.at[index, "AI Action & Risk Warning"] = ai_signal
        
        # Kesin çözüm: 10 saniye uyku. (Dakikada max 6 istek). Bu, ücretsiz limiti asla aşmaz.
        time.sleep(10) 
        
    df_display = final_analysis_list.drop(columns=["Volume Status"])
    return df_display, macro_note

def load_signal_history():
    """signals_history.csv varsa okur, yoksa boş bir tablo döner."""
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(
                HISTORY_FILE,
                parse_dates=["run_date", "eval_date_1m", "eval_date_3m"],
            )
            for col in HISTORY_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.NA
            return df[HISTORY_COLUMNS]
        except Exception as e:
            print(f"[Uyari] signals_history.csv okunamadi, sifirdan olusturuluyor. Detay: {e}")
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def get_latest_price(symbol):
    """Bir sembolun en guncel kapanis fiyatini ceker (gerceklesen getiri hesabi icin)."""
    try:
        data = yf.Ticker(symbol).history(period="5d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except Exception:
        pass
    return None


def update_realized_returns(history_df):
    """Suresi dolan (1 ay / 3 ay) gecmis sinyaller icin gerceklesen getiriyi hesaplar."""
    if history_df.empty:
        return history_df

    today = pd.Timestamp(dt.date.today())
    price_cache = {}

    def _price_for(symbol):
        if symbol not in price_cache:
            price_cache[symbol] = get_latest_price(symbol)
        return price_cache[symbol]

    for idx, row in history_df.iterrows():
        run_date = row["run_date"]
        if pd.isna(run_date):
            continue
        days_passed = (today - run_date).days
        entry_price = row["price"]

        needs_1m = days_passed >= EVAL_DAYS_1M and pd.isna(row.get("realized_return_1m"))
        needs_3m = days_passed >= EVAL_DAYS_3M and pd.isna(row.get("realized_return_3m"))

        if not (needs_1m or needs_3m):
            continue

        current_price = _price_for(row["symbol"])
        if current_price is None or not entry_price:
            continue

        realized_return = round(((current_price - entry_price) / entry_price) * 100, 2)

        if needs_1m:
            history_df.at[idx, "realized_return_1m"] = realized_return
            history_df.at[idx, "eval_date_1m"] = today
        if needs_3m:
            history_df.at[idx, "realized_return_3m"] = realized_return
            history_df.at[idx, "eval_date_3m"] = today

    return history_df


def append_new_signals(history_df, final_analysis_list):
    """Bugunku sinyalleri gecmis tabloya yeni satirlar olarak ekler."""
    today = pd.Timestamp(dt.date.today())
    new_rows = []
    for _, row in final_analysis_list.iterrows():
        new_rows.append({
            "run_date": today,
            "symbol": row["Asset"],
            "category": row["Category"],
            "price": row["Price ($)"],
            "trend": row["Absolute Trend"],
            "momentum_3m": row["3M Momentum (%)"],
            "ai_signal": row["AI Action & Risk Warning"],
            "eval_date_1m": pd.NaT,
            "realized_return_1m": pd.NA,
            "eval_date_3m": pd.NaT,
            "realized_return_3m": pd.NA,
        })
    new_df = pd.DataFrame(new_rows)
    return pd.concat([history_df, new_df], ignore_index=True)


def generate_accuracy_summary(history_df):
    """1 aylik degerlendirmesi tamamlanmis sinyaller icin kisa bir performans ozeti uretir."""
    evaluated = history_df.dropna(subset=["realized_return_1m"])
    if evaluated.empty:
        return "Henuz 1 ayi dolmus/degerlendirilmis sinyal yok (ilk ay boyunca bu bolum bos kalacak)."

    avg_return = evaluated["realized_return_1m"].mean()
    hit_rate = (evaluated["realized_return_1m"] > 0).mean() * 100
    n = len(evaluated)

    sell_mask = evaluated["ai_signal"].str.contains("SELL|TAKE PROFIT", case=False, na=False)
    sell_signals = evaluated[sell_mask]
    sell_avg = sell_signals["realized_return_1m"].mean() if not sell_signals.empty else None

    summary = (
        f"Degerlendirilen Sinyal Sayisi (1A): {n} | "
        f"Ortalama Gerceklesen Getiri: {avg_return:.2f}% | "
        f"Pozitif Getiri Orani: {hit_rate:.1f}%"
    )
    if sell_avg is not None:
        summary += f" | SELL/TAKE PROFIT sinyali sonrasi ort. getiri: {sell_avg:.2f}% ({len(sell_signals)} sinyal)"
    return summary


def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram configuration missing.")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"```\n{message}\n```",
        "parse_mode": "MarkdownV2"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Report successfully sent to Telegram!")
        else:
            print(f"Telegram API Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Telegram connection error: {e}")

if __name__ == "__main__":
    watchlist = read_portfolio("portfolio.csv")
    if watchlist:
        # Çekirdek varlıkları (eğer csv'de yoksa) izleme listesinden temizle ki çiftlenmesin
        watchlist = [s for s in watchlist if s not in CORE_ASSETS]
        
        final_report, macro_note = dual_momentum_and_risk_analysis(watchlist)
        pd.set_option('display.max_colwidth', None)

        # --- Sinyal Gecmisi: onceki sinyalleri degerlendir, bugunkuleri ekle, kaydet ---
        print("\nStage 3: Sinyal gecmisi guncelleniyor (gerceklesen getiriler hesaplaniyor)...\n")
        history_df = load_signal_history()
        history_df = update_realized_returns(history_df)
        history_df = append_new_signals(history_df, final_report)  # final_report 'Category' kolonunu icerir
        history_df.to_csv(HISTORY_FILE, index=False)
        accuracy_summary = generate_accuracy_summary(history_df)

        report_text = "=" * 65 + "\n"
        report_text += "🌍 ALPHAGUARD GLOBAL STRATEGIC TACTICAL NOTE\n"
        report_text += "=" * 65 + "\n"
        report_text += f"{macro_note}\n\n"
        report_text += "=" * 65 + "\n"
        report_text += "🏛️ ALPHAGUARD CORE & SATELLITE PORTFOLIO REPORT\n"
        report_text += "=" * 65 + "\n"
        report_text += final_report.to_string(index=False)
        report_text += "\n\n" + "=" * 65 + "\n"
        report_text += "📊 GECMIS SINYAL PERFORMANSI (1 Aylik)\n"
        report_text += "=" * 65 + "\n"
        report_text += accuracy_summary
        
        print(report_text)
        send_telegram_message(report_text)
