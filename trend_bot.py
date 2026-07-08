#!/usr/bin/env python
# coding: utf-8

# In[9]:




# In[10]:


import pandas as pd
import yfinance as yf
import warnings
import google.generativeai as genai
import time
import os
import requests

warnings.filterwarnings('ignore')

# --- GEMINI AI & TELEGRAM SETTINGS ---
API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-3.5-flash')

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
    """Secure connection engine fortified against API rate limits."""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip().replace('\n', ' ')
        except Exception as e:
            error_message = str(e).lower()
            if "429" in error_message or "exhausted" in error_message or "quota" in error_message:
                time.sleep(10 * (attempt + 1)) 
            else:
                time.sleep(3)
                
            if attempt == max_retries - 1:
                return "AI Limit/Connection Error (Will resolve in next run)"

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

def ai_risk_and_action_analysis(symbol, trend, momentum_3m, volume_comment):
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
        
        prompt = f"""
        You are an elite hedge fund manager. The status of the asset {symbol} in our portfolio is:
        - Long-Term Trend: {trend}
        - Last 3 Months Return: {momentum_3m:.2f}%
        - Volume (Strength) Status: {volume_comment}
        - Fresh News: {news_text}
        
        Blend all this data. 
        Rule 1: If the return is high (15%+) BUT the news is bad OR the volume status says 'Decreasing (Weak Participation / Reluctance)', give a 'TAKE PROFIT / SELL WARNING ⚠️' and state why the price might drop based on the news or volume.
        Rule 2: If volume is strong and news is good, hold the position ("HOLD 🟢 - [Reason]").
        Be short, clear, and use a maximum of 30 words.
        """
        return secure_ai_query(prompt)
    except Exception:
        return "Monitor Technical Support Level"

def dual_momentum_and_risk_analysis(symbols):
    results = []
    print(f"AlphaGuard AI Initiating...\nStage 1: Calculating Dual Momentum and Trading Volumes for {len(symbols)} assets...\n")
    
    for symbol in symbols:
        try:
            data = yf.download(symbol, period="2y", interval="1mo", progress=False)
            if data.empty or len(data) < 10:
                continue
            
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
                    volume_comment = "Increasing (Institutional Interest / Strong Participation)"
                elif volume_change < -15:
                    volume_comment = "Decreasing (Weak Participation / Reluctance)"
                else:
                    volume_comment = "Stable (Normal Course)"
            except Exception:
                volume_comment = "No Data"

            results.append({
                "Asset": symbol,
                "Price ($)": round(current_price, 2),
                "Absolute Trend": trend,
                "3M Momentum (%)": round(momentum_3m, 2),
                "Volume Status": volume_comment,
                "AI Action & Risk Warning": "---"
            })
        except Exception:
            pass
            
    df = pd.DataFrame(results)
    df = df.sort_values(by="3M Momentum (%)", ascending=False).reset_index(drop=True)
    
    uptrend_assets = df[df["Absolute Trend"] == "UPTREND 🟢"]
    top_15_leaders = uptrend_assets.head(15)
    
    macro_note = global_macro_intelligence()
    
    print(f"\nStage 2: Top 15 leading assets selected. Initiating AI Volume and News Analysis...\n")
    for index, row in top_15_leaders.iterrows():
        symbol = row["Asset"]
        trend = row["Absolute Trend"]
        momentum = row["3M Momentum (%)"]
        vol = row["Volume Status"]
        
        print(f">> Sending {symbol} to AI (Volume: {vol})...")
        ai_signal = ai_risk_and_action_analysis(symbol, trend, momentum, vol)
        df.at[index, "AI Action & Risk Warning"] = ai_signal
        
        time.sleep(4.5) 
        
    df_display = df.drop(columns=["Volume Status"])
    return df_display, macro_note

def send_telegram_message(message):
    """Sends the compiled report to the user via Telegram bot."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram configuration missing. Printing to console only.")
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
        final_report, macro_note = dual_momentum_and_risk_analysis(watchlist)
        pd.set_option('display.max_colwidth', None)
        
        report_text = "=" * 65 + "\n"
        report_text += "🌍 ALPHAGUARD GLOBAL STRATEGIC TACTICAL NOTE\n"
        report_text += "=" * 65 + "\n"
        report_text += f"{macro_note}\n\n"
        report_text += "=" * 65 + "\n"
        report_text += "🏆 ALPHAGUARD AI - RISK/SELL WARNING ENGINE (TOP 15)\n"
        report_text += "=" * 65 + "\n"
        report_text += final_report.head(20).to_string(index=False)
        
        print(report_text)
        send_telegram_message(report_text)
