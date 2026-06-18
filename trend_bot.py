#!/usr/bin/env python
# coding: utf-8

# In[9]:




# In[10]:


import os
from dotenv import load_dotenv

# .env dosyasındaki gizli bilgileri okur
load_dotenv()

# --- CONFIGURATION (AYARLAR) ---
TELEGRAM_TOKEN = os.getenv("8832252707:AAFG0qh4txKDyzFDnCg4Nb-4KwGjsAfARMY")
TELEGRAM_CHAT_ID = os.getenv("6539033974")


# In[11]:


import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION (AYARLAR) ---
# GitHub'a yüklemeden önce burayı temizleyeceğiz veya güvenli hale getireceğiz!
TELEGRAM_TOKEN = "8832252707:AAFG0qh4txKDyzFDnCg4Nb-4KwGjsAfARMY"
TELEGRAM_CHAT_ID = "6539033974"

ASSETS = {
    'Bitcoin': 'BTC-USD',
    'S&P 500': '^GSPC',
    'Altın': 'GC=F'
}

def send_telegram_message(message):
    """Hazırlanan raporu Telegram üzerinden telefona gönderir."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Bildirim Telegram'a başarıyla gönderildi.")
        else:
            print(f"❌ Telegram Hatası: {response.text}")
    except Exception as e:
        print(f"❌ Bağlantı Hatası: {e}")

def analyze_trends():
    """Varlıkları analiz eder ve tek bir metin raporu oluşturur."""
    report = f"📊 *HAFTALIK MAKRO TREND RAPORU*\n"
    report += f"🗓 Tarih: {datetime.now().strftime('%Y-%m-%d')}\n"
    report += f"Strateji: Meb Faber 10 Aylık SMA\n"
    report += "────────────────────────\n\n"

    for name, ticker in ASSETS.items():
        try:
            # 2 yıllık aylık veri çekme
            data = yf.download(ticker, period="2y", interval="1mo", progress=False)
            closed_months = data.copy()[:-1]

            # 10 Aylık SMA Hesaplama
            closed_months['10_SMA'] = closed_months['Close'].rolling(window=10).mean()

            last_closed = closed_months.iloc[-1]
            current_price = float(data.iloc[-1]['Close'].iloc[0]) if isinstance(data.iloc[-1]['Close'], pd.Series) else float(data.iloc[-1]['Close'])
            sma_10 = float(last_closed['10_SMA'].iloc[0]) if isinstance(last_closed['10_SMA'], pd.Series) else float(last_closed['10_SMA'])
            close_price = float(last_closed['Close'].iloc[0]) if isinstance(last_closed['Close'], pd.Series) else float(last_closed['Close'])

            # Trend Analizi
            if close_price > sma_10:
                trend_str = "YÜKSELİŞ 🟢"
            else:
                trend_str = "DÜŞÜŞ 🔴"

            fark_yuzde = ((current_price - sma_10) / sma_10) * 100

            # Rapora ekleme yapıyoruz (Markdown formatında)
            report += f"🔸 *{name} ({ticker})*\n"
            report += f"  • Durum: {trend_str}\n"
            report += f"  • Güncel Fiyat: {current_price:,.2f}\n"
            report += f"  • 10 Aylık SMA: {sma_10:,.2f}\n"
            report += f"  • SMA'ya Uzaklık: %{fark_yuzde:.2f}\n\n"

        except Exception as e:
            report += f"❌ *{name}* analiz edilirken hata oluştu.\n\n"

    return report

# --- ÇALIŞTIRMA ---
if __name__ == "__main__":
    # 1. Analizi yap ve raporu oluştur
    final_report = analyze_trends()

    # 2. Ekrana yazdır (Notebook içinde görmek için)
    print(final_report)

    # 3. Telefona gönder!pip install python-dotenv
    send_telegram_message(final_report)


# In[12]:


get_ipython().system('pip install python-dotenv')

