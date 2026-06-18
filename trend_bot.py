#!/usr/bin/env python
# coding: utf-8

# In[9]:




# In[10]:


import os
import yfinance as yf
import pandas as pd
import requests
from dotenv import load_dotenv

# .env dosyasındaki gizli bilgileri okur
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Analiz edilecek varlıklar
assets = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "S&P 500 (^GSPC)": "^GSPC",
    "Altın (GC=F)": "GC=F"
}

message = "📊 *Haftalık Makro Trend ve Erken Uyarı Raporu*\n\n"

for name, ticker_str in assets.items():
    # Verileri Yahoo Finance üzerinden güvenli bir şekilde çeker
    ticker = yf.Ticker(ticker_str)
    df = ticker.history(period="3y", interval="1mo")
    df = df.dropna()
    
    # Güncel fiyat
    current_price = float(df['Close'].iloc[-1])
    
    # 10 Aylık SMA Hesaplaması
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    current_sma = float(df['SMA_10'].iloc[-1])
    
    # MACD ve Sinyal Çizgisi Hesaplaması (Pandas ile Matematiksel Modelleme)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    macd_val = float(df['MACD'].iloc[-1])
    signal_val = float(df['Signal'].iloc[-1])
    
    # 1. Katman: Ana Makro Trend Durumu (SMA)
    if current_price > current_sma:
        trend = "YÜKSELİŞ 🟢"
    else:
        trend = "DÜŞÜŞ 🔴"
        
    # 2. Katman: Momentum ve Erken Uyarı Durumu (MACD Kesimi)
    if macd_val > signal_val:
        warning = "POZİTİF (Trend Güçlü) 🔥"
    else:
        warning = "NEGATİF (Trend Zayıflıyor) ⚠️"
        
    # Ortalamaya olan yüzde uzaklık
    distance = ((current_price - current_sma) / current_sma) * 100
    
    # Mesaj metnini yapılandırma
    message += f"🔸 *{name}*\n"
    message += f"• Makro Trend (SMA): {trend}\n"
    message += f"• Erken Uyarı (MACD): {warning}\n"
    message += f"• Güncel Fiyat: ${current_price:,.2f}\n"
    message += f"• 10 Aylık SMA: ${current_sma:,.2f}\n"
    message += f"• SMA'ya Uzaklık: %{distance:+.2f}\n\n"

# Telegram API üzerinden mesaj gönderme işlemi
url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
payload = {
    "chat_id": TELEGRAM_CHAT_ID,
    "text": message,
    "parse_mode": "Markdown"
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    print("✅ Bildirim Telegram'a başarıyla gönderildi.")
else:
    print(f"❌ Hata: {response.text}")

# --- ÇALIŞTIRMA ---
if __name__ == "__main__":
    # 1. Analizi yap ve raporu oluştur
    final_report = analyze_trends()

    # 2. Ekrana yazdır (Notebook içinde görmek için)
    print(final_report)

    # 3. Telefona gönder!pip install python-dotenv
    send_telegram_message(final_report)


# In[12]:

