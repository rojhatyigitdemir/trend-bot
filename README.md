AlphaGuard AI: A Quantitative Core-Satellite Portfolio Management System

AlphaGuard AI is an automated, cloud-based system for portfolio management. It combines traditional financial momentum models with macroeconomic text analysis using Large Language Models (LLMs). The main goal is to provide systematic asset allocation and risk management.

## Theoretical Framework

The system uses a hybrid approach. It connects established financial theories with artificial intelligence.

### 1. Core-Satellite Portfolio Theory
The portfolio is divided into two parts:
* **Core Foundation:** Assets held permanently to protect against inflation and systemic risks (Real Estate, Global Bonds, Gold, Oil, and Bitcoin).
* **Satellite Allocation:** A dynamically selected Top 10 list of high-growth assets.

### 2. AI-Enhanced Dual Momentum
The algorithm uses the "Dual Momentum" concept, supported by AI-driven rules:
* **Absolute Momentum:** A 10-month Simple Moving Average (SMA). Assets below this line are labeled as a downtrend. This acts as a protective circuit breaker against long bear markets.
* **Relative Momentum:** Satellite assets are ranked based on their 3-month returns. This ensures capital goes to the best-performing sectors.
* **Volume Validation:** The system compares the latest monthly volume against the previous 3-month average. Price increases without strong volume trigger immediate risk warnings.

### 3. LLM-Based Macro Sentiment
Traditional mathematical models are slow to react to sudden news or shocks. AlphaGuard AI solves this by reading live financial news for both global markets and individual stocks. This text data is processed by Google Gemini.
* **AI Risk Management:** If a stock's momentum is high but the news is bad or volume is dropping, the AI acts as a risk manager. It automatically issues "Take Profit" or "Sell" warnings.

## Technical Infrastructure
* **Serverless and Autonomous:** Runs completely on GitHub Actions without needing a local server.
* **Scheduled Execution:** Analyzes the market daily without human intervention.
* **Batch Processing:** Uses a single, combined JSON prompt to overcome API limits and reduce execution time to seconds.
* **Continuous Backtesting:** Automatically saves past signals to the repository to track its own 1-month and 3-month performance.
* **Real-time Delivery:** Sends analysis, AI logic outputs, and portfolio updates instantly via Telegram.

---
*Disclaimer: This repository is for academic and theoretical research in quantitative finance. It is not financial advice.*
