import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta
import matplotlib

# --- 1. 基礎設定 ---
st.set_page_config(page_title="台股行動全能終端-旗艦版", layout="wide")
st.title("🛡️ 台股專業全能預測 (50+ 權值股擴充版)")

# 解決 Linux 伺服器亂碼：使用英文標示核心資訊
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False 

# --- 2. 擴充股票清單 (分類整理) ---
stocks_dict = {
    "--- 指數與熱門 ETF ---": None,
    "元大台灣50 (0050)": "0050.TW",
    "元大高股息 (0056)": "0056.TW",
    "國泰永續高股息 (00878)": "00878.TW",
    "群益台灣精選高息 (00919)": "00919.TW",
    "復華台灣科技優息 (00929)": "00929.TW",
    "--- 半導體與 AI 族群 ---": None,
    "台積電 (2330)": "2330.TW",
    "聯發科 (2454)": "2454.TW",
    "鴻海 (2317)": "2317.TW",
    "廣達 (2382)": "2382.TW",
    "緯穎 (6669)": "6669.TW",
    "緯創 (3231)": "3231.TW",
    "技嘉 (2376)": "2376.TW",
    "世芯-KY (3661)": "3661.TW",
    "創意 (3443)": "3443.TW",
    "日月光投控 (3711)": "3711.TW",
    "聯電 (2303)": "2303.TW",
    "台達電 (2308)": "2308.TW",
    "智邦 (2345)": "2345.TW",
    "奇鋐 (3017)": "3017.TW",
    "雙鴻 (3324)": "3324.TW",
    "--- 金強傳產權值 ---": None,
    "富邦金 (2881)": "2881.TW",
    "國泰金 (2882)": "2882.TW",
    "中信金 (2891)": "2891.TW",
    "兆豐金 (2886)": "2886.TW",
    "玉山金 (2884)": "2884.TW",
    "元大金 (2885)": "2885.TW",
    "第一金 (2892)": "2892.TW",
    "合庫金 (5880)": "5880.TW",
    "台新金 (2887)": "2887.TW",
    "長榮 (2603)": "2603.TW",
    "陽明 (2609)": "2609.TW",
    "萬海 (2615)": "2615.TW",
    "中鋼 (2002)": "2002.TW",
    "台泥 (1101)": "1101.TW",
    "台塑 (1301)": "1301.TW",
    "南亞 (1303)": "1303.TW",
    "台化 (1326)": "1326.TW",
    "--- 重電與綠能 ---": None,
    "華城 (1519)": "1519.TW",
    "士電 (1503)": "1503.TW",
    "中興電 (1513)": "1513.TW",
    "亞力 (1514)": "1514.TW"
}

# 移除分類標籤以利抓取資料
actual_stocks = {k: v for k, v in stocks_dict.items() if v is not None}

# --- 3. 側邊欄控制 ---
st.sidebar.header("⚙️ Setting")
target_name = st.sidebar.selectbox("Select Ticker", list(stocks_dict.keys()))

if stocks_dict[target_name] is None:
    st.warning("Please select a valid stock name, not a category header.")
    st.stop()

symbol = stocks_dict[target_name]

# 週期與預測設定
period_label = st.sidebar.selectbox("Timeframe", ["Daily (1y)", "Weekly (2y)", "Monthly (5y)"])
period_map = {"Daily (1y)": "1y", "Weekly (2y)": "2y", "Monthly (5y)": "5y"}
selected_period = period_map[period_label]

predict_days = st.sidebar.slider("Forecast Days", 5, 20, 7)
noise_level = st.sidebar.slider("Risk Band (Width)", 1.0, 3.0, 1.8)

# --- 4. 核心功能：美股與資料載入 ---
@st.cache_data(ttl=3600)
def get_global_data():
    try:
        adr = yf.download("TSM", period="2d", progress=False)
        sox = yf.download("^SOX", period="2d", progress=False)
        if isinstance(adr.columns, pd.MultiIndex): adr.columns = adr.columns.get_level_values(0)
        if isinstance(sox.columns, pd.MultiIndex): sox.columns = sox.columns.get_level_values(0)
        adr_pc = ((adr['Close'].iloc[-1] - adr['Close'].iloc[-2]) / adr['Close'].iloc[-2]) * 100
        sox_pc = ((sox['Close'].iloc[-1] - sox['Close'].iloc[-2]) / sox['Close'].iloc[-2]) * 100
        return float(adr_pc), float(sox_pc)
    except:
        return 0.0, 0.0

@st.cache_data(ttl=3600)
def load_data(sym, p):
    df = yf.download(sym, period=p, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    return df

def predict_logic(df, days, noise, adr_impact):
    train = df['Close'].dropna().tail(100).values
    try:
        model = ExponentialSmoothing(train, trend='add').fit()
        forecast = model.forecast(days)
        # 美股修正因子 (0.4 權重)
        forecast += (adr_impact / 100) * train[-1] * 0.4
        std = np.std(train[-20:]) * noise
        return forecast, forecast-std, forecast+std
    except:
        return None, None, None

# --- 5. UI 顯示 ---
adr_p, sox_p = get_global_data()
st.subheader("🌐 Global Market Impact")
c1, c2 = st.columns(2)
c1.metric("TSM ADR (US)", f"{adr_p:+.2f}%")
c2.metric("PHLX Semi (^SOX)", f"{sox_p:+.2f}%")

df = load_data(symbol, selected_period)

if df is not None:
    f_p, lo_b, up_b = predict_logic(df, predict_days, noise_level, adr_p)
    last_date = df.index[-1]
    f_dates = [pd.to_datetime(last_date).date() + timedelta(days=i) for i in range(1, predict_days + 1)]

    st.markdown("---")
    cur_p = float(df['Close'].iloc[-1])
    st.subheader(f"📊 {target_name} ({period_label})")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Price", f"{cur_p:.2f}")
    m2.metric("Target (Forecast)", f"{f_p[-1]:.2f}" if f_p is not None else "N/A")
    m3.metric("MA20", f"{df['MA20'].iloc[-1]:.2f}")

    # 圖表
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index[-80:], df['Close'].tail(80), label='History', color='#2c3e50', linewidth=1.5)
    if f_p is not None:
        ax.plot(f_dates, f_p, 'r-o', markersize=4, label='Forecast (US Adjusted)')
        ax.fill_between(f_dates, lo_b, up_b, color='red', alpha=0.1, label='Risk Band')
    
    ax.set_title(f"Trend Analysis - {symbol}", fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)

    st.write("Volume Analysis")
    st.bar_chart(df['Volume'].tail(60))
else:
    st.error("Data fetch failed. Please check the ticker or interval.")
