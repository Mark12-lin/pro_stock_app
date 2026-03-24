import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta
import matplotlib

# --- 1. 基礎設定與字體處理 ---
st.set_page_config(page_title="台股行動全能終端", layout="wide")
st.title("🛡️ 台股專業全能預測 (含週期切換與美股連動)")

# 解決 Linux 伺服器亂碼：優先使用系統字體，若失敗則不強制指定特定中文
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False 

# 股票清單
stocks = {
    "台積電 (2330)": "2330.TW", "元大台灣50 (0050)": "0050.TW", "鴻海 (2317)": "2317.TW",
    "聯發科 (2454)": "2454.TW", "廣達 (2382)": "2382.TW", "長榮 (2603)": "2603.TW",
    "世芯-KY (3661)": "3661.TW", "緯穎 (6669)": "6669.TW", "華城 (1519)": "1519.TW"
}

# --- 2. 側邊欄控制 (找回週期切換) ---
st.sidebar.header("⚙️ 參數設定")
target_name = st.sidebar.selectbox("選擇股票", list(stocks.keys()))
symbol = stocks[target_name]

# 找回你的時線、周線、月線功能
period_label = st.sidebar.selectbox("分析週期 (History)", ["日線 (1年)", "週線 (2年)", "月線 (5年)", "長期 (Max)"])
period_map = {"日線 (1年)": "1y", "週線 (2年)": "2y", "月線 (5年)": "5y", "長期 (Max)": "max"}
selected_period = period_map[period_label]

predict_days = st.sidebar.slider("未來預測天數", 5, 20, 7)
noise_level = st.sidebar.slider("謹慎係數 (陰影)", 1.0, 3.0, 1.8)

# --- 3. 核心功能：美股連動 ---
@st.cache_data(ttl=3600)
def get_us_impact():
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

# --- 4. 數據載入與專業建模 ---
@st.cache_data(ttl=3600)
def load_stock_data(sym, p):
    df = yf.download(sym, period=p, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    return df

def run_prediction(df, days, noise, adr_pc):
    # 取最近數據建模
    train_data = df['Close'].dropna().tail(100).values
    try:
        model = ExponentialSmoothing(train_data, trend='add').fit()
        forecast = model.forecast(days)
        # 美股修正因子 (影響 40% 的權重)
        forecast += (adr_pc / 100) * train_data[-1] * 0.4
        
        std = np.std(train_data[-20:]) * noise
        return forecast, forecast-std, forecast+std
    except:
        return None, None, None

# --- 5. UI 呈現 ---
adr_change, sox_change = get_us_impact()

st.subheader("🌐 全球連動：昨晚美股表現")
c1, c2 = st.columns(2)
c1.metric("TSM ADR", f"{adr_change:+.2f}%")
c2.metric("PHLX Semiconductor (^SOX)", f"{sox_change:+.2f}%")

df = load_stock_data(symbol, selected_period)

if df is not None:
    f_preds, low_b, up_b = run_prediction(df, predict_days, noise_level, adr_change)
    
    # 建立預測日期
    last_date = df.index[-1]
    f_dates = [pd.to_datetime(last_date).date() + timedelta(days=i) for i in range(1, predict_days + 1)]

    # 數據看板
    st.markdown("---")
    cur_p = float(df['Close'].iloc[-1])
    st.subheader(f"📊 {target_name} - {period_label} 趨勢預測")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Price", f"{cur_p:.2f}")
    m2.metric("MA20 (Monthly)", f"{df['MA20'].iloc[-1]:.2f}")
    m3.metric("MA60 (Quarterly)", f"{df['MA60'].iloc[-1]:.2f}")

    # 圖表
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index[-90:], df['Close'].tail(90), label='History', color='#2c3e50', linewidth=1.5)
    if f_preds is not None:
        ax.plot(f_dates, f_preds, 'r-o', markersize=4, label='Model Forecast')
        ax.fill_between(f_dates, low_b, up_b, color='red', alpha=0.1, label='Confidence Band')
    
    ax.set_title(f"{symbol} Trend Analysis", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)

    # 成交量
    st.write("成交量 (Volume)")
    st.bar_chart(df['Volume'].tail(60))
else:
    st.error("無法抓取資料，請確認代碼或稍後再試。")
