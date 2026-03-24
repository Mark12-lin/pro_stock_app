import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# 頁面設定
st.set_page_config(page_title="台股行動分析終端", layout="wide")
st.title("🚀 台股專業預測終端 (行動雲端版)")

# 股票清單 (包含你關注的 0050, 2330 等)
stocks = {
    "台積電 (2330)": "2330.TW", "元大台灣50 (0050)": "0050.TW", 
    "鴻海 (2317)": "2317.TW", "聯發科 (2454)": "2454.TW",
    "長榮 (2603)": "2603.TW", "世芯-KY (3661)": "3661.TW"
}

# --- 側邊欄控制 ---
st.sidebar.header("設定面板")
target_name = st.sidebar.selectbox("選擇分析標的", list(stocks.keys()))
symbol = stocks[target_name]
predict_days = st.sidebar.slider("預測未來天數", 5, 20, 10)

# --- 資料抓取與快取 (設定1小時過期，確保每日更新) ---
@st.cache_data(ttl=3600)
def load_data(sym):
    df = yf.download(sym, period="1y")
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    return df

df = load_data(symbol)

# --- 漲跌幅預測邏輯 (線性回歸) ---
def get_prediction(df, days):
    # 使用最近 30 天的資料來預測趨勢
    recent_df = df.tail(30).copy()
    X = np.array(range(len(recent_df))).reshape(-1, 1)
    y = recent_df['Close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 預測未來
    future_X = np.array(range(len(recent_df), len(recent_df) + days)).reshape(-1, 1)
    future_preds = model.predict(future_X)
    
    # 建立時間軸
    last_date = recent_df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    return future_dates, future_preds

future_dates, future_preds = get_prediction(df, predict_days)

# --- UI 顯示：智慧推薦 ---
last_price = float(df['Close'].iloc[-1])
pred_final = future_preds[-1]
diff = ((pred_final - last_price) / last_price) * 100

st.subheader("💡 系統診斷報告")
col_a, col_b = st.columns(2)
with col_a:
    if diff > 0:
        st.success(f"預測趨勢：看多 ↗️ (預計 {predict_days} 天後漲幅約 {diff:.2f}%)")
    else:
        st.error(f"預測趨勢：偏空 ↘️ (預計 {predict_days} 天後跌幅約 {abs(diff):.2f}%)")

# --- 圖表區：股價 + 成交量 + 預測線 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=False)

# 主圖：歷史與預測
ax1.plot(df.index[-60:], df['Close'].tail(60), label='歷史收盤價', color='#1f77b4', linewidth=2)
ax1.plot(df.index[-60:], df['MA20'].tail(60), label='MA20 (月線)', color='orange', linestyle='--')
ax1.plot(future_dates, future_preds, label='趨勢預測線', color='red', linestyle=':', marker='o')
ax1.set_title(f"{target_name} 趨勢預測圖", fontsize=16)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 副圖：成交量
colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(-60, 0)]
ax2.bar(df.index[-60:], df['Volume'].tail(60), color=colors)
ax2.set_ylabel("成交量")

st.pyplot(fig)

# --- 數據面板 ---
st.write("### 核心指標")
c1, c2, c3 = st.columns(3)
c1.metric("目前股價", f"{last_price:.2f}", f"{last_price - df['Close'].iloc[-2]:.2f}")
c2.metric("預估目標價", f"{pred_final:.2f}")
c3.metric("RSI (14)", f"{((df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / df['Close'].diff().abs().rolling(14).mean()) * 100).iloc[-1]:.1f}")