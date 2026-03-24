import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import matplotlib

# --- 1. 基礎設定 ---
st.set_page_config(page_title="台股行動分析終端", layout="wide")
st.title("🚀 台股專業預測終端 (2026 行動雲端版)")

# 解決中文亂碼 (Streamlit Cloud 環境通常內建支援，若無則顯示英文)
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False 

# 股票清單
stocks = {
    "台積電 (2330)": "2330.TW", 
    "元大台灣50 (0050)": "0050.TW", 
    "鴻海 (2317)": "2317.TW", 
    "聯發科 (2454)": "2454.TW",
    "長榮 (2603)": "2603.TW", 
    "世芯-KY (3661)": "3661.TW",
    "廣達 (2382)": "2382.TW"
}

# --- 2. 側邊欄控制 (必須放在最前面) ---
st.sidebar.header("⚙️ 設定面板")
target_name = st.sidebar.selectbox("選擇分析標的", list(stocks.keys()))
symbol = stocks[target_name]
predict_days = st.sidebar.slider("預測未來天數 (紅點線)", 5, 20, 10)

# --- 3. 核心函數定義 ---
@st.cache_data(ttl=3600)
def load_data(sym):
    try:
        df = yf.download(sym, period="1y", progress=False)
        if df.empty:
            return None
        # 處理 yfinance 可能產生的多層索引 (重要修正)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        return df
    except:
        return None

def get_prediction(df, days):
    # 確保資料乾淨且取最近 30 天
    recent_df = df.dropna(subset=['Close']).tail(30).copy()
    if len(recent_df) < 15: return None, None
    
    # 轉換格式
    X = np.array(range(len(recent_df))).reshape(-1, 1)
    y = recent_df['Close'].values.flatten()
    
    # 建立線性回歸模型
    model = LinearRegression()
    model.fit(X, y)
    
    # 預測未來
    future_X = np.array(range(len(recent_df), len(recent_df) + days)).reshape(-1, 1)
    future_preds = model.predict(future_X)
    
    # 建立日期軸
    last_date = recent_df.index[-1]
    future_dates = [pd.to_datetime(last_date).date() + timedelta(days=i) for i in range(1, days + 1)]
    return future_dates, future_preds

# --- 4. 執行邏輯與 UI 顯示 ---
df = load_data(symbol)

if df is not None and len(df) >= 30:
    # 執行預測
    future_dates, future_preds = get_prediction(df, predict_days)
    
    if future_dates is not None:
        last_price = float(df['Close'].iloc[-1])
        pred_final = future_preds[-1]
        diff = ((pred_final - last_price) / last_price) * 100

        # 頂部診斷報告
        st.subheader("💡 系統診斷與預測趨勢")
        col_a, col_b = st.columns(2)
        with col_a:
            if diff > 0:
                st.success(f"📈 預測趨勢：看多 (預計 {predict_days} 天後目標價約 {pred_final:.2f}, 漲幅 {diff:.2f}%)")
            else:
                st.error(f"📉 預測趨勢：看空 (預計 {predict_days} 天後目標價約 {pred_final:.2f}, 跌幅 {abs(diff):.2f}%)")

        # 圖表顯示
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # 主圖：股價與預測線
        ax1.plot(df.index[-60:], df['Close'].tail(60), label='歷史收盤價', color='#1f77b4', linewidth=2)
        ax1.plot(df.index[-60:], df['MA20'].tail(60), label='MA20 (月線)', color='orange', linestyle='--')
        ax1.plot(future_dates, future_preds, label='趨勢預測線', color='red', linestyle=':', marker='o', markersize=4)
        ax1.set_title(f"{target_name} 趨勢預測圖 (手機橫屏效果更佳)", fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 副圖：成交量 (紅跌綠漲)
        colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(-60, 0)]
        ax2.bar(df.index[-60:], df['Volume'].tail(60), color=colors, alpha=0.7)
        ax2.set_ylabel("成交量")
        
        st.pyplot(fig)

        # 底部指標數據 (大字體)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        change = last_price - float(df['Close'].iloc[-2])
        c1.metric("目前股價", f"{last_price:.2f}", f"{change:.2f}")
        c2.metric("MA20 (月線)", f"{df['MA20'].iloc[-1]:.2f}")
        c3.metric("昨日成交量", f"{int(df['Volume'].iloc[-1]):,}")
    else:
        st.warning("預測模型計算失敗，請更換標的再試。")
else:
    st.error("⚠️ 無法獲取足夠的歷史資料，請確認網路連線或稍後再試。")

st.caption("註：預測線僅供趨勢參考，投資有風險，請謹慎評估。")
