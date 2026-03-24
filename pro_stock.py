import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta
import matplotlib

# --- 1. 基礎設定 ---
st.set_page_config(page_title="台股行動分析終端-專業版", layout="wide")
st.title("🛡️ 台股專業預測終端 (高階建模版)")

# 股票清單擴充
stocks = {
    "台積電 (2330)": "2330.TW", "元大台灣50 (0050)": "0050.TW", "鴻海 (2317)": "2317.TW",
    "聯發科 (2454)": "2454.TW", "廣達 (2382)": "2382.TW", "長榮 (2603)": "2603.TW",
    "世芯-KY (3661)": "3661.TW", "台達電 (2308)": "2308.TW", "日月光 (3711)": "3711.TW",
    "富邦金 (2881)": "2881.TW", "國泰金 (2882)": "2882.TW", "中信金 (2891)": "2891.TW",
    "兆豐金 (2886)": "2886.TW", "玉山金 (2884)": "2884.TW", "元大金 (2885)": "2885.TW",
    "台塑 (1301)": "1301.TW", "南亞 (1303)": "1303.TW", "中鋼 (2002)": "2002.TW",
    "緯穎 (6669)": "6669.TW", "技嘉 (2376)": "2376.TW", "創意 (3443)": "3443.TW",
    "智邦 (2345)": "2345.TW", "聯電 (2303)": "2303.TW", "奇鋐 (3017)": "3017.TW",
    "材料-KY (4763)": "4763.TW", "華城 (1519)": "1519.TW", "士電 (1503)": "1503.TW"
}

# --- 2. 側邊欄 ---
st.sidebar.header("⚙️ 專業參數設定")
target_name = st.sidebar.selectbox("選擇分析標的", list(stocks.keys()))
symbol = stocks[target_name]
predict_days = st.sidebar.slider("預測天數", 5, 20, 7)
confidence_level = st.sidebar.slider("謹慎係數 (預估偏差值)", 1.0, 3.0, 2.0)

# --- 3. 核心函數：Holt-Winters 指數平滑模型 (比線性回歸更穩健) ---
@st.cache_data(ttl=3600)
def load_data(sym):
    df = yf.download(sym, period="2y", progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    return df

def get_robust_prediction(df, days, noise_factor):
    data = df['Close'].tail(120).values
    # 使用 Holt-Winters 模型捕捉趨勢與季節性
    try:
        model = ExponentialSmoothing(data, trend='add', seasonal=None).fit()
        forecast = model.forecast(days)
        
        # 計算標準差來建立「謹慎區間」
        std_dev = np.std(data[-20:]) * noise_factor
        lower_bound = forecast - std_dev
        upper_bound = forecast + std_dev
        
        last_date = df.index[-1]
        future_dates = [pd.to_datetime(last_date).date() + timedelta(days=i) for i in range(1, days + 1)]
        return future_dates, forecast, lower_bound, upper_bound
    except:
        return None, None, None, None

# --- 4. 執行與顯示 ---
df = load_data(symbol)

if df is not None and len(df) > 100:
    f_dates, f_preds, low_b, up_b = get_robust_prediction(df, predict_days, confidence_level)
    
    if f_dates:
        last_p = float(df['Close'].iloc[-1])
        st.subheader(f"📊 {target_name} 趨勢診斷")
        
        # 視覺化指標
        c1, c2, c3 = st.columns(3)
        c1.metric("目前價格", f"{last_p:.2f}")
        c2.metric("預期目標", f"{f_preds[-1]:.2f}")
        c3.metric("謹慎區間下限", f"{low_b[-1]:.2f}", delta_color="inverse")

        # 圖表
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 繪製歷史
        ax1.plot(df.index[-90:], df['Close'].tail(90), label='歷史股價', color='#2c3e50', linewidth=2)
        ax1.plot(df.index[-90:], df['MA20'].tail(90), label='MA20', color='#f39c12', linestyle='--')
        
        # 繪製預測與陰影 (謹慎區間)
        ax1.plot(f_dates, f_preds, color='red', marker='o', label='模型預估值')
        ax1.fill_between(f_dates, low_b, up_b, color='red', alpha=0.15, label='謹慎波動範圍')
        
        ax1.set_title(f"{target_name} 多因子趨勢預測", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        st.pyplot(fig)
        
        st.info("💡 說明：紅色陰影區代表市場可能的波動範圍。若預測線斜率平緩且陰影區變大，代表目前方向不明確，應謹慎操作。")

else:
    st.error("資料獲取失敗或數據量不足。")
