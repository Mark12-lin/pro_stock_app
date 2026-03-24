import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta
import matplotlib

# --- 1. 基礎設定 ---
st.set_page_config(page_title="台股行動分析終端-連動版", layout="wide")
st.title("🛡️ 台股專業預測終端 (美股連動實戰版)")

# 擴充股票清單 (包含半導體、金融、重電、AI伺服器)
stocks = {
    "台積電 (2330)": "2330.TW", "元大台灣50 (0050)": "0050.TW", "鴻海 (2317)": "2317.TW",
    "聯發科 (2454)": "2454.TW", "廣達 (2382)": "2382.TW", "長榮 (2603)": "2603.TW",
    "世芯-KY (3661)": "3661.TW", "台達電 (2308)": "2308.TW", "日月光 (3711)": "3711.TW",
    "富邦金 (2881)": "2881.TW", "國泰金 (2882)": "2882.TW", "中信金 (2891)": "2891.TW",
    "兆豐金 (2886)": "2886.TW", "玉山金 (2884)": "2884.TW", "中鋼 (2002)": "2002.TW",
    "緯穎 (6669)": "6669.TW", "技嘉 (2376)": "2376.TW", "創意 (3443)": "3443.TW",
    "聯電 (2303)": "2303.TW", "奇鋐 (3017)": "3017.TW", "華城 (1519)": "1519.TW",
    "士電 (1503)": "1503.TW"
}

# --- 2. 側邊欄控制 ---
st.sidebar.header("⚙️ 參數設定")
target_name = st.sidebar.selectbox("選擇分析標的", list(stocks.keys()))
symbol = stocks[target_name]
predict_days = st.sidebar.slider("預測未來天數", 5, 20, 7)
noise_level = st.sidebar.slider("謹慎係數 (陰影寬度)", 1.0, 3.0, 1.8)

# --- 3. 核心功能：抓取美股連動資料 ---
@st.cache_data(ttl=3600)
def get_us_market_status():
    try:
        # 抓取台積電 ADR (TSM) 與費城半導體 (^SOX)
        adr = yf.download("TSM", period="2d", progress=False)
        sox = yf.download("^SOX", period="2d", progress=False)
        
        # 處理多層索引
        if isinstance(adr.columns, pd.MultiIndex): adr.columns = adr.columns.get_level_values(0)
        if isinstance(sox.columns, pd.MultiIndex): sox.columns = sox.columns.get_level_values(0)
        
        adr_pc = ((adr['Close'].iloc[-1] - adr['Close'].iloc[-2]) / adr['Close'].iloc[-2]) * 100
        sox_pc = ((sox['Close'].iloc[-1] - sox['Close'].iloc[-2]) / sox['Close'].iloc[-2]) * 100
        return float(adr_pc), float(sox_pc)
    except:
        return 0.0, 0.0

# --- 4. 數據載入與預測模型 ---
@st.cache_data(ttl=3600)
def load_data(sym):
    df = yf.download(sym, period="2y", progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df['MA20'] = df['Close'].rolling(20).mean()
    return df

def get_pro_prediction(df, days, noise, adr_impact):
    data = df['Close'].tail(120).values
    try:
        # Holt-Winters 模型
        model = ExponentialSmoothing(data, trend='add', seasonal=None).fit()
        forecast = model.forecast(days)
        
        # 加入美股修正因子 (影響預測起點)
        correction = (adr_impact / 100) * data[-1] * 0.5 # 假設影響權重為 50%
        forecast = forecast + correction
        
        # 建立謹慎區間 (標準差)
        std_dev = np.std(data[-20:]) * noise
        lower_bound = forecast - std_dev
        upper_bound = forecast + std_dev
        
        last_date = df.index[-1]
        future_dates = [pd.to_datetime(last_date).date() + timedelta(days=i) for i in range(1, days + 1)]
        return future_dates, forecast, lower_bound, upper_bound
    except:
        return None, None, None, None

# --- 5. UI 呈現 ---
adr_change, sox_change = get_us_market_status()

# 顯示即時美股警告
st.subheader("🌐 國際市場連動監測")
c_adr, c_sox = st.columns(2)
with c_adr:
    st.metric("台積電 ADR (昨晚)", f"{adr_change:+.2f}%")
    if adr_change > 1.5: st.info("🔥 ADR 強勢，今日開盤有利")
    elif adr_change < -1.5: st.warning("❄️ ADR 疲弱，留意開盤壓力")

with c_sox:
    st.metric("費城半導體 (昨晚)", f"{sox_change:+.2f}%")

df = load_data(symbol)

if df is not None and len(df) > 100:
    f_dates, f_preds, low_b, up_b = get_pro_prediction(df, predict_days, noise_level, adr_change)
    
    if f_dates:
        # 指標顯示
        st.markdown("---")
        cur_p = float(df['Close'].iloc[-1])
        diff_p = f_preds[-1] - cur_p
        
        st.subheader(f"📈 {target_name} 趨勢預測 (已整合美股修正)")
        col1, col2, col3 = st.columns(3)
        col1.metric("目前價格", f"{cur_p:.2f}")
        col2.metric("目標價 (預估)", f"{f_preds[-1]:.2f}", f"{diff_p:+.2f}")
        col3.metric("風險下限", f"{low_b[-1]:.2f}", delta_color="inverse")

        # 圖表繪製
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index[-60:], df['Close'].tail(60), label='歷史股價', color='#2c3e50')
        ax.plot(f_dates, f_preds, 'r-o', markersize=4, label='模型預估 (含美股修正)')
        ax.fill_between(f_dates, low_b, up_b, color='red', alpha=0.1, label='謹慎波動區間')
        ax.axhline(y=cur_p, color='gray', linestyle=':', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
        
        st.caption("註：本模型參考前一晚美股漲跌進行權重修正。預測線僅供參考，請結合當日成交量判斷。")
else:
    st.error("無法取得數據，請檢查網路或股票代碼。")
