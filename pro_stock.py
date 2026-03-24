import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta

# --- 1. 基礎設定 ---
st.set_page_config(page_title="台股 K線預測終端-旗艦版", layout="wide")
st.title("🕯️ 台股互動 K線與美股連動預測")

# --- 2. 完整股票清單 (分類整理) ---
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
    "--- 金融傳產權值 ---": None,
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

# --- 3. 側邊欄控制 ---
st.sidebar.header("⚙️ Setting")
target_name = st.sidebar.selectbox("Select Ticker", list(stocks_dict.keys()))

# 如果選到分類標題就停止
if stocks_dict[target_name] is None:
    st.warning("請選擇有效的股票名稱，而非分類標籤。")
    st.stop()

symbol = stocks_dict[target_name]
period_val = st.sidebar.selectbox("分析週期", ["1y", "2y", "5y"])
predict_days = st.sidebar.slider("未來預測天數", 5, 20, 7)
risk_factor = st.sidebar.slider("風險帶寬度 (Risk Band)", 1.0, 3.0, 1.8)

# --- 4. 核心後端功能 ---
@st.cache_data(ttl=3600)
def get_adr_impact():
    try:
        adr = yf.download("TSM", period="2d", progress=False)
        if isinstance(adr.columns, pd.MultiIndex): adr.columns = adr.columns.get_level_values(0)
        return float(((adr['Close'].iloc[-1] - adr['Close'].iloc[-2]) / adr['Close'].iloc[-2]) * 100)
    except:
        return 0.0

@st.cache_data(ttl=3600)
def load_stock_data(sym, p):
    df = yf.download(sym, period=p, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    return df

def get_forecast(df, days, noise, adr_pc):
    train = df['Close'].dropna().tail(120).values
    try:
        model = ExponentialSmoothing(train, trend='add').fit()
        forecast = model.forecast(days)
        # 美股 ADR 修正因子
        forecast += (adr_pc / 100) * train[-1] * 0.4
        std = np.std(train[-20:]) * noise
        return forecast, forecast-std, forecast+std
    except:
        return None, None, None

# --- 5. 前端 UI 呈現 ---
adr_p = get_adr_impact()
st.info(f"🌐 昨晚台積電 ADR 漲跌: {adr_p:+.2f}% (已自動修正預測起點)")

df = load_stock_data(symbol, period_val)

if df is not None:
    f_p, lo, up = get_forecast(df, predict_days, risk_factor, adr_p)
    f_dates = [df.index[-1] + timedelta(days=i) for i in range(1, predict_days + 1)]

    # 繪製互動式 Plotly 圖表
    fig = go.Figure()

    # K線 (紅漲綠跌)
    fig.add_trace(go.Candlestick(
        x=df.index[-60:], open=df['Open'].tail(60), high=df['High'].tail(60),
        low=df['Low'].tail(60), close=df['Close'].tail(60), name='Candlestick',
        increasing_line_color='red', decreasing_line_color='green'
    ))

    # 均線
    fig.add_trace(go.Scatter(x=df.index[-60:], y=df['MA20'].tail(60), name='MA20', line=dict(color='orange', width=1.2)))
    fig.add_trace(go.Scatter(x=df.index[-60:], y=df['MA60'].tail(60), name='MA60', line=dict(color='purple', width=1.2)))

    # 預測線與陰影
    if f_p is not None:
        fig.add_trace(go.Scatter(x=f_dates, y=f_p, name='Forecast', line=dict(color='blue', dash='dot')))
        fig.add_trace(go.Scatter(
            x=f_dates + f_dates[::-1], y=list(up) + list(lo)[::-1],
            fill='toself', fillcolor='rgba(0,0,255,0.1)', line_color='rgba(255,255,255,0)', name='Risk Band'
        ))

    # --- 修正後的佈局設定 (請確保縮進正確) ---
    fig.update_layout(
        title=f"{target_name} 互動 K線與趨勢預測",
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        height=700, 
        yaxis_title="Price (TWD)",
        legend=dict(
            orientation="h",      # 水平排列
            yanchor="bottom",
            y=-0.25,              # 稍微拉低一點避免重疊
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=150)        # 留白空間
    )

    # 顯示圖表
    st.plotly_chart(fig, use_container_width=True)

    # 數據指標卡
    m1, m2, m3 = st.columns(3)
    cur_p = float(df['Close'].iloc[-1])
    m1.metric("當前價格", f"{cur_p:.2f}")
    m2.metric("預期目標 (n天後)", f"{f_p[-1]:.2f}" if f_p is not None else "N/A")
    m3.metric("月線 (MA20)", f"{df['MA20'].iloc[-1]:.2f}")

    st.write("### 成交量分析")
    st.bar_chart(df['Volume'].tail(60))

else:
    st.error("資料獲取失敗，請檢查網路或股票代號。")
