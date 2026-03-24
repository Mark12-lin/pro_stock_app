import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta

# --- 1. 基礎設定 ---
st.set_page_config(page_title="台股決策儀表板-專業版", layout="wide")
st.title("🛡️ 台股互動 K線與多因子決策終端")

# --- 2. 完整股票清單 ---
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
st.sidebar.header("⚙️ 設定面板")
target_name = st.sidebar.selectbox("選擇標的", list(stocks_dict.keys()))

if stocks_dict[target_name] is None:
    st.warning("請選擇有效股票。")
    st.stop()

symbol = stocks_dict[target_name]
period_val = st.sidebar.selectbox("數據長度", ["1y", "2y", "5y"])
predict_days = st.sidebar.slider("預測天數", 5, 20, 7)
risk_factor = st.sidebar.slider("風險帶寬度", 1.0, 3.0, 1.8)

# --- 4. 運算引擎 ---
@st.cache_data(ttl=3600)
def get_adr():
    try:
        adr = yf.download("TSM", period="2d", progress=False)
        if isinstance(adr.columns, pd.MultiIndex): adr.columns = adr.columns.get_level_values(0)
        return float(((adr['Close'].iloc[-1] - adr['Close'].iloc[-2]) / adr['Close'].iloc[-2]) * 100)
    except: return 0.0

@st.cache_data(ttl=3600)
def load_data(sym, p):
    df = yf.download(sym, period=p, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    return df

def analyze_tech(df):
    # RSI 計算
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # 乖離率
    bias = ((df['Close'].iloc[-1] - df['MA20'].iloc[-1]) / df['MA20'].iloc[-1]) * 100
    # 量能比 (今日 vs 5日均量)
    vol_ratio = df['Volume'].iloc[-1] / df['Volume'].tail(5).mean()
    return rsi.iloc[-1], bias, vol_ratio

def run_forecast(df, days, noise, adr_pc):
    train = df['Close'].dropna().tail(120).values
    model = ExponentialSmoothing(train, trend='add').fit()
    forecast = model.forecast(days) + (adr_pc / 100 * train[-1] * 0.4)
    std = np.std(train[-20:]) * noise
    return forecast, forecast-std, forecast+std

# --- 5. UI 呈現 ---
adr_p = get_adr()
st.info(f"🌐 昨晚台積電 ADR 表現: {adr_p:+.2f}% (已自動修正預測)")

df = load_data(symbol, period_val)

if df is not None:
    f_p, lo, up = run_forecast(df, predict_days, risk_factor, adr_p)
    f_dates = [df.index[-1] + timedelta(days=i) for i in range(1, predict_days + 1)]
    rsi_v, bias_v, vol_v = analyze_tech(df)

    # 繪製 K 線圖
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index[-60:], open=df['Open'].tail(60), high=df['High'].tail(60),
        low=df['Low'].tail(60), close=df['Close'].tail(60), name='K線',
        increasing_line_color='red', decreasing_line_color='green'
    ))
    fig.add_trace(go.Scatter(x=df.index[-60:], y=df['MA20'].tail(60), name='月線(MA20)', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=f_dates, y=f_p, name='預測趨勢', line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(x=f_dates+f_dates[::-1], y=list(up)+list(lo)[::-1], fill='toself', fillcolor='rgba(0,0,255,0.05)', line_color='rgba(255,255,255,0)', name='預測風險帶'))

    fig.update_layout(
        height=600, template="plotly_white", xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(b=100)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 關鍵決策卡片 ---
    st.markdown("### ⚖️ 多因子決策指標")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("RSI (14天強弱)", f"{rsi_v:.1f}")
        if rsi_v > 75: st.error("🔥 市場極度過熱")
        elif rsi_v < 30: st.success("❄️ 進入超跌區")
        else: st.info("🟢 走勢溫和")

    with c2:
        st.metric("月線乖離率", f"{bias_v:+.2f}%")
        if bias_v > 6: st.warning("⚠️ 向上乖離過大")
        elif bias_v < -6: st.warning("⚠️ 向下乖離過大")
        else: st.info("🟢 價格回歸正常")

    with c3:
        st.metric("相對成交量", f"{vol_v:.2f}x")
        if vol_v > 1.5: st.warning("💥 爆量出現")
        else: st.info("⚪ 量能平穩")

    # 底部數據
    st.write("---")
    st.write(f"💡 **模型觀點**：預計 {predict_days} 天後目標價約為 **{f_p[-1]:.2f}**，但請參考下方 RSI 指標避免在過熱時追高。")

else:
    st.error("讀取失敗")
