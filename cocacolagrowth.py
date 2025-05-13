import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression
plt.rc('font', family='Malgun Gothic')

# 코카콜라 (KO) 티커 불러오기
ticker = "KO"
stock = yf.Ticker(ticker)

# 최신 기업 정보 가져오기
info = stock.info

# 과거 재무 데이터 가져오기
financials = stock.financials  # 최근 4년치 손익계산서

# 매출 성장률 직접 계산 (최근 2년 비교)
if "Total Revenue" in financials.index:
    recent_revenue = financials.loc["Total Revenue"].dropna()
    if len(recent_revenue) > 1:
        revenue_growth = (recent_revenue.iloc[0] - recent_revenue.iloc[1]) / recent_revenue.iloc[1]
    else:
        revenue_growth = None
else:
    revenue_growth = None

# EPS 성장률 직접 계산
if "Net Income" in financials.index:
    recent_net_income = financials.loc["Net Income"].dropna()
    if len(recent_net_income) > 1:
        eps_growth = (recent_net_income.iloc[0] - recent_net_income.iloc[1]) / recent_net_income.iloc[1]
    else:
        eps_growth = None
else:
    eps_growth = None

# PEG 계산 (PER / EPS 성장률)
per = info.get("trailingPE")
peg = per / eps_growth if eps_growth and per else None

# 성장 가능성 지표 수집
growth_data = {
    "지표": ["매출 성장률 (YoY)", "EPS 성장률 (YoY)", "ROE", "ROA", "부채비율 (D/E)", "이자보상배율", "PER", "PBR", "PEG"],
    "값": [
        revenue_growth,  # 매출 성장률 (직접 계산)
        eps_growth,  # EPS 성장률 (직접 계산)
        info.get("returnOnEquity"),  # ROE
        info.get("returnOnAssets"),  # ROA
        info.get("debtToEquity"),  # 부채비율
        info.get("ebitda") / info.get("totalDebt") if info.get("totalDebt") else None,  # 이자보상배율
        per,  # PER
        info.get("priceToBook"),  # PBR
        peg  # PEG (직접 계산)
    ]
}

# 데이터프레임 변환
growth_df = pd.DataFrame(growth_data)

# 성장 가능성 지표 출력
print("코카콜라(KO) 성장 가능성 지표")
print(growth_df)

