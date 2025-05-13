import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression
plt.rc('font', family='Malgun Gothic')


# 코카콜라 (KO) 주가 및 배당금 데이터 가져오기
ticker = "KO"
stock = yf.Ticker(ticker)

# 10년치 주가 및 배당금 데이터 가져오기
data = stock.history(period="10y")
dividends = data["Dividends"]
close_prices = data["Close"]

# 데이터 정리
df = pd.DataFrame({"Close Price": close_prices, "Dividends": dividends})
df = df[df["Dividends"] > 0]  # 배당금이 지급된 날만 필터링

# 상관관계 분석
correlation = df.corr()
print("배당금과 주가의 상관계수:\n", correlation)

# 시각화
fig, ax1 = plt.subplots(figsize=(12,6))

ax1.set_xlabel("Date")
ax1.set_ylabel("Close Price (USD)", color="blue")
ax1.plot(df.index, df["Close Price"], color="blue", label="Close Price")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.set_ylabel("Dividends (USD)", color="red")
ax2.bar(df.index, df["Dividends"], color="red", alpha=0.6, label="Dividends")
ax2.tick_params(axis="y", labelcolor="red")

plt.title("코카콜라(KO) 배당금과 주가 비교")
plt.show()

correlation = df.corr()
print("배당금과 주가의 상관계수:\n", correlation)

from sklearn.linear_model import LinearRegression
import numpy as np

# 배당금과 주가 관계 모델링
df.dropna(inplace=True)  # NaN 값 제거
X = df["Dividends"].values.reshape(-1, 1)  # 배당금 (독립 변수)
y = df["Close Price"].values  # 주가 (종속 변수)

model = LinearRegression()
model.fit(X, y)

# 예측값 계산
predictions = model.predict(X)

# 회귀선 시각화
plt.figure(figsize=(10, 5))
plt.scatter(df["Dividends"], df["Close Price"], color="blue", alpha=0.5, label="Actual Data")
plt.plot(df["Dividends"], predictions, color="red", label="Regression Line")
plt.xlabel("Dividends (USD)")
plt.ylabel("Close Price (USD)")
plt.title("배당금과 주가 간의 회귀 분석")
plt.legend()
plt.show()
