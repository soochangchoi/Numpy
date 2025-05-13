import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression

plt.rc('font', family='Malgun Gothic')


# 코카콜라

# 1. 주식 데이터 다운로드 (코카콜라)
stock = yf.download("KO ", period="10y")

# MultiIndex가 적용된 경우, 컬럼을 단순화
if isinstance(stock.columns, pd.MultiIndex):
    stock.columns = stock.columns.get_level_values(0)  # 첫 번째 레벨의 컬럼 이름만 사용

# 'Close' 열이 없으면 'Adj Close' 사용
if "Close" not in stock.columns:
    print(" 'Close' 열이 없습니다. 'Adj Close'를 대신 사용합니다.")
    stock["Close"] = stock["Adj Close"]

# 이동평균선 계산 (NumPy 활용)
def moving_average(data, window):
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')

# NaN을 제거한 후 NumPy 배열로 변환
close_prices = stock["Close"].dropna().values
stock = stock.dropna(subset=["Close"])  # NaN이 있는 행 제거

# 이동평균 계산
sma20 = moving_average(close_prices, 20)
sma200 = moving_average(close_prices, 200)

# 이동평균 결과를 stock 데이터프레임에 맞게 조정
stock["SMA20"] = np.nan  # NaN으로 채우기 (초기값을 맞추기 위해)
stock["SMA200"] = np.nan

# 20일 이동평균선 적용 (19번째 인덱스부터)
stock.iloc[19:, stock.columns.get_loc("SMA20")] = sma20

# 200일 이동평균선 적용 (199번째 인덱스부터)
sma200_length = len(sma200)  # sma200의 길이 확인
if len(stock) >= 199 + sma200_length:
    stock.iloc[199:199 + sma200_length, stock.columns.get_loc("SMA200")] = sma200
else:
    print("SMA200 길이가 데이터 범위를 초과하여 적용할 수 없습니다.")

# NaN 개수 확인
print("NaN 개수 확인:\n", stock.isna().sum())

# NaN이 없는 데이터만 필터링
valid_data = stock.dropna(subset=["Close", "SMA20", "SMA200"])

# 3. 상관계수 계산
corr_matrix = np.corrcoef(valid_data[["Close", "SMA20", "SMA200"]].dropna().T)
corr_df = pd.DataFrame(corr_matrix, index=["Close", "SMA20", "SMA200"], columns=["Close", "SMA20", "SMA200"])

print(f"20일 이동평균선과 종가의 상관계수: {corr_df.loc['Close', 'SMA20']:.4f}")
print(f"200일 이동평균선과 종가의 상관계수: {corr_df.loc['Close', 'SMA200']:.4f}")

# 4. 회귀선 구하기 (LinearRegression을 사용)
def get_regression_line(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model.predict(x.reshape(-1, 1)), model.coef_[0], model.intercept_

# 종가와 SMA20, SMA200에 대한 회귀선과 회귀계수 구하기
x_sma20 = np.arange(len(valid_data))  # x축 값 (인덱스)
y_sma20 = valid_data["SMA20"].dropna().values
regression_line_sma20, coef_sma20, intercept_sma20 = get_regression_line(x_sma20, y_sma20)

x_sma200 = np.arange(len(valid_data))  # x축 값 (인덱스)
y_sma200 = valid_data["SMA200"].dropna().values
regression_line_sma200, coef_sma200, intercept_sma200 = get_regression_line(x_sma200, y_sma200)

# 5. 그래프 시각화
plt.figure(figsize=(12,6))
plt.plot(stock.index, stock["Close"], label="Close Price", color="black", alpha=0.6)
plt.plot(stock.index, stock["SMA20"], label="20-day SMA", color="blue", linestyle="dashed")
plt.plot(stock.index, stock["SMA200"], label="200-day SMA", color="red", linestyle="dashed")
plt.plot(valid_data.index[:len(regression_line_sma20)], regression_line_sma20, label=f"SMA20 Regression Line (Slope: {coef_sma20:.4f})", color="blue", linestyle="solid")
plt.plot(valid_data.index[:len(regression_line_sma200)], regression_line_sma200, label=f"SMA200 Regression Line (Slope: {coef_sma200:.4f})", color="red", linestyle="solid")

plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("코카콜라 종가 이동 평균과 회귀선")
plt.legend()
plt.show()

# 6. 상관계수 히트맵
plt.figure(figsize=(6, 4))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".4f", linewidths=0.5)
plt.title("Correlation Heatmap: 종가 MV")
plt.show()

# 회귀계수 출력
print(f"SMA20 회귀계수: {coef_sma20:.4f}, 절편: {intercept_sma20:.4f}")
print(f"SMA200 회귀계수: {coef_sma200:.4f}, 절편: {intercept_sma200:.4f}")
