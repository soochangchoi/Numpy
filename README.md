# 🥤 Coca-Cola & S&P500 Stock Analysis Project

코카콜라(KO) 및 S&P500 지수의 주가, 거래량, 배당금, 성장성 분석 및 시각화 프로젝트입니다.  
이동평균선, 회귀분석, 상관계수 히트맵 등을 통해 장기 주가 흐름과 주요 지표를 분석합니다.

---

## 📂 프로젝트 구성

| 구분            | 주요 기능                               | 스크립트                              |
|----------------|--------------------------------------|---------------------------------------|
| 📈 시가 분석        | 시가 이동평균, 회귀선, 상관계수 분석                | `cocacola Open.py`, `S&P500 Open.py` |
| 📉 종가 분석        | 종가 이동평균, 회귀선, 상관계수 분석                | `cocacola close.py`                  |
| 📊 고가 / 저가 분석  | 상한가, 하한가 이동평균, 회귀선, 상관계수 분석       | `cocacola High.py`, `cocacola Low.py` |
| 📊 거래량 분석       | 거래량 이동평균, 회귀선, 상관계수 분석              | `cocacola Volume.py`                 |
| 💰 배당금 분석       | 배당금과 주가의 상관관계 분석 및 회귀선 시각화       | `cocacoladiv.py`                     |
| 🚀 성장성 분석       | 매출, EPS, ROE, ROA, PER, PEG 등 성장성 지표 분석 | `cocacolagrowth.py`                  |

---

## ▶ 실행 방법

### 1. 코카콜라 시가 분석
```bash
python coca_analysis/cocacola\ Open.py
2. S&P500 시가 분석
bash
복사
편집
python sp500_analysis/S&P500\ Open.py
3. 기타 분석 실행
bash
복사
편집
python coca_analysis/cocacola\ close.py
python coca_analysis/cocacola\ High.py
python coca_analysis/cocacola\ Low.py
python coca_analysis/cocacola\ Volume.py
python coca_analysis/cocacoladiv.py
python coca_analysis/cocacolagrowth.py
📊 분석 결과
 20일, 200일 이동평균선과 주가 비교

 상관계수 히트맵

 회귀분석을 통한 추세선 시각화

 배당금과 주가 간의 관계 분석

 재무 지표 기반 성장성 평가 (PEG, ROE 등)

🛠 사용 기술
Python 3.9

yfinance

numpy

pandas

seaborn

matplotlib

scikit-learn
