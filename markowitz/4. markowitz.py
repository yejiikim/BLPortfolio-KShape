import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import Timestamp
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

import warnings
warnings.filterwarnings('ignore')

# 데이터 병합
sectors_data = {}
excel_file_path = '../stocks_2000_2020_data_by_sector-2.xlsx'
snp_price_data = pd.ExcelFile(excel_file_path)
for sheet_name in snp_price_data.sheet_names:
    sheet_data = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    sheet_data['Date'] = pd.to_datetime(sheet_data['Unnamed: 0'])
    sheet_data.set_index('Date', inplace=True)
    sheet_data.drop(columns='Unnamed: 0', inplace=True)
    sectors_data[sheet_name] = sheet_data

# 데이터프레임 생성
df = pd.DataFrame()
for sector, data in sectors_data.items():
    data.columns = [f"{sector}_{col}" for col in data.columns]
    if df.empty:
        df = data
    else:
        df = df.join(data, how='outer')

# 로그수익률 시계열 생성
return_df = np.log(df / df.shift(1))

# 날짜 계산
invest_start_day = return_df.index[return_df.index>='2003-01-01'][0]
learn_window = 60
invest_window = 20
invest_end_day = return_df.index[return_df.index.tolist().index(invest_start_day) + invest_window]
learn_start_day = return_df.index[return_df.index.tolist().index(invest_start_day) - learn_window]
learn_end_day = return_df.index[return_df.index.tolist().index(invest_start_day) - 1]
returns_all = [0]
while invest_end_day< Timestamp('2020-12-30'):
    # 학습 데이터 수집
    learn_data = df.loc[learn_start_day:learn_end_day].dropna(axis=1)
    learn_data = learn_data.iloc[:, :100]
    # 포트폴리오 계산
    mu = expected_returns.mean_historical_return(learn_data)
    S = risk_models.sample_cov(learn_data)
    ef = EfficientFrontier(mu, S)
    try:
        ef.min_volatility()
        w = ef.clean_weights()
        w_df = pd.DataFrame([w.values()], columns=w)
        # 수익률 계산
        returns = []
        for i, v in enumerate(w_df):
            returns.append(np.log(df.loc[invest_end_day, v] / df.loc[invest_start_day, v]))
        w_df.loc[1] = returns
        w_df.loc[2] = w_df.loc[0] * w_df.loc[1]
        this_return = w_df.loc[2].sum()
    except:
        print('error')
        this_return = 0
    returns_all.append(this_return)
    # 다음 투자 기간으로 리밸런싱
    next_invest_end_index = return_df.index.tolist().index(invest_end_day) + invest_window + 1 # 다음 투자기간 종료
    next_invest_start_index = return_df.index.tolist().index(invest_start_day) + invest_window + 1 # 다음 투자기간 시작

    # 인덱스가 범위를 벗어나는지 확인
    if next_invest_end_index >= len(return_df.index) or next_invest_start_index >= len(return_df.index):
        break

    # 다음 투자 update
    invest_start_day = return_df.index[next_invest_start_index]
    invest_end_day = return_df.index[next_invest_end_index]
    learn_start_day = return_df.index[next_invest_start_index - learn_window]
    learn_end_day = return_df.index[next_invest_start_index - 1]

# 월별 로그수익률 계산
monthly_returns = returns_all

df_monthly_returns = pd.DataFrame(monthly_returns, columns=['0'])
df_monthly_returns.reset_index(inplace=True)
df_monthly_returns.rename(columns={"index": "Unnamed: 0"}, inplace=True)
df_monthly_returns.to_csv("monthly_returns_markowitz.csv", index=False)


# monthly_returns_df = pd.DataFrame({'Date': df.index[1:len(monthly_returns) + 1], 'Cumulative_Log_Return': np.cumsum(monthly_returns)})
# monthly_returns_df.to_csv("monthly_returns_markowitz.csv", index=False)

# monthly_cumulative_returns_df = pd.DataFrame({'Date': df.index[1:len(monthly_returns)+1], 'Cumulative_Log_Return': np.cumsum(monthly_returns)})
# monthly_cumulative_returns_df.to_csv("markowitz_cumulative_returns.csv", index=False)

# 연 로그수익률, 표준편차 계산
annual_log_return = np.mean(monthly_returns) * 12 / invest_window * 20
annual_volatility = np.std(monthly_returns) * np.sqrt(12/invest_window*20)  # 연별 표준편차


# 1. Sharpe Ratio
risk_free_rate = 0.02
sharpe_ratio = (annual_log_return - risk_free_rate) / annual_volatility


# 2. Sortino Ratio
downside_risks = [r for r in monthly_returns if r < 0]
downside_volatility = np.std(downside_risks) * np.sqrt(12)
sortino_ratio = (annual_log_return - risk_free_rate) / downside_volatility


# 3. Profit Factor
gross_profit = sum([r for r in monthly_returns if r > 0])
gross_loss = abs(sum([r for r in monthly_returns if r < 0]))
profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf  # gross_loss가 0이면 profit_factor는 무한대
# 누적 수익률 계산
cumulative_returns = np.cumsum(monthly_returns)

print(len(cumulative_returns))
# MDD 계산
running_max = np.maximum.accumulate(cumulative_returns)
drawdown = cumulative_returns - running_max
mdd = np.min(drawdown)  # MDD는 drawdown의 최소값

# Calmar Ratio 계산
if mdd != 0:
    calmar_ratio = annual_log_return / abs(mdd)
else:
    calmar_ratio = np.mean(cumulative_returns)  # MDD가 0인 경우 평균 누적 수익률로 대체

# 연도별로 cumulative_returns 분할
years = list(range(2003, 2021))
annual_returns = []

for i in range(0, len(years) - 1):  # 마지막 연도는 별도로 처리
    start_index = i * (12 // (invest_window // 20))
    end_index = start_index + (12 // (invest_window // 20))

    # out-of-bounds 오류
    if end_index >= len(cumulative_returns):
        end_index = len(cumulative_returns) - 1

    annual_return = cumulative_returns[end_index] - cumulative_returns[start_index]
    annual_returns.append(annual_return)

# 마지막 연도의 수익률을 별도로 계산
start_index = (years[-1] - 2003) * (12 // (invest_window // 20))
if start_index < len(cumulative_returns):
    annual_return = cumulative_returns[-1] - cumulative_returns[start_index]
    annual_returns.append(annual_return)

# profitable과 unprofitable years 계산
profitable_years = len([r for r in annual_returns if r > 0])
unprofitable_years = len([r for r in annual_returns if r <= 0])

print(f"Profitable Years: {profitable_years}")
print(f"Unprofitable Years: {unprofitable_years}")
print(round(annual_log_return, 3))
print(round(annual_volatility, 3))
print(round(sharpe_ratio, 3))
print(round(sortino_ratio, 3))
print(round(downside_volatility, 3))
print(round(profit_factor, 3))
print(round(gross_profit, 3))
print(round(gross_loss, 3))
print(round(mdd, 3))
print(round(calmar_ratio, 3))

