import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from pypfopt import BlackLittermanModel, CovarianceShrinkage
from pandas import Timestamp
from pypfopt import EfficientFrontier

import warnings
warnings.filterwarnings('ignore')

# 데이터 병합
sectors_data = {}
excel_file_path = 'stocks_2000_2020_data_by_sector-2.xlsx'
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

def calculate_performance_metrics(start_date, end_date, return_df, df, learn_window=60, invest_window=20,
                                  n_clusters=15):
    # 날짜 계산
    invest_start_day = df.index[df.index >= start_date][0]
    invest_end_day = df.index[df.index.tolist().index(invest_start_day) + invest_window]
    learn_start_day = df.index[df.index.tolist().index(invest_start_day) - learn_window]
    learn_end_day = df.index[df.index.tolist().index(invest_start_day) - 1]

    # 누적 수익률 초기화
    cumulative_returns = [0]

    while invest_start_day < Timestamp(end_date):
        # 학습 데이터 수집
        learn_data = df.loc[learn_start_day:learn_end_day].dropna(axis=1)
        learn_return_data = np.log(learn_data / learn_data.shift(1)).dropna()

        # 데이터 형태 변환
        kshape_data = learn_return_data.T.values.reshape(-1, learn_return_data.shape[0])

        # kshape 클러스터링
        kshape = KShape(n_clusters=n_clusters, verbose=False)
        kshape_data = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(kshape_data)
        y_pred = kshape.fit_predict(kshape_data[:, :, 0])

        # 클러스터별 평균 수익률 추출 및 P, Q 행렬 구성
        Q = []
        P = np.zeros((n_clusters, len(learn_data.columns)))

        for cluster in range(n_clusters):
            cluster_indices = np.where(y_pred == cluster)[0]
            if len(cluster_indices) == 0:
                Q.append(0)
                continue
            cluster_data = learn_return_data.iloc[:, cluster_indices]
            cluster_mean_return = cluster_data.mean().mean() * 250
            Q.append(cluster_mean_return)
            P[cluster, cluster_indices] = 1 / len(cluster_indices)

        Q = np.array(Q).reshape(-1, 1)

        # Omega 행렬
        omega = np.eye(n_clusters) * 0.01

        # 블랙리터만 뷰 설정 및 최적화
        S = CovarianceShrinkage(learn_data).ledoit_wolf()
        bl = BlackLittermanModel(S, P=P, Q=Q, omega=omega)
        ret_bl = bl.bl_returns()
        ef = EfficientFrontier(ret_bl, S)
        try:
            weights = ef.max_sharpe()
        except:
            for i in weights.keys():
                weights[i] = 1 / len(weights)  # 1/n 씩 투자했다고 가정

        # 투자 기간 동안 수익률 계산
        invest_data = df.loc[invest_start_day:invest_end_day, learn_data.columns]
        invest_log_return = np.log(invest_data / invest_data.shift(1))
        portfolio_log_return = (invest_log_return * pd.Series(weights)).sum(axis=1)
        cumulative_log_return = portfolio_log_return.sum()

        # 누적 로그 수익률 업데이트
        cumulative_returns.append(cumulative_returns[-1] + cumulative_log_return)

        # 다음 투자 기간으로 리밸런싱
        next_invest_end_index = return_df.index.tolist().index(invest_end_day) + invest_window + 1
        next_invest_start_index = return_df.index.tolist().index(invest_start_day) + invest_window + 1

        if next_invest_end_index >= len(return_df.index) or next_invest_start_index >= len(return_df.index):
            break

        # 다음 투자 update
        invest_start_day = return_df.index[next_invest_start_index]
        invest_end_day = return_df.index[next_invest_end_index]
        learn_start_day = return_df.index[next_invest_start_index - learn_window]
        learn_end_day = return_df.index[next_invest_start_index - 1]

    # 성능 지표 계산
    monthly_returns = [(cumulative_returns[i + 1] - cumulative_returns[i]) for i in range(len(cumulative_returns) - 1)]

    # Sharpe Ratio
    annual_log_return = np.mean(monthly_returns) * 12 / invest_window * 20
    annual_volatility = np.std(monthly_returns) * np.sqrt(12 / invest_window * 20)
    risk_free_rate = 0.02
    sharpe_ratio = (annual_log_return - risk_free_rate) / annual_volatility

    # Sortino Ratio
    downside_risks = [r for r in monthly_returns if r < 0]
    downside_volatility = np.std(downside_risks) * np.sqrt(12)
    sortino_ratio = (annual_log_return - risk_free_rate) / downside_volatility

    # Profit Factor
    gross_profit = sum([r for r in monthly_returns if r > 0])
    gross_loss = abs(sum([r for r in monthly_returns if r < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

    cumulative_returns = np.cumsum(monthly_returns) # 누적 합계

     # MDD
    running_max = np.maximum.accumulate(cumulative_returns) # 최대 누적 수익률(각 시점까지 누적수익률 최대치)
    drawdown = cumulative_returns - running_max # 누적 수익률 - 최대 누적 수익률
    mdd = np.min(drawdown) # 가장 큰 손실률 택

    # Calmar
    calmar_ratio = annual_log_return / abs(mdd) if mdd != 0 else np.mean(cumulative_returns)

    return {
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Profit Factor": profit_factor,
        "MDD": mdd,
        "Calmar Ratio": calmar_ratio
    }


# 기간별로 성능 지표 계산
periods = [
    ("2003-01-01", "2007-12-31"),
    ("2008-01-01", "2012-12-31"),
    ("2013-01-01", "2016-12-31"),
    ("2017-01-01", "2020-12-31"),
]

performance_metrics_by_period = {}

for start, end in periods:
    performance_metrics_by_period[f"{start[:4]}-{end[:4]}"] = calculate_performance_metrics(start, end, return_df, df)

performance_df = pd.DataFrame(performance_metrics_by_period).T
print(performance_df)
