import pandas as pd
import numpy as np

def load_data_from_excel(file_path):
    available_sheets = pd.ExcelFile(file_path).sheet_names
    snp500 = pd.DataFrame()

    for sheet_name in available_sheets:
        sheet_data = pd.read_excel(file_path, sheet_name=sheet_name)
        sheet_data['Date'] = pd.to_datetime(sheet_data['Unnamed: 0'])
        sheet_data.set_index('Date', inplace=True)
        sheet_data.drop(columns='Unnamed: 0', inplace=True)
        sheet_data.columns = [f"{sheet_name}_{col}" for col in sheet_data.columns]
        if snp500.empty:
            snp500 = sheet_data
        else:
            snp500 = snp500.join(sheet_data, how='outer')

    # 날짜를 2003년부터 2020년까지로
    snp500 = snp500[(snp500.index >= '2003-01-01') & (snp500.index <= '2020-12-31')]

    return snp500

def compute_statistics(snp500):
    daily_returns_snp500 = snp500.pct_change(1).dropna()
    monthly_returns_snp500 = daily_returns_snp500.resample('M').apply(lambda x: (x + 1).prod() - 1)
    log_monthly_returns_snp500 = np.log(monthly_returns_snp500 + 1).mean(axis=1)

    annual_log_return = np.mean(log_monthly_returns_snp500) * 12
    annual_volatility = np.std(log_monthly_returns_snp500) * np.sqrt(12)

    risk_free_rate = 0.02
    sharpe_ratio = (annual_log_return - risk_free_rate) / annual_volatility

    downside_risks = [r for r in log_monthly_returns_snp500 if r < 0]
    downside_volatility = np.std(downside_risks) * np.sqrt(12)
    sortino_ratio = (annual_log_return - risk_free_rate) / downside_volatility

    gross_profit = sum([r for r in log_monthly_returns_snp500 if r > 0])
    gross_loss = abs(sum([r for r in log_monthly_returns_snp500 if r < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

    cumulative_returns = np.cumsum(log_monthly_returns_snp500)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    mdd = np.min(drawdown)
    calmar_ratio = annual_log_return / abs(mdd) if mdd != 0 else np.mean(cumulative_returns)

    annual_returns = {}
    for index, monthly_return in monthly_returns_snp500.iterrows():
        current_year = index.year
        if current_year in annual_returns:
            annual_returns[current_year] += monthly_return.sum()
        else:
            annual_returns[current_year] = monthly_return.sum()

    # 마지막 연도 (2020년)도 포함하여 계산
    profitable_years_count = sum(1 for annual_return in annual_returns.values() if annual_return > 0)
    unprofitable_years_count = sum(1 for annual_return in annual_returns.values() if annual_return <= 0)

    results = {
        "annual_log_return": annual_log_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "downside_volatility": downside_volatility,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "mdd": mdd,
        "calmar_ratio": calmar_ratio,
        "profitable_years_count": profitable_years_count,
        "unprofitable_years_count": unprofitable_years_count
    }

    return results

def main():
    file_path = 'stocks_2000_2020_data_by_sector-2.xlsx'
    df_snp500 = load_data_from_excel(file_path)
    statistics = compute_statistics(df_snp500)

    for key, value in statistics.items():
        print(f"{key}: {round(value, 3)}")

if __name__ == "__main__":
    main()
