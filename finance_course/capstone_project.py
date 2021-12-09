# Capstone Project: Analyze Financial Data with Python

# Import packages/modules
import numpy as np
import pandas as pd
import pandas_datareader as web
import cvxopt as opt
import matplotlib.pyplot as plt
from cvxopt import blas, solvers
from datetime import datetime, date

# 1 Import 4 Recommended Stocks
# For this project we will recommend Nvidia, Microsoft, Ali Baba, and Nio

start_time = datetime(2019, 1, 1)
end_time = datetime(2021, 9, 12)

tickers = ["NVDA", "MSFT", "BABA", "NIO"]

pd.set_option('display.max_columns', 10)
stock_data = web.DataReader(tickers, "yahoo", start=start_time, end=end_time)  # Save the stock data as whole
stock_adj_close = web.DataReader(tickers, "yahoo", start=start_time, end=end_time)["Adj Close"]  # Just 'Adj Close'
# column
# print(stock_adj_close)

# 1.1 Clean the data (if needed)
stock_adj_close.fillna(0, inplace=True)
# print(stock_adj_close)

# 1.2 Figure of the stocks
fig1 = plt.figure(figsize=(12, 8))
fig1.subplots_adjust(hspace=1)
for i in range(len(tickers)):
    plt.subplot(3, 2, i + 1)
    stock_adj_close[tickers[i]].plot()
    plt.xlabel('Date')
    plt.ylabel('Adj Close')
    plt.tight_layout()
    plt.title(str(tickers[i]), fontsize=12, fontweight='bold')
# plt.savefig('stock-performance.jpeg')
# plt.show()


# 2 Calculate Financial Statistics
# 2.1 Simple Rate of Return
simple_daily_returns = stock_adj_close.pct_change()
print(simple_daily_returns)

# Subplots for Simple Rate of Return for Each Stock
# Keep count of plot subplots:
plot = 0
fig2 = plt.figure(figsize=(15, 15))
fig2.subplots_adjust(hspace=1, wspace=1)

# For loop to go through length of tickers:
for i in range(len(tickers)):
    plot += 1
    # Set up subplots
    plt.subplot(3, 2, plot)
    # plot of "simply daily returns" for each stock:
    plt.plot(simple_daily_returns[tickers[i]])
    plt.title(tickers[i], fontsize=12, fontweight='bold')
    plt.tight_layout()
# plt.show()

# 2.2 Log Rate of Return
simple_daily_returns_log = np.log(simple_daily_returns)
# print(simple_daily_returns_log)

# 2.3 Mean of Simple Daily Rate of Returns
mean_simple = simple_daily_returns.mean()

# Bar Chart for Mean of Simple Daily Rate of Returns
mean_simple.keys()  # Get the key(stock symbols)

# Grab each daily mean value for the y_axis
height = []
for i in mean_simple.keys():
    height.append(mean_simple[i])

# Arrange keys on x_axis based on length
x_pos = np.arange(len(mean_simple.keys()))

plt.figure(figsize=(12, 8))
plt.bar(x_pos, height)
plt.xticks(x_pos, mean_simple.keys())
plt.xlabel("Stocks")
plt.ylabel("Daily Mean")
plt.title("Daily Mean Rate of Return")
plt.show()

# 2.3 Variance of Simple Daily Rate of Returns
var_simply = simple_daily_returns.var()

# 2.4 Covariance of Simply Daily Rate of Returns
cov_simply = simple_daily_returns.cov()

# 2.4 SD of Simple Daily Rate of Returns
std_simply = simple_daily_returns.std()

# 2.5 Correlations
corr_matrix = simple_daily_returns.corr()
print(corr_matrix)


# Nvidia is highest correlated to Microsoft (0.73) and least to Nio (0.26)
# Ali Baba and Nio is not highly correlated Microsoft and Nvidia
# All the stocks are somewhat correlated, no negative correlations

# 3 Mean-Variance Optimization (Optimized Portfolio)
# Return a set of portfolio options with volatility and annualized returns

def return_portfolios(expected_returns, cov_matrix):
    port_returns = []
    port_volatility = []
    stock_weights = []

    selected = expected_returns.axes[0]

    num_assets = len(selected)
    num_portfolios = 5000

    for single_portfolio in range(num_portfolios):
        # get stock portfolio weights by dividing random number assigned to each stock with the sum of random numbers
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, expected_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)

    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility}

    for counter, symbol in enumerate(selected):
        portfolio[symbol + ' Weight'] = [Weight[counter] for Weight in stock_weights]

    df = pd.DataFrame(portfolio)

    column_order = ['Returns', 'Volatility'] + [stock + ' Weight' for stock in selected]

    df = df[column_order]

    return df


def optimal_portfolio(returns):
    n = returns.shape[1]
    returns = np.transpose(returns.values)

    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x'] for mu in mus]

    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


# Portfolio options:
portfolio_options = return_portfolios(mean_simple, cov_simply)
print(portfolio_options)

# Optimal portfolio:
weights, returns, risks = optimal_portfolio(simple_daily_returns[1:])
print(weights.shape)

# 4 Efficient Frontier Figure
plt.figure(figsize=(15, 15))
portfolio_options.plot.scatter(x="Volatility", y="Returns", s=20)
plt.title('Efficient Frontier', fontsize=15)
plt.xlabel("Volatility")
plt.ylabel("Expected Daily Returns")

# Place markers for single stock portfolios
std = np.sqrt(np.diagonal(cov_simply))
plt.scatter(std, mean_simple, marker='x', color='red', s=200, linewidth=3)
plt.plot(risks, returns, color='green', linewidth=3)
plt.grid(color='grey', linestyle=':', linewidth=2)
# plt.savefig('efficient_frontier.jpeg')
plt.show()

# 5 Portfolio Recommendation
# Finding the optimal portfolio
rf = 0.01  # risk factor
optimal_risky_port = portfolio_options.iloc[((portfolio_options['Returns'] - rf)
                                             / portfolio_options['Volatility']).idxmax()]
print("Optimal Portfolio for Max Returns""\n", optimal_risky_port)

# Minimum volatility portfolio
min_vol_port = portfolio_options.iloc[portfolio_options['Volatility'].idxmin()]
print("Minimum Risk Portfolio""\n", min_vol_port)


