
# VaRmcs
Calculation of  portfolio VaR using Monte Carlo Simulation. The portfolio is composed as follows:
- long 1000 European call options on IBM with maturity 1 year
- short (written) 1000 European call options on Apple with maturity 1 year.

The options are priced according to the Black-Scholes pricing model for vanilla options. The assumption of normality of stock returns is tested by performing a Jarque–Bera test on IBM and Apple log returns from 11/16/2011 to 11/16/2016. The VaR is evaluated at a 5% confidence level for a forecast horizon of 10 trading days, the data generating processes are two correlated random walks with drift (using Cholesky decomposition to account for the historical correlation between stock prices).

NOTE: data have to be donwloaded manually (IBM_5yrs, AAPL_5yrs).
