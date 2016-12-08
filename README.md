# VaRmcs
Calculation of  portfolio VaR using Monte Carlo Simulation. The portfolio is composed as follows:
- long 1000 European call options on IBM with maturity 1 year
- short (written) 1000 European call options on Apple with maturity 1 year.

The options are priced according to the Black-Scholes pricing model for vanilla options. The assumption of normality of stock returns is tested by performing a Jarque–Bera test on IBM and Apple log returns from 11/16/2011 to 11/16/2016. The VaR is evaluated at a 5% confidence level for a forecast horizon of 10 trading days, the data generating processes are two correlated random walks with drift (using Cholesky decomposition to account for the historical correlation between stock prices).

NOTE: data have to be donwloaded manually (IBM_5yrs, AAPL_5yrs).
In forecasting stock prices during a window of 10 days, we need to specify a data generating process underpinning our predictions. For the single call position we have chosen to represent the behavior of stock prices according to a random walk in which the drift function is set to zero and the diﬀusion to one: Pt = Pt−1 + t s.t. t ∼ iid(0,σ2 ) For the options portfolio we simulated two correlated random walks as we believe that the share prices of companies belonging to the same sector are (positively) correlated in a deterministic way, i.e. by specifying a time-invariant correlation structure equal to the correlation matrix ρnxn calculated from the log daily returns during the study period: (P1,t = 1,t P2,t = ρ +p(1−ρ2)2,twhere i,t for i = 1,2 are two uncorrelated Gaussian random errors. The above results are achieved trough a Choleski decomposition of the matrix ρnxn. Figure 10 displays the behavior of the forecasted (correlated) stock prices for a 10 day period. The rationale behind the MCS method for the estimation of VaR is that the distribution of the call premium is non-normal as opposed to the one of the simulated stock prices that is log-normal, measures of variability (such as the standard deviation) that rely on the normality assumption cannot hence be employed to calculate the actual VaR. However, as can be seen from Figures 11 and 12, the relationship between the synthetic stock prices and the corresponding changes in option price seems to be linear. A possible explanation of this result may rely on the fact that the forecast window is so short (10 trading days) that we may have focused on relatively small oscillations in the stock price that corresponds to changes in the option price in a linear way. The J-B test for normality, at 5% conﬁdence level, on the simulated changes in call premia for both stocks suggests indeed that they do come from a normal distribution. By carefully looking at Figure 12 alone, one could notice the plausible curvature of the line when the stock price of Apple approaches roughly $100 thereby suggesting that, for lower stock values, the non-linearity relation can appear.
