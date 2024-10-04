# finance
Projects related to financial modeling, primarily in Python

**ARIMA**: useful for time-series modeling and forecasting

**ETS**: exponential smoothing

**GARCH**: used to model volatility

**CAPM**: asset pricing model

**ML basic**: basic ML models including linear regression, logistic regression, k-nearest neighbors, gradient boosting

**XGBoost**: gradient boosting ML model useful for forecasting

**options_pricing**: Black-Scholes for European-style options, Binomial Tree Model for America-style (early exercise)

**PINN**: physics-informed neural network. useful for options pricing when loss function residual is Black-Scholes PDE

TO-DO:

- ensure PINNs are using PDE as residual (2nd derivative) rather than explicit BS solution
- compare to real world data and ensure getting accurate predictions
- find novel solutions in non-constant volatility settings