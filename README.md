# portfolio_optimization_efficient_frontiers
Visualization of Efficient Frontier and Unoptimized Portfolio in Four Different Scenarios

This script is a study in portfolio optimization under four different scenarios.

I developed it to demonstrate to a portfolio manager that an optimized portfolio subject to certain constraints would hypothetically
enable us to create a portfolio with less risk and greater return.

I say 'hypothetically' because an ex-ante study like this uses historical volatility as a predictor for future volatility.
If the particular range of historical volatility utilized to calculate mean returns and standard deviation is NOT a good predictor
of future period returns and risk, the results may be very different than projected.

The fund mandate required a portfolio with a large proportion (but not all) SP500 constituents.  An optimized portfolio designed to 
maximize Sharpe Ratio without constraints results in a highly concentrated portfolio with only a handful of constituents.

To contrain the optimization so the final portfolio would have the same number of constituents as the initial portfolio, I restricted 
the amount the starting weight of each constituent could vary.  This range was +/- 15%.

I generated 4 sets of results in the final visualization to illustrate the sensitive of the study to the period of daily revenue values.
one-month, two month, three month, and four month periods were used as a basis for calculating mean revenue, standard deviation, and correlation.

Code is annotated in the base file.

Result is the PNG file.
