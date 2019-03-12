#!/usr/bin/env python2.7

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import data_inputs
import scipy.optimize as sco
from collections import OrderedDict

def pull_data():
    
    #Pull Portfolio Positions for 3/31/17 and Perform Some Formatting

    model_holdings_df = pd.read_csv(data_inputs.base_path + '\\model_holdings.csv')

    model_holdings_df['Ticker'] = model_holdings_df['Ticker'].str.upper()
    model_holdings_df['Ticker'] = model_holdings_df['Ticker'].apply(lambda x: str(x))
    
    model_holdings_df = model_holdings_df[model_holdings_df['FinalDecision'] == 'Invested']
    model_holdings_df = model_holdings_df.sort_values('Ticker')
    model_holdings_df = model_holdings_df.reset_index().drop('index',1)
    model_holdings_df['RebalanceWeight'] = model_holdings_df['RebalanceWeight'] / 100.0
    
    # Pull Grossed Up Weight Values from model_holdings_df (holdings for 3/31/17
    
    grossed_up_wgts_df = model_holdings_df[['Ticker','RebalanceWeight']]
    grossed_up_wgts_df = grossed_up_wgts_df.sort_values('Ticker')

    #Load Daily Security Returns
    
    agg_returns_df = pd.read_csv(data_inputs.base_path + '\\agg_returns_df_3_year.csv')
    agg_returns_df['Date'] = pd.to_datetime(agg_returns_df['Date'])
    agg_returns_df.set_index("Date",inplace=True)
    
    #Create Return Histories for Volatility Analyses; Store Each History in a Dictionary
    
    agg_returns_1_month = agg_returns_df[agg_returns_df.index >= pd.to_datetime('2017-02-28')]
    agg_returns_2_month = agg_returns_df[agg_returns_df.index >= pd.to_datetime('2017-01-31')]
    agg_returns_3_month = agg_returns_df[agg_returns_df.index >= pd.to_datetime('2016-12-31')]
    agg_returns_4_month = agg_returns_df[agg_returns_df.index >= pd.to_datetime('2016-11-30')]
    
    return_period_dict=OrderedDict()
    return_period_dict['1_month'] = agg_returns_1_month
    return_period_dict['2_month'] = agg_returns_2_month
    return_period_dict['3_month'] = agg_returns_3_month
    return_period_dict['4_month'] = agg_returns_4_month
    
    return return_period_dict, model_holdings_df, grossed_up_wgts_df

def return_range(agg_returns_df):

    '''

    Create 50000 portfolios allowing starting weigths of 3/31 portfolio to deviate +/- 15%

    Sum of weights add to 1 (full investment)

    Identify the Maximum Return Portfolio and it's Return
    Identify the Minimum Volatility Portfolio and its Return.

    These return values will form the endpoints of the efficient frontier for a particular scenario.

    Note:  The 50000th portfolio will be the STARTING WEIGHTS for the Optimized portfolios.
           These are saved for easy plotting at conclusion of script.

    '''

    cols = list(grossed_up_wgts_df['Ticker'].unique())
    agg_returns_df = agg_returns_df[cols]
    grossed_up_wgts_array = np.array(grossed_up_wgts_df['RebalanceWeight']).T

    # calculate mean daily return and covariance of daily returns

    mean_daily_returns = agg_returns_df.mean()
    cov_matrix = agg_returns_df.cov()

    # set number of runs of random portfolio weights
    num_portfolios = 50000

    # Create an empty array to hold results
    results = np.zeros((3 + len(agg_returns_df.columns), num_portfolios))

    #Generate portfolios, returns and volatilities (standard deviation)

    for i in xrange(num_portfolios):

        if np.mod(i, 10000) == 0:

            print i

        if i == num_portfolios - 1:

            weights = grossed_up_wgts_array

        else:

            lb = grossed_up_wgts_array * .85
            ub = grossed_up_wgts_array * 1.15

            weights = np.random.uniform(lb, ub)

            weights /= np.sum(weights)

        # calculate portfolio return and volatility
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

        # store portfolio return, volatility, and Sharpe Ratio (Rf Rate eliminated for simplicity)
        #in results array.  Append portfolio constituent weights to results array
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
        results[2, i] = results[0, i] / results[1, i]
        # iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results[j + 3, i] = weights[j]

    #Convert array to dataframe

    #Create Column Names
    stats_list = ['ret', 'stdev', 'sharpe']
    stats_list.extend(agg_returns_df.columns)

    #Convert the array to a dataframe
    results_frame = pd.DataFrame(results.T, columns=stats_list)

    print 'Max Return Portfolio - Monte Carlo:  {results}'.format(results=results_frame['ret'].max())

    minimum_vol = results_frame['stdev'].min()
    print 'Return at Min Vol Portfolio - Monte Carlo:  {results}'.format(results=(results_frame.loc[results_frame['stdev'] == minimum_vol,'ret'].values[0]))

    return results_frame, stats_list


def efficient_frontier(agg_returns_df):

    '''
    Generate data for plotting the efficient frontier

    Input a dataframe of returns for constituents (e.g, 1 month, 2 months, 3 months, 4 months)

    Establish stats (average daily return, standard deviation)
    Load in weights

    1. Determine statistics and weights for Max Sharpe Portfolio
    2. Determine statistics and weights for Min Vol Portfolio
    3. Generate 40 equally spaced points between Max Sharpe portfolio return and Min Vol portfolio (trets)
    4. Determine the minimum volatility portfolio for each return point. (tvols)
    5. Return trets, tvols and other information at completion of function

    '''

    results_frame, stats_list = return_range(agg_returns_df)

    cols = list(grossed_up_wgts_df['Ticker'].unique())
    agg_returns_df = agg_returns_df[cols]
    grossed_up_wgts_array = np.array(grossed_up_wgts_df['RebalanceWeight']).T

    mean_daily_returns = agg_returns_df.mean()
    cov_matrix = agg_returns_df.cov()

    def statistics(weights):

        weights = np.array(weights)
        pret = np.sum(mean_daily_returns * weights) * 252
        pvol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

        return np.array([pret, pvol, pret / pvol])

    print "Determining Optimized Sharpe Portfolio"

    def min_func_sharpe(weights):
        return -statistics(weights)[2]

    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})

    bnds = tuple((0.85*x,1.15*x) for x in grossed_up_wgts_array)

    opts = sco.minimize(min_func_sharpe, grossed_up_wgts_array, method='SLSQP', bounds=bnds, constraints=cons)

    optimized_sharpe_data_array = np.zeros(3)
    optimized_sharpe_data_array[0] = statistics(opts['x'])[0]
    optimized_sharpe_data_array[1] = statistics(opts['x'])[1]
    optimized_sharpe_data_array[2] = statistics(opts['x'])[2]
    optimized_sharpe_data_array = np.concatenate((optimized_sharpe_data_array, opts['x']))
    optimized_sharpe_data_df = pd.DataFrame(optimized_sharpe_data_array)
    optimized_sharpe_data_df = optimized_sharpe_data_df.T
    optimized_sharpe_data_df.columns = stats_list
    optimized_sharpe_data_df = optimized_sharpe_data_df.T

    print optimized_sharpe_data_array[0]

    print "Determining Minimum Vol Portfolio"

    def min_func_variance(weights):
        return (statistics(weights)[1] ** 2)


    optv = sco.minimize(min_func_variance, grossed_up_wgts_array, method='SLSQP', bounds=bnds, constraints=cons)

    optimized_variance_data_array = np.zeros(3)
    optimized_variance_data_array[0] = statistics(optv['x'])[0]
    optimized_variance_data_array[1] = statistics(optv['x'])[1]
    optimized_variance_data_array[2] = statistics(optv['x'])[2]
    optimized_variance_data_array = np.concatenate((optimized_variance_data_array, optv['x']))
    optimized_variance_data_df = pd.DataFrame(optimized_variance_data_array)
    optimized_variance_data_df = optimized_variance_data_df.T
    optimized_variance_data_df.columns = stats_list
    optimized_variance_data_df = optimized_variance_data_df.T

    def min_func_port(weights):
        return statistics(weights)[1]

    trets = np.linspace(optimized_variance_data_array[0],optimized_sharpe_data_array[0],40)
    tvols = []

    print 'Maximum Return Portfolio Return: {mr}'.format(mr=results_frame['ret'].max())

    print 'Determining Minimum Volatility portfolio for portfolios at Specified Return levels'

    for tret in trets:

        cons=({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tret},
              {'type': 'eq', 'fun': lambda x: np.sum(x)-1})
        res = sco.minimize(min_func_port, grossed_up_wgts_array, method='SLSQP', bounds=bnds, constraints=cons)
        tvols.append(res['fun'])

        print 'Minimum Volatility at Return Level {tret} is {vol}'.format(tret=tret,vol = res['fun'] )

    tvols= np.array(tvols)

    return optimized_sharpe_data_array, optimized_variance_data_array, trets, tvols, results_frame
              
if __name__ == '__main__':

    os.chdir(data_inputs.base_path)
    
    #Pull Data

    return_period_dict, model_holdings_df, grossed_up_wgts_df = pull_data()

    '''
    
    Generate Elements for Subequent Visualization

    optimized_sharpe_data_array:  portfolio return, portfolio stdev, sharpe, optimized weights by ticker
                                  for max Sharpe Portfolio constraining weights of each security to vary within +/- 15%
    
    optimized_variance_data_array:   portfolio return, portfolio stdev, sharpe, optimized weights by ticker
                                     for Minimum Volatility Portfolio constraining weights of each security to vary 
                                     within +/- 15%                     
                                      
    trets:  list of portfolio return values between Minimum Volatility Portfolio return and Max Return Portfolio
            return for which to calculate efficient frontier plotting points
            
    tvols:  list of minimum volatilities at specified return levels in trets    
    
    '''
    for key in return_period_dict.keys():

        optimized_sharpe_data_array, \
        optimized_variance_data_array, \
        trets, \
        tvols, \
        results_frame = efficient_frontier(return_period_dict[key])
    
        if key == '1_month':
             
            optimized_sharpe_data_array_1_month = optimized_sharpe_data_array.copy()
            optimized_variance_data_array_1_month = optimized_variance_data_array.copy()
            trets_1_month = trets.copy()
            tvols_1_month = tvols.copy()
            results_frame_1_month = results_frame.copy()
            SPX_port_1_month = results_frame.iloc[results_frame.index.max()]
    
    
        elif key == '2_month':
    
            optimized_sharpe_data_array_2_month = optimized_sharpe_data_array.copy()
            optimized_variance_data_array_2_month = optimized_variance_data_array.copy()
            trets_2_month = trets.copy()
            tvols_2_month = tvols.copy()
            results_frame_2_month = results_frame.copy()
            SPX_port_2_month = results_frame.iloc[results_frame.index.max()]
    
        elif key == '3_month':
    
            optimized_sharpe_data_array_3_month = optimized_sharpe_data_array.copy()
            optimized_variance_data_array_3_month = optimized_variance_data_array.copy()
            trets_3_month = trets.copy()
            tvols_3_month = tvols.copy()
            results_frame_3_month = results_frame.copy()
            SPX_port_3_month = results_frame.iloc[results_frame.index.max()]
    
        elif key == '4_month':
    
            optimized_sharpe_data_array_4_month = optimized_sharpe_data_array.copy()
            optimized_variance_data_array_4_month = optimized_variance_data_array.copy()
            trets_4_month = trets.copy()
            tvols_4_month = tvols.copy()
            results_frame_4_month = results_frame.copy()
            SPX_port_4_month = results_frame.iloc[results_frame.index.max()]
    
        else:
    
            pass

    '''
    
    Plot Efficient Frontiers, Min Vol Portfolios, Max Sharpe Portfolios and UnOptimized Portfolios
    
    Max Sharpe Portfolio
    Min Vol Portfolio
    UnOptimized Portfolio
    Efficient Frontier Portfolios between Min Vol and Max Sharpe.
    
    '''
    #Plot EF and Stats Derived Using 1 month return figures
    
    plt.plot(SPX_port_1_month[1],SPX_port_1_month[0],'g*',markersize=15.0)
    plt.plot(optimized_sharpe_data_array_1_month[1],optimized_sharpe_data_array_1_month[0],'r*', markersize=15.0)
    plt.plot(optimized_variance_data_array_1_month[1],optimized_variance_data_array_1_month[0],'y*', markersize=15.0)
    plt.scatter(tvols_1_month,trets_1_month,c=trets_1_month/tvols_1_month, marker = 'x')
    
    #Plot EF and Stats Derived Using 2 month return figures
    
    plt.plot(SPX_port_2_month[1],SPX_port_2_month[0],'g*',markersize=15.0)
    plt.plot(optimized_sharpe_data_array_2_month[1],optimized_sharpe_data_array_2_month[0],'r*', markersize=15.0)
    plt.plot(optimized_variance_data_array_2_month[1],optimized_variance_data_array_2_month[0],'y*', markersize=15.0)
    plt.scatter(tvols_2_month,trets_2_month,c=trets_2_month/tvols_2_month, marker = 'X')
    
    #Plot EF and Stats Derived Using 3 month return figures
    
    plt.plot(SPX_port_3_month[1],SPX_port_3_month[0],'g*',markersize=15.0)
    plt.plot(optimized_sharpe_data_array_3_month[1],optimized_sharpe_data_array_3_month[0],'r*', markersize=15.0)
    plt.plot(optimized_variance_data_array_3_month[1],optimized_variance_data_array_3_month[0],'y*', markersize=15.0)
    plt.scatter(tvols_3_month,trets_3_month,c=trets_3_month/tvols_3_month, marker = 'd')
    
    #Plot EF and Stats Derived Using 4 month return figures
    
    plt.plot(SPX_port_4_month[1],SPX_port_4_month[0],'g*',markersize=15.0)
    plt.plot(optimized_sharpe_data_array_4_month[1],optimized_sharpe_data_array_4_month[0],'r*', markersize=15.0)
    plt.plot(optimized_variance_data_array_4_month[1],optimized_variance_data_array_4_month[0],'y*', markersize=15.0)
    plt.scatter(tvols_4_month,trets_4_month,c=trets_4_month/tvols_4_month, marker = '*')

    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')

    plt.show()

