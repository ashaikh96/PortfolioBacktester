#main code used for backtesting portfolios

#imports
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os
import riskparity as erc_ver1
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pandas.tseries.offsets import MonthEnd
from datetime import timedelta
import statsmodels.api as sm

class portfolio:
    
    def __init__(self, name, code, descrip, positions,trend,rebal_freq,stat_dyn):
        '''initialize portfolio object'''
        
        #set portfolio characteristics
        self.assets = list(positions)
        self.name = name
        self.code = code
        self.descrip = descrip
        self.positions = positions
        self.trend = trend
        self.rebal = rebal_freq
        self.wgt_method = stat_dyn
        self.firstday = positions.index[0]
        self.lastday = positions.index[-1]
        self.stats.name = name
        #calculate returns of portfolio
        self.calculate_rets()
        #calculate excess returns of portfolio
        self.excess_rets = (self.returns - risk_free).dropna()
        #calculate statistics of portfolio
        self.stats = pd.Series()
        self.portfolio_metrics()     
        
    def portfolio_metrics(self):
        '''calculates stats of portfolio'''
        #compound annual growth rate
        self.exp_ret = ((1+self.returns.mean())**252)-1
        #annual volatility
        self.vol = self.returns.std()*np.sqrt(252)
        #sharpe ratio
        self.sharpe = (((1+self.excess_rets.mean())**252)-1)/self.vol
        #sortino ratio
        self.sortino = (((1+self.excess_rets.mean())**252)-1)/self.sortino_calc()
        #cumulative return of portfolio
        cum_returns = (1 + self.returns).cumprod()
        #max drawdown
        self.dd = (1 - cum_returns.div(cum_returns.cummax()))*-1
        self.maxdd = self.dd.min()
        #total return
        self.tot_ret = cum_returns.values[-1]-1
        #historical VaR
        self.VaR = self.Hist_VaR(0.99)
        #historical CVaR
        self.CVaR = self.Hist_CVaR()
        #beta,alpha,r-squared
        self.beta, self.alpha, self.R2 = self.beta_calc()
        #annualize alpha
        self.alpha = ((1+self.alpha)**252)-1
        #treynor ratio
        self.treynor= (((1+self.excess_rets.mean())**252)-1)/self.beta
        
        #store in stats series
        self.stats.loc['CAGR'] = self.exp_ret
        self.stats.loc['Vol'] = self.vol
        self.stats.loc['Sharpe'] = self.sharpe
        self.stats.loc['Sortino'] = self.sortino
        self.stats.loc['Total Return'] = self.tot_ret
        self.stats.loc['Max DD'] = self.maxdd
        self.stats.loc['VaR'] = self.VaR
        self.stats.loc['CVaR'] = self.CVaR
        self.stats.loc['Beta'] = self.beta
        self.stats.loc['Alpha'] = self.alpha
        self.stats.loc['R2'] = self.R2
        self.stats.loc['Treynor'] = self.treynor
        self.stats.loc['Leverage'] = leverage
        
    def beta_calc(self,ref='SPY'):
        '''regresses portfolio returns on market factor'''
        #preprocess data
        temp_df = pd.concat([self.returns,Returns[ref]], axis=1,sort=True)
        temp_df.dropna(how='any',inplace=True)
        self.index_rets = temp_df[ref]
        
        #independent variable
        x = temp_df[ref]
        
        #dependent variable
        y = temp_df[self.code]
        
        #add constant for intercept
        x1 = sm.add_constant(x)
        
        #run regression
        model = sm.OLS(y, x1)
        results = model.fit()     
        
        #return beta, alpha, r-squared
        return results.params[ref],results.params['const'],results.rsquared
        
    
    def sortino_calc(self):
        '''calculates denominator of sortino ratio'''
        neg_rets = self.returns[self.returns < 0]**2
        denom = np.sqrt(neg_rets.sum()/len(self.returns))*np.sqrt(252)
        return denom
            
    def Hist_VaR(self,percentile):
        '''calculates historical value at risk for given percentile'''
        return self.returns.quantile(q=(1-percentile),interpolation='lower')
    
    def Hist_CVaR(self):
        '''calculate conditional value at risk using portfolio value at risk'''
        temp_data = self.returns[self.returns<self.VaR]
        return temp_data.mean()
        
    def calculate_rets(self):
        '''calculate portfolio returns based on positions'''
        rets = self.positions*Returns[self.assets]
        rets = rets.dropna()
        self.asset_returns_wgt = rets
        self.returns = rets.sum(axis=1)
        self.returns.name = self.code
        
    def plot_dd(self,startdate,enddate):
        '''plot drawdown plot'''
        if not startdate:
            startdate = self.firstday
        if not enddate:
            enddate = self.lastday        
        ax = self.dd.plot()
        plt.ylabel('Drawdown (%)')
        plt.title('Underwater Chart - ' + self.name +" Portfolio")  
        #format y-axis as percentage
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])          
        plt.tight_layout()
        plt.fill_between(self.dd.index,self.dd.values, alpha=0.8)
        plt.show()          
        
    def yearly_returns(self,returns):
        '''calculate yearly returns'''
        rets = returns + 1
        rets = rets.resample('A').prod()-1
        return rets
    
    def plot_rets(self, startdate, enddate):
        '''plot cumulative returns from start to end date'''
        if not startdate:
            startdate = self.firstday
        if not enddate:
            enddate = self.lastday
            
        print('Plotting...')
        cumul_rets = (1+self.returns[(self.returns.index>=startdate) & (self.returns.index<=enddate)]).cumprod()-1
        ax = cumul_rets.plot()
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.title('Cumulative Returns - ' + self.name +" Portfolio")  
        #format y-axis as percentage
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])          
        plt.tight_layout()
        plt.show()                    
   
    def tearsheet(self,startdate,enddate):
        #8.5X11 so that can fit on standard page
        plt.rcParams["figure.figsize"] = (8.5,11)  
        
        #Chart 1: Cumulative Return
        cum_returns = (1 + self.returns).cumprod()-1
        ind_cum_returns  = (1+self.index_rets).cumprod()-1
        ax1 = plt.subplot2grid((11, 3), (0, 0), colspan=3,rowspan=3)
        ax1.plot(cum_returns,label='Portfolio')
        ax1.plot(ind_cum_returns,linestyle='-',label='S&P 500',color='#000000',linewidth=0.8,alpha=0.7)
        ax1.set_ylabel('Return (%)',fontsize=9)
        ax1.set_title("Portfolio Tearsheet: " + self.name,y=1.1)  
        ax1.margins(x=0,y=0)
        y_range_add1 = (cum_returns.max()-cum_returns.min())/20
        y_range_add2 = (ind_cum_returns.max()-ind_cum_returns.min())/20
        y_range_add = max(y_range_add1,y_range_add2)
        ax1.set_ylim(min(ind_cum_returns.min(),cum_returns.min())-y_range_add,max(ind_cum_returns.max(),cum_returns.max())+y_range_add)
        ax1.grid(linestyle='--',alpha=0.5,linewidth=0.7)
        ax1.set_axisbelow(True)
        #format y-axis as percentage
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
        ax1.legend(loc='upper left',fontsize=8,handlelength=0.8)
        ax1.set_facecolor('#FFFFFF')
        
        
        #Chart 2: Drawdown Graph
        ax2 = plt.subplot2grid((11, 3), (3, 0), colspan=3,rowspan=2)
        ax2.plot(self.dd, color = 'red')
        ax2.margins(x=0,y=0)
        ax2.grid(linestyle='--',alpha=0.5,linewidth=0.7)
        #format y-axis as percentage
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))  
        ax2.set_ylim(self.dd.min()*1.05,0)
        ax2.set_ylabel('Drawdown (%)',fontsize=9)
        ax2.set_facecolor('#FFFFFF')          
        ax2.set_axisbelow(True)
        ax2.fill_between(self.dd.index,self.dd.values, alpha=0.5,color='red')
        
        #Table 1: Basic Info
        ax6= plt.subplot2grid((11, 3), (7, 0), rowspan=2, colspan=1)
        ax6.text(0.5, 8.5, 'Portfolio Code:', fontsize=9)
        ax6.text(9.5 , 8.5, self.code, horizontalalignment='right', fontsize=9)   
        ax6.text(0.5, 7.0, 'Start Date:', fontsize=9)
        ax6.text(9.5 , 7.0, self.firstday.date(), horizontalalignment='right', fontsize=9)
        ax6.text(0.5, 5.5, 'End Date:', fontsize=9)
        ax6.text(9.5 , 5.5, self.lastday.date(), horizontalalignment='right', fontsize=9)   
        ax6.text(0.5, 4.0, 'Rebalanced:', fontsize=9)
        ax6.text(9.5 , 4.0, self.rebal, horizontalalignment='right', fontsize=9)
        ax6.text(0.5, 2.5, 'Trend Following:', fontsize=9)
        ax6.text(9.5 , 2.5, self.trend, horizontalalignment='right', fontsize=9)    
        ax6.text(0.5, 1, 'Weighting:', fontsize=9)
        ax6.text(9.5, 1, self.wgt_method, horizontalalignment='right', fontsize=9)              
        ax6.set_title('Overview',fontsize=10)
        ax6.grid(False)
        ax6.spines['top'].set_linewidth(0.75)
        ax6.spines['bottom'].set_linewidth(0.75)
        ax6.spines['right'].set_visible(False)
        ax6.spines['left'].set_visible(False)
        ax6.get_yaxis().set_visible(False)
        ax6.get_xaxis().set_visible(False)
        ax6.set_ylabel('')
        ax6.set_xlabel('')
        ax6.axis([0, 10, 0, 10])    
        
        #Table 2: Basic Statitics
        ax4 = plt.subplot2grid((11, 3), (7, 1), rowspan=2, colspan=1)
        ax4.text(0.5, 8.5, 'Total Return:', fontsize=9)
        ax4.text(9.5 , 8.5, '{:.2%}'.format(self.tot_ret), horizontalalignment='right', fontsize=9)   
        ax4.text(0.5, 7.0, 'CAGR:', fontsize=9)
        ax4.text(9.5 , 7.0, '{:.2%}'.format(self.exp_ret), horizontalalignment='right', fontsize=9)
        ax4.text(0.5, 5.5, 'Annual Volatility:', fontsize=9)
        ax4.text(9.5 , 5.5, '{:.2%}'.format(self.vol), horizontalalignment='right', fontsize=9)   
        ax4.text(0.5, 4.0, 'Sharpe:', fontsize=9)
        ax4.text(9.5 , 4.0, '{:.2f}'.format(self.sharpe), horizontalalignment='right', fontsize=9)
        ax4.text(0.5, 2.5, 'Max Drawdown:', fontsize=9)
        ax4.text(9.5 , 2.5, '{:.2%}'.format(self.maxdd), horizontalalignment='right', fontsize=9)    
        ax4.text(0.5, 1, 'Sortino:', fontsize=9)
        ax4.text(9.5, 1, '{:.2f}'.format(self.sortino), horizontalalignment='right', fontsize=9)        
        ax4.set_title('Statistics',fontsize=10)
        ax4.grid(False)
        ax4.spines['top'].set_linewidth(0.75)
        ax4.spines['bottom'].set_linewidth(0.75)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.get_yaxis().set_visible(False)
        ax4.get_xaxis().set_visible(False)
        ax4.set_ylabel('')
        ax4.set_xlabel('')
        ax4.axis([0, 10, 0, 10])                
        
        #Table 3: Additional Statistics
        ax6= plt.subplot2grid((11, 3), (7, 2), rowspan=2, colspan=1)
        ax6.text(0.5, 8.5, r'$VaR_{99\%}:$', fontsize=9)
        ax6.text(9.5 , 8.5, '{:.2%}'.format(self.VaR), horizontalalignment='right', fontsize=9)   
        ax6.text(0.5, 7.0, r'$CVaR_{99\%}:$', fontsize=9)
        ax6.text(9.5 , 7.0, '{:.2%}'.format(self.CVaR), horizontalalignment='right', fontsize=9)
        ax6.text(0.5, 5.5, 'Beta:', fontsize=9)
        ax6.text(9.5 , 5.5, '{:.2f}'.format(self.beta), horizontalalignment='right', fontsize=9)   
        ax6.text(0.5, 4.0, 'Alpha:', fontsize=9)
        ax6.text(9.5 , 4.0, '{:.2%}'.format(self.alpha), horizontalalignment='right', fontsize=9)
        ax6.text(0.5, 2.5, 'R-Squared:', fontsize=9)
        ax6.text(9.5 , 2.5, '{:.2%}'.format(self.R2), horizontalalignment='right', fontsize=9)    
        ax6.text(0.5, 1, 'Treynor:', fontsize=9)
        ax6.text(9.5, 1, '{:.2f}'.format(self.treynor), horizontalalignment='right', fontsize=9)         
        ax6.set_title('Statistics #2',fontsize=10)
        ax6.grid(False)
        ax6.spines['top'].set_linewidth(0.75)
        ax6.spines['bottom'].set_linewidth(0.75)
        ax6.spines['right'].set_visible(False)
        ax6.spines['left'].set_visible(False)
        ax6.get_yaxis().set_visible(False)
        ax6.get_xaxis().set_visible(False)
        ax6.set_ylabel('')
        ax6.set_xlabel('')
        ax6.axis([0, 10, 0, 10])     
        
        # yearly returns chart
        ax7 = plt.subplot2grid((11, 3), (9, 2), rowspan=2, colspan=1)
        yearly_rets = self.yearly_returns(self.returns)
        ax7.bar(yearly_rets.index.strftime("%Y"),yearly_rets.values,width=0.8,alpha=1)
        #format y-axis as percentage
        ax7.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
        ax7.set_xticklabels(yearly_rets.index.strftime("%Y"),rotation = '70',horizontalalignment='center',fontsize=7)
        #ax7.xaxis_date()
        ax7.set_facecolor('#FFFFFF')
        ax7.set_axisbelow(True)
        ax7.grid(linestyle='--',alpha=0.5,linewidth=0.7)
        ax7.set_title('Yearly Returns (%)',fontsize=10)
        
        #Positioning Chart
        ax5 = plt.subplot2grid((11, 3), (5, 0), rowspan=2, colspan=3)
        ax5.stackplot(self.positions.index,self.positions.T,labels = self.positions.columns,alpha=1)
        ax5.grid(linestyle='--',alpha=0.5,linewidth=0.7,axis='y')
        ax5.set_axisbelow(True)
        ax5.margins(x=0,y=0)
        ax5.set_ylim(0,1)
        ax5.legend(loc=8,ncol=10,mode=None,bbox_to_anchor=(0., 1.02, 1., .102),fontsize=7,edgecolor="#FFFFFF",handlelength=0.6)
        #ax5.legend(loc=8,ncol=len(list(self.positions)),mode=None,bbox_to_anchor=(0., 1.02, 1., .102),fontsize=7,edgecolor="#FFFFFF",handlelength=0.6)
        ax5.set_ylabel('Weight (%)',fontsize=9)
        
        #format y-axis as percentage
        ax5.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))        
        ax5.set_facecolor('#FFFFFF')          
        
        #Heatmap
        ax10 = plt.subplot2grid((11, 3), (9, 0), rowspan=2, colspan=2)
        assets_rets = self.yearly_returns(self.asset_returns_wgt)
        sns.heatmap(assets_rets.T, linewidth=0.5, yticklabels=True,ax=ax10,xticklabels=assets_rets.index.strftime("%Y"), center=0, annot=False, cbar=False, fmt='.1%', cmap='RdYlGn',annot_kws={"size": 6.5})
        #ax10.set_yticklabels(ax10.get_yticklabels(),rotation=0,fontsize=8)
        ax10.set_yticklabels(ax10.get_yticklabels(),rotation=0,fontsize=5)
        ax10.set_xticklabels(ax10.get_xticklabels(),fontsize=8,rotation = 70)
        ax10.tick_params(axis='both',bottom=False,left=False)
        ax10.set_xlabel('')
        ax10.set_title('Performance Attribution',fontsize=10)
        
        #adjust plot layout    
        plt.tight_layout()
        
        #save tearsheet
        plt.subplots_adjust(left=0.115, right=0.95, top=0.93,bottom=0.07)
        plt.savefig('Tearsheet.pdf')
        
        #reset figsize to standard
        plt.rcParams["figure.figsize"] = (12,7)        

def load_data(fname, Prices):
    '''load in historical prices'''
    ticker = fname[0:-6]
    data = pd.read_csv("Data/ETF_adjusted/" + fname)
    data.set_index('Date', inplace=True)
    data = data[["Adj Close"]]
    data.columns = [ticker]
    data.index = pd.to_datetime(data.index)    
    Prices = pd.concat([Prices, data], axis=1, sort=True)
    return Prices
 
def EW_TF_positions(Prices,rebal_freq,rolling_window):
    '''determine daily postions in equal weight portfolio with trend following overlay'''
    Prices = Prices.dropna(how='any')
    positions = pd.DataFrame(columns = Prices.columns)
    assets = list(Prices)
    #calculat equal weights
    ew = 1/len(assets)    
    #calculate moving average
    SMA = Prices.rolling(window=rolling_window).mean()
    SMA = SMA.iloc[200:]
    #determine positions
    
    #for daily rebalancing
    if rebal_freq == 'D':
        for day in SMA.index:
            for asset in SMA.columns:
                #identify trending assets
                if Prices.loc[day,asset] >= SMA.loc[day,asset]:
                    positions.loc[day,asset] = 1
                else:
                    positions.loc[day,asset] = 0
        return positions.shift(1).dropna(how='any')*ew 
    #for other reblancing frequencies
    else:
        day = SMA.index[0]
        new_day = day
        while day <= SMA.index[-1]:
            if day in SMA.index:
                if day >= new_day:
                    #rebalance time
                    #determine trending assets
                    for asset in SMA.columns:
                        if Prices.loc[day,asset] >= SMA.loc[day,asset]:
                            positions.loc[day,asset] = 1
                        else:
                            positions.loc[day,asset] = 0
                    #calculate postions
                    positions.loc[day] = positions.loc[day]*ew
                    last_pos = positions.loc[day]
                    #determine next rebalance date
                    if rebal_freq == 'M':
                        new_day = day + pd.DateOffset(months=1)
                    if rebal_freq == 'Y':
                        new_day = day + pd.DateOffset(years=1)
                    day = day + timedelta(days=1)
                else:
                    #update postion based on returns of asset
                    temp_data = last_pos*(1+Returns[assets].loc[day])
                    #normalize weights
                    if temp_data.astype(bool).sum() == len(assets):
                        positions.loc[day] = temp_data/temp_data.sum()
                    else:
                        positions.loc[day] = temp_data                        
                    last_pos = positions.loc[day]
                    #go to next day
                    day = day + timedelta(days=1)
            else:
                #go to next day
                day = day + timedelta(days=1)           
        #return postions
        return positions.shift(1).dropna(how='any')        
    
def EW_positions(Prices,rebal_freq='D'):
    '''determine daily positions in equal weight portfolio'''
    Prices = Prices.dropna(how='any')
    assets = list(Prices)
    ew = 1/len(assets)
    #daily rebalancing
    if rebal_freq == 'D':
        #set all postions to ew
        Prices[:] = ew
        positions = Prices
    #other rebalancing frequencies
    else:
        positions = pd.DataFrame(columns = Prices.columns)
        day = Prices.index[0]
        new_day = day
        while day <= Prices.index[-1]:
            if day in Prices.index:
                if day >= new_day:
                    #rebalance to equal weights
                    positions.loc[day] = ew
                    last_pos = ew
                    #determine next reblance day
                    if rebal_freq == 'M':
                        new_day = day + pd.DateOffset(months=1)
                    if rebal_freq == 'Y':
                        new_day = day + pd.DateOffset(years=1)
                    if rebal_freq == 'W':
                        new_day = day + pd.DateOffset(weeks=1)                    
                    day = day + timedelta(days=1)
                else:
                    #update weights based on returns of assets
                    positions.loc[day] = last_pos*(1+Returns[assets].loc[day])
                    last_pos = positions.loc[day]
                    #go to next day
                    day = day + timedelta(days=1)  
            else:
                #go to next day
                day = day + timedelta(days=1)            
    return realloc(positions.shift(1).dropna(how='any'))


def risk_parity_generator_V2(Prices,rebal_freq,TF=None, rolling_window=None,static=False,target=None,cash='SHV'):
    '''generates positions for risk parity portfolio'''
    #initalize data variables
    Prices = Prices.dropna(how='any')
    assets = list(Prices)
    assets_cash = assets.copy()
    assets_cash.remove(cash)
    positions = pd.DataFrame(columns = assets)
    pos_check = pd.DataFrame(columns = assets)
    #calculate risk parity weights using all data
    static_weights = erc_ver1.get_weights(Prices[assets_cash].pct_change().dropna(how='any'),target)
    
    #if trend following option selected
    if TF:
        #calculate moving averages
        SMA = Prices.rolling(window=rolling_window).mean()
        SMA = SMA.iloc[rolling_window:]
        day = SMA.index[0]
        new_day = day
        while day <= SMA.index[-1]:
            if day in SMA.index:
                if day >= new_day:
                    #rebalance
                    if static:
                        weights = static_weights
                    else:
                        #determine weights based on rolling windo of data
                        weights = erc_ver1.get_weights(Prices[assets_cash].loc[Prices.index <= day].iloc[-1*rolling_window::].pct_change().dropna(how='any'),target)
                    for asset in SMA.columns:
                        #determine trending assets
                        if Prices.loc[day,asset] >= SMA.loc[day,asset]:
                            positions.loc[day,asset] = 1
                        else:
                            positions.loc[day,asset] = 0
                    #determine position for the day
                    positions.loc[day,cash]=0
                    last_pos = positions.loc[day]*weights
                    pos_check.loc[day] = last_pos
                    last_pos.loc[cash] = 1 - last_pos.sum()
                    positions.loc[day] = last_pos
                    
                    #determine next rebalance date
                    if rebal_freq == 'M':
                        new_day = day + pd.DateOffset(months=1)
                    if rebal_freq == 'Y':
                        new_day = day + pd.DateOffset(years=1)
                    #go to next day
                    day = day + timedelta(days=1)
                else:
                    #update weights based on asset returns
                    temp_data = last_pos*(1+Returns[assets].loc[day])
                    positions.loc[day] = temp_data/temp_data.sum()
                    last_pos = positions.loc[day]                    
                    #go to next date
                    day = day + timedelta(days=1)          
            else:
                #go to next date
                day = day + timedelta(days=1)
        return positions.shift(1).dropna(how='any')
    else:
        #no trend following
        positions = pd.DataFrame(columns = assets_cash)
        day = Prices[assets_cash].index[rolling_window]
        new_day = day
        while day <= Prices.index[-1]:
            if day in Prices.index:
                if day >= new_day:
                    if static:
                        weights = static_weights
                    else:
                        #calculate weights based on rolling window lookback
                        weights = erc_ver1.get_weights(Prices[assets_cash].loc[Prices.index <= day].iloc[-1*rolling_window::].pct_change().dropna(how='any'),target)
                    positions.loc[day] = weights
                    last_pos = weights
                    #determine next rebalance date
                    if rebal_freq == 'M':
                        new_day = day + pd.DateOffset(months=1)
                    if rebal_freq == 'Y':
                        new_day = day + pd.DateOffset(years=1)
                    #go to next day
                    day = day + timedelta(days=1)
                else:
                    #update weights based on asset returns
                    temp_data = last_pos*(1+Returns[assets_cash].loc[day])
                    #normalize weights
                    if temp_data.astype(bool).sum() == len(assets_cash):
                        positions.loc[day] = temp_data/temp_data.sum()
                    else:
                        positions.loc[day] = temp_data  
                    last_pos = positions.loc[day]
                    #go to next day
                    day = day + timedelta(days=1)                    
            else:
                #go to next day
                day = day + timedelta(days=1)
        return positions.shift(1).dropna(how='any')        

def realloc(positions):
    '''normalizes position weights so that they sum to 1'''
    positions = positions.loc[:,:].div(positions.sum(axis=1),axis=0)
    return positions
        
def inverse_vol(returns):
    '''calculates inverse volatility weights'''
    inv_vol = 1/returns.std()
    weights = inv_vol/inv_vol.sum()
    return weights

if __name__ == "__main__":
    
    print("Started...")
    
    #plotting styles
    plt.rcParams["figure.figsize"] = (12,7)    
    plt.rcParams.update({'font.size': 9})
    plt.rcParams.update({'mathtext.default':  'regular' })
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#aec7e8','#ffbb78','#98df8a','#ff9896','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#F1C40F','#85929E','#D7BDE2','#1ABC9C','#D6EAF8'])
    
    #load prices
    Prices = pd.DataFrame()
    for fname in os.listdir("Data/ETF_adjusted"):
        Prices = load_data(fname,Prices).dropna(how='any')
    
    #calculate returns
    Returns = Prices.pct_change().dropna(how='any')
    
    #load in risk free rate
    risk_free = pd.read_csv("Data/RF.csv")
    risk_free.set_index('Date', inplace=True)
    risk_free.index = pd.to_datetime(risk_free.index)   
    risk_free = risk_free['Rate']
    
    #preprocess risk_free rate data
    Temp = pd.concat([Returns,risk_free], axis=1,sort=True)
    Temp.dropna(subset = list(Returns),how='any',inplace=True)
    Temp.fillna(method='ffill',inplace=True)
    risk_free = Temp['Rate']
    
    #set Leverage (1 means no leverage)
    leverage = 1.0
     
    #calculate daily fee and transaction cost adjustemnts
    fee_adj = ((1.005)**(1/252))-1
    t_cost = ((1.00192)**(1/252))-1        

    for i in list(Returns):
        if i not in ['SPX','SPY']:
            #adjust return series to incoporate transaction costs and leverage costs
            Returns[i] = Returns[i]*leverage - (leverage-1)*(risk_free + fee_adj) - (leverage)*t_cost
                
    print("Prices and returns loaded!")
    
    #Determine Positions
    
    assets = ['MTUM','VLUE','QUAL','SIZE','USMV','IVLU','IMTM','IQLT','EMGF','ACWV','SPTL','AGG','EMB','TIP','HYG','SCHH','DBC','GLD','SHV']
    #risk targets
    target = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.12, 0.08, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]        
    #generate portfolio positions
    RP_pos = risk_parity_generator_V2(Prices[assets],'M',TF=True, rolling_window=200,static=False,target=target)
    #create portfolio with positions 
    RP_Port = portfolio("Conservative","RP 1.5x","Risk parity portfolio with dynamic weights reblanced monthly",RP_pos, '200 SMA','Monthly','RP 200')    

    print("Done!")
    