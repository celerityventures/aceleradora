import pandas as pd
def drawdown(return_series):
    """
    Input returns series, returns wealth index, peaks and drawdown
    """
    wealth_index=1000*(1+return_series).cumprod()
    previous_peaks=wealth_index.cummax()
    drawdowns=(wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth":wealth_index,
        "Peaks":previous_peaks,
        "Drawdown":drawdowns
    })

def get_ffme_returns():
    """
    load portfo
    """
    
    me_m=pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",index_col=0,parse_dates=True, na_values=-99.99)
    rets=me_m[['Lo 10', 'Hi 10']]
    rets.columns=['SmallCap','LargeCap']
    rets=rets/100
    rets.index=pd.to_datetime(rets.index,format="%Y%m")
    return rets

def get_hfi_returns():
    """
    load portfo
    """
    
    hfi=pd.read_csv("data/edhec-hedgefundindices.csv",header=0,index_col=0,parse_dates=True,infer_datetime_format=True)
    hfi=hfi/100
    hfi.index=hfi.index.to_period("M")
    return hfi

def semidev(r):
    """return semid"""
    return r[r<0].std()

import scipy.stats

def is_normal(r,level=0.01):
    """Jarque vera test
    """
    statistic, p_value=scipy.stats.jarque_bera(r)
    return p_value>level
import numpy as np
def var_historic(r,level=5):
    """
    VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")
        
import scipy.stats

def var_gaussian(r, level=5, modified=False):
    """
    Return parametric gaussian VaR of a Series or DataFrame
    """
    z=scipy.stats.norm.ppf(level/100)
    if modified:
        #modify based on skewness and kurtosis
        s=r.skew()
        k=r.kurt()
        z=(z+ (z**2-1)*s/6+
           (z**3-3*z)*(k-3)/24-
           (2*z**3-5*z)*(s**2)/36)
    return -(r.mean()+z*r.std(ddof=0))

def cvar_historic(r, level=5):
    """
    Compute the Conditional VaR of Series or DataFrame
    """
    if isinstance(r,pd.Series):
        is_beyond=r<=-var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r,pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or Dataframe")
        
def get_ind_returns():
    """
    Load and format Ken French 30 industry Portfolios Value Weightes Monthly Returns
    """
    ind=pd.read_csv("data/ind30_m_vw_rets.csv", header=0,index_col=0, parse_dates=True )/100
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    ind.columns=ind.columns.str.strip()
    return ind

def get_ind_size():
    """
    """
    ind=pd.read_csv("data/ind30_m_size.csv", header=0,index_col=0, parse_dates=True )
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    ind.columns=ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    """
    """
    ind=pd.read_csv("data/ind30_m_nfirms.csv", header=0,index_col=0, parse_dates=True )
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    ind.columns=ind.columns.str.strip()
    return ind

def annualize_rets(r,periods_per_year):
    """ annualizes set of returns"""
    compounded_growth=(1+r).prod()
    n_periods=r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r,periods_per_year):
    """
    Annualizes vol of set of returns based on periods per year
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """annualized sharpe ratio"""
    rf_per_period=(1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret=r-rf_per_period
    ann_ex_ret=annualize_rets(excess_ret,periods_per_year)
    ann_vol=annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def portfolio_return(weights,returns):
    """weights to returns"""
    return weights.T@returns
def portfolio_vol(weights,covmat):
    """weights to vol"""
    return (weights.T@covmat@weights)**0.5

def plot_ef2(n_points,er,cov,style=".-"):
    """
    Plot the 2-asset frontier
    """
    if er.shape[0] !=2 or cov.shape[0] !=2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights=[np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    rets=[portfolio_return(w,er)for w in weights]
    vols=[portfolio_vol(w,cov) for w in weights]
    ef=pd.DataFrame({
        "Returns":rets,
        "Volatility":vols})
    return ef.plot.line(x="Volatility",y="Returns",style=style)

import numpy as np

from scipy.optimize import minimize
def minimize_vol(target_return,er,cov):
    """
    target ret->W
    """
    n=er.shape[0]
    init_guess=np.repeat(1/n,n)
    bounds=((0.0,1.0),)*n
    return_is_target={
        'type':'eq',
        'args':(er,),
        'fun':lambda weights,er: target_return-portfolio_return(weights,er)
    }
    weights_sum_to_1={
        'type':'eq',
        'fun':lambda weights:np.sum(weights)-1
    }
    results=minimize(portfolio_vol,init_guess,
                     args=(cov,),method="SLSQP",
                     options={'disp':False},
                     constraints=(return_is_target,weights_sum_to_1),
                     bounds=bounds
                    )
    return results.x


import pandas as pd
def optimal_weights(n_points,er,cov):
    """list of weights to run optimizer"""
    target_rs=np.linspace(er.min(),er.max(),n_points)
    weights=[minimize_vol(target_return,er,cov) for target_return in target_rs]
    return weights

def msr(riskfree_rate,er,cov):
    """
    RiskFree rate + ER + COV->W
    """
    n=er.shape[0]
    init_guess=np.repeat(1/n,n)
    bounds=((0.0,1.0),)*n
    weights_sum_to_1={
        'type':'eq',
        'fun':lambda weights:np.sum(weights)-1
    }
    def neg_sharpe_ratio(weights,riskfree_rate,er,cov):
        """
        Returns negative of sharpe ratio
        """
        r=portfolio_return(weights,er)
        vol=portfolio_vol(weights,cov)
        return -(r-riskfree_rate)/vol
    
    results=minimize(neg_sharpe_ratio,init_guess,
                     args=(riskfree_rate,er,cov,),method="SLSQP",
                     options={'disp':False},
                     constraints=(weights_sum_to_1),
                     bounds=bounds
                    )
    return results.x


def gmv(cov):
    """
    returns the weights of the Global Minimum Volatility portfolio
    given the covariance matrix
    """
    n=cov.shape[0]
    return msr(0,np.repeat(1,n),cov)
def plot_ef(n_points,er,cov, show_cml=False,style=".-",riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plot the N-asset frontier
    """
   
    weights=optimal_weights(n_points,er,cov)
    rets=[portfolio_return(w,er)for w in weights]
    vols=[portfolio_vol(w,cov) for w in weights]
    ef=pd.DataFrame({
        "Returns":rets,
        "Volatility":vols})
    ax= ef.plot.line(x="Volatility",y="Returns",style=style)
    if show_ew:
        # show equal weights
        n=er.shape[0]
        w_ew=np.repeat(1/n,n)
        r_ew=portfolio_return(w_ew,er)
        vol_ew=portfolio_vol(w_ew,cov)
        # display equal weights
        ax.plot([vol_ew],[r_ew],color="goldenrod",marker="o",markersize=10)
    if show_gmv:
        # show glob min variance
        w_gmv=gmv(cov)
        r_gmv=portfolio_return(w_gmv,er)
        vol_gmv=portfolio_vol(w_gmv,cov)
        # display gmv
        ax.plot([vol_gmv],[r_gmv],color="midnightblue",marker="o",markersize=10)
    if show_cml:
        ax.set_xlim(left=0)
        rf=0.1
        w_msr=msr(riskfree_rate,er,cov)
        r_msr=portfolio_return(w_msr,er)
        vol_msr=portfolio_vol(w_msr,cov)
        # add cml
        cml_x=[0,vol_msr]
        cml_y=[riskfree_rate,r_msr]
        ax.plot(cml_x,cml_y,color="green",marker="o",linestyle="dashed",markersize=12,linewidth=2)
    return ax

#%load_ext autoreload
#%autoreload 2
#%matplotlib inline
#import edhec_risk_kit as erk