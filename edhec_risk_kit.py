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