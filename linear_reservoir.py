# -*- coding: utf-8 -*-
"""
===========
Linear Tank
===========
Lumped hydrological model.

This is the Sugawara (TANK) hydrological model implementation by Juan Chacon at 
IHE-Delft, NL. This code implements a two tank the version with linear response

@author: Juan Carlos Chacon-Hurtado (jc.chaconh@gmail.com)                                  

Version
-------
03-05-2017 - V_0.0 - First implementation
"""
from __future__ import division
import numpy as np
import scipy.optimize as opt


INITIAL_Q = 1.0
INITIAL_PARAM = [0.01, 1.0]
X_LB = [0.0001, 0.7]
X_UB = [1.0, 1.7]

#%%
def _step(prec_step, evap_step, q_old, param, extra_param):
    '''
    ======
    Step
    ======
    
    This function makes a single step fowrward using the linear tank model.
    
    Parameters
    ----------
    prec_step : float
        Average precipitation [mm/h]
    evap_step : float
        Potential Evapotranspiration [mm/h] 
    param : array_like [1]
        Parameter vector, set up as:
        Rainfall correction factor
        Recession coefficient K
    extra_param : array_like [2]
        Problem parameter vector setup as:
            DT: Number of hours in the time step
            AREA: Catchment area [km²]
    
    Returns
    -------
    q_new : float
        Discharge [m3/s]
        
    '''
    # Transformation of precipitation into inflow (m³/hr)
    inp = np.max([(prec_step*param[1] - evap_step)*extra_param[1]*1000.0, 0])
    
    # Get discharge in m³/hr
    q_sim = ((q_old*3600.0)*np.exp(-param[0]*extra_param[0]) + 
            inp*(1.0 - np.exp(-param[0]*extra_param[0])))
    
    # Change to m³/s    
    q_sim = q_sim/3600.0
    return q_sim

def simulate(avg_prec, evap, param, extra_param):
    '''
    ========
    Simulate
    ========

    Run the HBV model for the number of steps (n) in precipitation. The
    resluts are (n+1) simulation of discharge as the model calculates step n+1
    
    Parameters
    ----------
    avg_prec : array_like [n]
        Average precipitation [mm/h]
    evap : array_like [n]
        Potential Evapotranspiration [mm/h]
    param : array_like [1]
        Parameter vector, set up as:
        Rainfall correction factor
        Recession coefficient K
    extra_param : array_like [2]
        Problem parameter vector setup as:
        [tfac, area]

    Returns
    -------
    q_sim : array_like [n]
        Discharge for the n time steps of the precipitation vector [m3/s]
        
    '''
    q_sim = [INITIAL_Q, ]

    for i in xrange(len(prec)):
        step_res = _step(avg_prec[i], evap[i], q_sim[i], param, extra_param)
        q_sim.append(step_res)

    return q_sim

def _nse(q_rec, q_sim):
    '''
    ===
    NSE
    ===
    
    Nash-Sutcliffe efficiency. Metric for the estimation of performance of the 
    hydrological model
    
    Parameters
    ----------
    q_rec : array_like [n]
        Measured discharge [m3/s]
    q_sim : array_like [n] 
        Simulated discharge [m3/s]
        
    Returns
    -------
    f : float
        NSE value
    '''
    a = np.square(np.subtract(q_rec, q_sim))
    b = np.square(np.subtract(q_rec, np.nanmean(q_rec)))
    if a.any < 0.0:
        return(np.nan)
    f = 1.0 - (np.nansum(a)/np.nansum(b))
    return f


def _rmse(q_rec,q_sim):
    '''
    ====
    RMSE
    ====
    
    Root Mean Squared Error. Metric for the estimation of performance of the 
    hydrological model.
    
    Parameters
    ----------
    q_rec : array_like [n]
        Measured discharge [m3/s]
    q_sim : array_like [n] 
        Simulated discharge [m3/s]
        
    Returns
    -------
    f : float
        RMSE value
    '''
    erro = np.square(np.subtract(q_rec,q_sim))
    if erro.any < 0:
        return(np.nan)
    f = np.sqrt(np.nanmean(erro))
    return f

def calibrate(prec, evap, extra_param, q_rec, x_0=None, x_lb=X_LB, x_ub=X_UB, 
              obj_fun=_rmse, wu=10, minimise=True, verbose=False):
    '''
    ======
    Calibrate
    ======
    
    This function makes the calibration of the Sugawara hydrological model.
    
    Parameters
    ----------
    prec : array_like [n]
        Average precipitation [mm/h]
    evap : array_like [n]
        Potential Evapotranspiration [mm/h] 
    extra_param : array_like [2]
        Problem parameter vector setup as:
        DT: Number of hours in the time step
        AREA: Catchment area [km²]
    q_rec : array_like [n]
        Measurements of discharge [m3/s]
    x_0 : array_like [18], optional
        First guess of the parameter vector. If unspecified, a random value
        will be sampled between the boundaries of the parameter set
    x_lb : array_like [18], optional
        Lower boundary of the parameter vector. 
    x_ub : array_like [18], optional
        First guess of the parameter vector.  
    obj_fun : function, optional
        Function that takes 2 parameters, recorded and simulated discharge. If
        unspecified, RMSE is used.
    wu : int, optional
        Warming up period. This accounts for the number of steps that the model
        is run before calculating the performance function.
    minimise : bool, optional
        If True, the optimisation corresponds to the minimisation of the 
        objective function. If False, the optimial of the objective function is
        maximised.
    verbose : bool, optional
        If True, displays the result of each model evaluation when performing
        the calibration of the hydrological model.
    Returns
    -------
    param : array_like [6]
        Optimal parameter set
        k1: Upper tank upper discharge coefficient
        k2: Upper tank lower discharge coefficient
        k3: Percolation to lower tank coefficient
        k4: Lower tank discharge coefficient
        d1: Upper tank upper discharge position
        d2: Upper tank lower discharge position
    performance : float
        Optimal value of the objective function
    '''

    def cal_fun(param_cal):
        q_sim = simulate(prec[:-1], evap[:-1], param_cal, extra_param)[0]
        try:
            if minimise:
                perf_fun = obj_fun(q_rec[wu:], q_sim[wu:])
                if verbose: print perf_fun
            else:
                perf_fun = -obj_fun(q_rec[wu:], q_sim[wu:])
                if verbose: print -perf_fun
        
        except:
            perf_fun = np.nan
            if verbose: print perf_fun

        return perf_fun
    
    if x_0 is None:
        # Randomly generated
        x_0 = np.random.uniform(x_lb, x_ub)
    
    # Boundaries
    x_b = zip(x_lb, x_ub)
    
    cal_res = opt.minimize(cal_fun, INITIAL_PARAM, bounds=x_b,
                           method='L-BFGS-B')

    return cal_res.x, cal_res.fun



if __name__ == '__main__':
    '''
    Testing function    
    '''
    import matplotlib.pyplot as plt
    #    for i in xrange(1000):
    prec = np.random.uniform(0, 100, 1000)
    evap = np.random.uniform(0, 10, 1000)
    st = [np.random.uniform(0, 30), np.random.uniform(0, 30)]
    param = [0.1819, 0.0412, 0.3348, 0.0448, 3.2259, 0.3800,1,1, 1]
    extra_param = [1.0, 145.0]
    q_sim = simulate(prec, evap, param, extra_param)
    
    plt.figure()
    plt.plot(q_sim)
    plt.xlabel('Time step [hr]')
    plt.ylabel('Discharge [m3/s]')
    plt.grid()
    
