# -*- coding: utf-8 -*-
"""
========
Sugawara
========
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

INITIAL_STATES = [10, 10]
INITIAL_Q = 1.0
INITIAL_PARAM = [0.5, 0.2, 0.01, 0.1, 10.0, 20.0, 1]

X_LB = [0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.8]
X_UB = [1.1, 1.1, 1.5, 1.1, 15.0, 1.0, 1.6]

def _step(prec, evap, st, param, extra_param):
    '''
    ======
    Step
    ======
    
    This function makes a single step fowrward using the Sugawara model.
    
    Parameters
    ----------
    prec : float
        Average precipitation [mm/h]
    evap : float
        Potential Evapotranspiration [mm/h] 
    st : array_like [2]
        Model states [mm]. Corresponds to the level in the top and bottom tank 
        respectively.
    param : array_like [6]
        Model parameters as:
        k1: Upper tank upper discharge coefficient
        k2: Upper tank lower discharge coefficient
        k3: Percolation to lower tank coefficient
        k4: Lower tank discharge coefficient
        d1: Upper tank upper discharge position
        d2: Upper tank lower discharge position
    extra_param : array_like [2]
        Problem parameter vector setup as:
            DT: Number of hours in the time step
            AREA: Catchment area [km²]
    
    Returns
    -------
    q_new : float
        Discharge [m3/s]
    st_new : array_like [2]
        Posterior model states
    '''

    # Old states
    S1Old = st[0]
    S2Old = st[1]

    #Parameters
    k1 = param[0]
    k2 = param[1]
    k3 = param[2]
    k4 = param[3]
    d1 = param[4]
    d2 = param[5]
    rfcf = param[6]
    ecorr = param[7]
    
    # Extra Parameters
    DT = extra_param[0]
    Area = extra_param[1]

    ## Top tank
    H1 = np.max([S1Old + prec*rfcf - evap*ecorr, 0])

    if H1 > 0:
        #direct runoff
        if H1 > d1:
            q1 = k1*(H1-d1)
        else:
            q1 = 0

        #Fast response component
        if H1 > d2:
            q2 = k2*(H1-d2)
        else:
            q2 = 0

        #Percolation to bottom tank
        q3 = k3 * H1
        #Check for availability of water in upper tank
        q123 = q1+q2+q3
        if q123 > H1:
            q1 = (q1/q123)*H1
            q2 = (q2/q123)*H1
            q3 = (q3/q123)*H1
    else:
        q1 = 0
        q2 = 0
        q3 = 0

    Q1 = q1+q2
    #State update top tank
    S1New = H1 - (q1+q2+q3)
    
    ## Bottom tank
    H2 = S2Old+q3
    Q2 = k4* H2

    #check if there is enough water
    if Q2 > H2:
        Q2 = H2

    #Bottom tank update
    S2New = H2 - Q2

    ## Total Flow
    # DT = 86400 #number of seconds in a day
    # Area = 2100# Area km²
    if (Q1 + Q2) >= 0:
        q_new = (Q1+Q2)*Area/(3.6*DT)
    else:
        q_new = 0

    st_new = [S1New, S2New]
    if S1New < 0:
        print('s1 below zero')
    return q_new, st_new

def simulate(prec, evap, param, extra_param):
    '''
    ======
    Simulate
    ======
    
    This function makes the simulation of a complete time seties using the 
    Sugawara model.
    
    Parameters
    ----------
    prec : array_like [n]
        Average precipitation [mm/h]
    evap : array_like [n]
        Potential Evapotranspiration [mm/h] 
    param : array_like [6]
        Model parameters as:
        k1: Upper tank upper discharge coefficient
        k2: Upper tank lower discharge coefficient
        k3: Percolation to lower tank coefficient
        k4: Lower tank discharge coefficient
        d1: Upper tank upper discharge position
        d2: Upper tank lower discharge position
    extra_param : array_like [2]
        Problem parameter vector setup as:
        DT: Number of hours in the time step
        AREA: Catchment area [km²]
    
    Returns
    -------
    q : array_like [n]
        Discharge [m3/s]
    st : array_like [n, 2]
        Posterior model states
    '''
    st = [INITIAL_STATES,]
    q = [10,]

    for i in xrange(len(prec)):
        step_res = _step(prec[i], evap[i], st[i], param, extra_param)
        q.append(step_res[0])
        st.append(step_res[1])

    return q, st

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


def _rmse(q_rec, q_sim):
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
    q_sim, st_sim = simulate(prec, evap, param, extra_param)
    
    plt.figure()
    plt.plot(q_sim)
    plt.xlabel('Time step [hr]')
    plt.ylabel('Discharge [m3/s]')
    plt.grid()
    
    plt.figure()
    plt.plot(np.array(st_sim)[:, 0], label='UT')
    plt.plot(np.array(st_sim)[:, 1], label='LT')
    plt.legend()
    plt.grid()
    plt.xlabel('Time step [hr]')
    plt.ylabel('State [mm]')