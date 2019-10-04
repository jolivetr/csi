'''Time utilities for time-series InSAR analysis.

.. author:

    Piyush Agram <piyush@gps.caltech.edu>
    
.. Dependencies:

    numpy, datetime, scipy.factorial, logmgr'''

import numpy as np
import datetime as dt
import sys
try:
    from scipy import factorial
except ImportError:
    from scipy.special import factorial
from numpy import arange as xrange

###########################Time-series utils##############################
def nCk(n,k):
    '''Combinatorial function.'''
    c = factorial(n)/(factorial(n-k)*factorial(k)*1.0)
    return c


def bspline_nu(n,tk,t):
    '''Non uniform b-splines.
       n - spline order
       tk - knots
       t - time vector'''

    B = np.zeros((len(tk),len(t)))
    assert (n+1) < len(tk), 'Not enough knots for order {} bspline'.format(n)

    for m in xrange(len(tk)-1):
        ind = where((t>=tk[m]) & (t<tk[m+1]))
        B[m,ind] = 1.0

    for p in xrange(n):
        for q in xrange(len(tk)-2-p):
            B[q,:] = ((t-tk[q])/(tk[p+q+1]-tk[q]))*B[q,:] + ((tk[p+q+2]-t)/(tk[p+q+2]-tk[q+1]))*B[q+1,:]

    ihalf = np.int_((n+1)/2)
    B = B[ihalf:len(tk)-ihalf,:]

    return B


def bspline(n,dtk,t):
    '''Uniform b-splines.
       n    -  Order
       dtk  -  Spacing
       t    -  Time vector'''
    x = (t/dtk) + n +1
    b = np.zeros(len(x))
    for k in range(n+2):
        m = x-k-(n+1)/2
        up = np.power(m,n)
        b = b+((-1)**k)*nCk(n+1,k)*up*(m>=0)

    b = b/(1.0*factorial(n))
    return b

def ispline(n,dtk,t):
    '''Uniform integrated b-splines
       n   - Order
       dtk - Spacing
       t   - Time vector'''
    x = (t/dtk)+n+1
    b = np.zeros(len(x))
    for k in range(n+2):
        m = x-k-(n+1)/2
        up = m**(n+1)
        b += ((-1)**k)*nCk(n+1,k)*up*(m>=0)

    b = b*dtk/((n+1.0)*factorial(n))
    return b


def Timefn(rep,t):
    '''Interprets a list as time-series representation and returns time function matrix.

    Args:
       
       * rep   - Representation of the functions (cnt).
       * t     - Time vector.

    Returns:
        
       * H      - a time-series matrix of size (Nsar x cnt)
       * vname  - Unique name for each of the model parameters
       * rflag  - Regularization family number for each model parameter'''
    Nsar = len(t)
    Nrep = len(rep)
    cnt = 0         #Number of model parameters
    H = []          #Greens functions
    rflag = []      #Regularization flag
    vname = []      #Parameter name
    regstart = 1    #Set of params that need to be regularized together
    for k in xrange(Nrep):
        fn = rep[k]	
        fname = fn[0].upper()
        if fname in ('LINEAR'):  #f = (t-t1)
            num = len(fn) - 1
            assert num==1, 'Undefined LINEAR sequence.'

            ts = fn[1]

            for m in xrange(len(ts)):
                hfn = (t-ts[m])
                vn = 'LINE/{:2.3f}'.format(ts[m])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)

        elif fname in ('LINEAR_FINITE'):
            num = len(fn) - 1
            assert num==1, 'Undefined LINEAR_FINITE sequence.'
            for trange in fn[1]:
                hfn = (t - trange[0]) * ((t >= trange[0]) & (t <= trange[1]))
                vn = 'LINEFIN/{:2.3f}/{:2.3f}'.format(trange[0], trange[1])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)
                
        elif fname in ('POLY'):
            num = len(fn) - 1
            
            assert num==2, 'Undefined POLY sequence.'
            
            order = fn[1]
            ts = fn[2]
            
            assert len(order) == len(ts), 'POLY: Orders and times dont match'
            for p in xrange(len(order)):
                g = (t-ts[p])
                for m in xrange(order[p]+1):
                    hfn = g**m
                    vn = 'P/{d}/{:2.1f}'.format(m,ts[p])
                    rf = 0.0
                    H.append(hfn)
                    vname.append(vn)
                    rflag.append(rf)
                
        elif fname in ('QUADRATIC'): 
            num = len(fn) - 1
            
            assert num==1, 'Undefined QUADRATIC sequence'
            
            ts = fn[1]
            for m in xrange(len(ts)):
                hfn = (t-ts[m])**2
                vn = 'QUAD/{:2.3f}'.format(ts[m])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)

        elif fname in ('OFFSET'): # constant offset
            num = len(fn) - 1
            if (num != 1):
                print('Undefined sequence: {}'.format(fn))
                print('Eg: [[\'OFFSET\'],[t_dummy]]')
                sys.exit(1)
            ts = fn[1]
            H.append(np.ones(t.shape, dtype=float))
            vname.append('OFFSET')
            rflag.append(0.0)

        elif fname in ('EXP'):   #f = (1-exp(-(t-t1)/tau1))*u(t-t1)
            num = len(fn) - 1
            assert num == 2, 'Undefined EXP sequence.'

            ts = fn[1]
            taus = fn[2]
            assert len(ts) == len(taus), 'EXP: Times and Taus dont match'

            for m in xrange(len(ts)):
                hfn = (1 - np.exp(-(t-ts[m])/taus[m]))*(t>=ts[m])
                vn = 'EXP/{:2.3f}/{:2.3f}'.format(ts[m],taus[m])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)

        elif fname in ('LOG'): #f = log(1+(t-t1)/tau1)*u(t-t1)
            num = len(fn) - 1
            assert num == 2, 'Undefined LOG sequence.'

            ts = fn[1]
            taus = fn[2]

            assert len(ts) == len(taus), 'LOG: Times and Taus dont match'

            for m in xrange(len(ts)):
                hfn = np.log(1.0+ ((t-ts[m])/taus[m])*(t>=ts[m]))
                vn = 'LOG/{:2.3f}/{:2.3f}'.format(ts[m],taus[m])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)

        elif fname in ('STEP'): #f = u(t-t1)
            num = len(fn) - 1
            assert num==1, 'Undefined STEP sequence.'

            ts = fn[1]

            for m in xrange(len(ts)):
                hfn = 1.0*(t>=ts[m])
                vn = 'STEP/{:2.3f}'.format(ts[m])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)

        elif fname in ('SEASONAL'): # f = cos(t/tau1) , sin(t/tau1)
            num = len(fn) - 1
            assert num == 1, 'Undefined SEASONAL sequence.'

            taus = fn[1]

            for m in xrange(len(taus)):
                #hfn = 1-np.cos(2*np.pi*t/taus[m])
                hfn = np.cos(2*np.pi*t/taus[m])
                vn = 'COS/{:2.3f}'.format(taus[m])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)
                hfn = np.sin(2*np.pi*t/taus[m])
                vn = 'SIN/{:2.3f}'.format(taus[m])
                rf = 0.0
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)

        elif fname in ('BSPLINE','BSPLINES'): #Currently only uniform splines.
            num = len(fn) - 1
            assert num == 2, 'Undefined BSPLINE sequence.'

            orders = fn[1]
            nums = fn[2]


            assert len(orders) == len(nums), 'BSPLINE: Orders and Numbers dont match. '

            for m in xrange(len(orders)):
                ts = np.linspace(t.min(),t.max(),nums[m])
                dtk = ts[2] - ts[1]
                for p in xrange(len(ts)):
                    hfn = bspline(orders[m],dtk,t-ts[p])
                    vn = 'Bsp/{}/{}'.format(p,orders[m])
                    rf = regstart
                    H.append(hfn)
                    vname.append(vn)
                    rflag.append(rf)
                    
            regstart = regstart+1   

        elif fname in ('ISPLINE','ISPLINES'): #Currently only uniform splines.
            num = len(fn) - 1

            assert num==2, ' Undefined ISPLINE sequence.'

            orders = fn[1]
            nums = fn[2]

            assert len(orders) == len(nums), 'Orders and Numbers dont match.'

            for m in xrange(len(orders)):
                ts = np.linspace(t.min(),t.max(),nums[m])
                dtk = ts[2] - ts[1]
                for p in xrange(len(ts)):
                    hfn = ispline(orders[m],dtk,t-ts[p])
                    vn = 'Isp/{}/{}'.format(p,orders[m])
                    rf = regstart
                    H.append(hfn)
                    vname.append(vn)
                    rflag.append(rf)
                    
                    
            regstart = regstart+1

        elif fname in ('SBAS'): #[[['SBAS'],[ind]]]
            num = len(fn)-1
            assert num == 1, 'Undefined SBAS sequence.'

            master = fn[1]
            num = len(t)

            for m in xrange(num):
                hfn = np.zeros(num)
                if m < master:
                    hfn[m:master] = -1
                elif m > master:
                    hfn[master+1:m+1] = 1

                rf = 0.0
                vn = 'SBAS/{}/{}'.format(m,master)
                H.append(hfn)
                vname.append(vn)
                rflag.append(rf)

    H = np.array(H)
    H = H.transpose()       #####For traditional column-wise representation.
    vname = np.array(vname)
    rflag = np.array(rflag)
    return  H,vname,rflag

def mName2Rep(mName):
	''' From mName given by TimeFn, returns the equivalent function representation

	Args:
		* mName   -> list of the model names

	Returns:
		* rep     -> list of parametric functions'''

	rep = []	

	m = 0
	while m<len(mName):

		# Get the model name
		model = mName[m].split('/')

		if model[0] in ('LINE'):			# Case Linear
			r = ['LINEAR',[np.float(model[1])]]
			rep.append(r)

		elif model[0] in ('LINEFIN'):
			r = ['LINEAR_FINITE',[[np.float(model[1]),np.float(model[2])]]]
			rep.append(r)

		elif model[0] in ('P'):				# Case Polynom
	
			# Check how many orders in the poly function
			tm = 1
			polyflag = True
			while polyflag:
				if m+tm==len(mName):
					polyflag = False
				else:
					tmodel = mName[m+tm].split('/')
					if tmodel[0] in ('P') and tmodel[1] not in ('0'):
						polyflag = True
						tm+=1
					else:
						polyflag = False

			tm = tm - 1

			r = ['POLY',[tm],[np.float(model[2])]]
			rep.append(r)
		
			m = m + tm 

		elif model[0] in ('QUAD'):			# Case Quadratic
			r = ['QUADRATIC',[np.float(model[1])]]
			rep.append(r)

		elif model[0] in ('OFFSET'):			# Case Offset
			r = ['OFFSET',[0]]
			rep.append(r)

		elif model[0] in ('EXP'):			# Case exponential
			t1 = np.float(model[1])
			tau = np.float(model[2])
			r = ['EXP',[t1],[tau]]
			rep.append(r)

		elif model[0] in ('LOG'):			# Case Logarithm
			t1 = np.float(model[1])
			tau = np.float(model[2])
			r = ['LOG',[t1],[tau]]
			rep.append(r)
	
		elif model[0] in ('STEP'):			# Case step function
			t1 = np.float(model[1])
			r = ['STEP',[t1]]
			rep.append(r)

		elif model[0] in ('COS'):			# Case seasonal
			tau = np.float(model[1])
			r = ['SEASONAL',[tau]]
			rep.append(r)
			m+=1

		elif model[0] in ('Bsp'):			# Case Bspline
			
			# Check how many B-Splines is there
			tm = 1
			bspflag = True
			while bspflag:
				if m+tm==len(mName):
					bspflag = False
				else:
					tmodel = mName[m+tm].split('/')
					if tmodel[0] in ('Bsp') and tmodel[1] not in ('0'):
						bspflag = True
						tm+=1
					else:
						bspflag = False

			order = np.int(model[2])
			r = ['BSPLINE',[order],[tm]]
			rep.append(r)
			m = m + tm - 1

		elif model[0] in ('Isp'):			# Case ISpline

			# Check How many I-splines is there
			tm = 1
			ispflag = True
			while ispflag:
				if m+tm==len(mName): 
					ispflag = False
				else:
					tmodel = mName[m+tm].split('/')
					if tmodel[0] in ('Isp') and tmodel[1] not in ('0'): 
						ispflag = True
						tm += 1
					else:
						ispflag = False
			
			order = np.int(model[2])
			r = ['ISPLINE',[order],[tm]] 
			rep.append(r)  
			m = m + tm - 1

		elif model[0] in ('SBAS'):
			
			# Check how many SBAS pieces is there
			tm = 1
			sbasflag = True
			while sbasflag:
				if m+tm==len(mName):
					sbasflag=False
				else:
					tmodel = mName[m+tm].split('/')
					if tmodel[0] in ('SBAS') and tmodel[1] not in ('0'):
						sbasflag = True
						tm += 1
					else:
						sbasflag = False

			master = np.int(model[2])
			r = ['SBAS',master]
			rep.append(r)
			m = m + tm - 1 
			
		# Increase the pointer
		m+=1

	return rep
			
		
	

#######################Time-series utils##################################################
