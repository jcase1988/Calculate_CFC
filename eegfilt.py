from __future__ import division # default division is float division
__author__ = 'johncase'

import scipy.io as sio
import numpy as np
import scipy.signal as sig


from numpy import asarray, array, arange, append, angle, complex128, float64, floor
from numpy import nonzero, sign, mat, sin, cos, exp, zeros, log10, unique, fix, ceil
from numpy.linalg import inv
from numpy import ones, prod, pi, NaN, zeros_like, ravel, any, linspace, diff, roots, kron
from numpy.fft import fft, ifft
from scipy.signal import convolve, freqz, zpk2tf, tf2zpk, remez, get_window
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz, hankel


def firls(m, bands, desired, weight=None):

    if weight==None : weight = ones(len(bands)/2)
    bands, desired, weight = array(bands), array(desired), array(weight)

    #if not desired[-1] == 0 and bands[-1] == 1 and m % 2 == 1:
    if m % 2 == 1:
        m = m + 1

    M = m/2
    w = kron(weight, [-1,1])
    omega = bands * pi
    i1 = arange(1,M+1)

    # generate the matrix q
    # as illustrated in the above-cited reference, the matrix can be
    # expressed as the sum of a hankel and toeplitz matrix. a factor of
    # 1/2 has been dropped and the final filter hficients multiplied
    # by 2 to compensate.
    cos_ints = append(omega, sin(mat(arange(1,m+1)).T*mat(omega))).reshape((-1,omega.shape[0]))
    q = append(1, 1.0/arange(1.0,m+1)) * array(mat(cos_ints) * mat(w).T).T[0]
    q = toeplitz(q[:M+1]) + hankel(q[:M+1], q[M : ])

    # the vector b is derived from solving the integral:
    #
    #           _ w
    #          /   2
    #  b  =   /       w(w) d(w) cos(kw) dw
    #   k    /    w
    #       -      1
    #
    # since we assume that w(w) is constant over each band (if not, the
    # computation of q above would be considerably more complex), but
    # d(w) is allowed to be a linear function, in general the function
    # w(w) d(w) is linear. the computations below are derived from the
    # fact that:
    #     _
    #    /                          a              ax + b
    #   /   (ax + b) cos(nx) dx =  --- cos (nx) +  ------ sin(nx)
    #  /                             2                n
    # -                             n
    #


    enum = append(omega[::2]**2 - omega[1::2]**2, cos(mat(i1).T * mat(omega[1::2])) - cos(mat(i1).T * mat(omega[::2]))).flatten()
    deno = mat(append(2, i1)).T * mat(omega[1::2] - omega[::2])
    cos_ints2 = enum.reshape(deno.shape)/array(deno)

    d = zeros_like(desired)
    d[::2]  = -weight * desired[::2]
    d[1::2] =  weight * desired[1::2]

    b = append(1, 1.0/i1) * array(mat(kron (cos_ints2, [1, 1]) + cos_ints[:M+1,:]) * mat(d).T)[:,0]

    # having computed the components q and b of the  matrix equation,
    # solve for the filter hficients.
    a = (array(inv(q)*mat(b).T).T)[0]
    h = append( a[:0:-1], append(2*a[0],  a[1:]))

    return h

def eegfilt(data,srate,locutoff,hicutoff,*arg):
#                                    ...,epochframes,filtorder,revfilt,firtype,causal)

    if data.ndim == 1:
        chans = 1
        frames = data.shape[0]
    else:
        chans,frames = data.shape

    nyq = srate*0.5 # Nyquist frequency
    #MINFREQ = 0.1/nyq;
    MINFREQ = 0
    minfac         = 3    # this many (lo)cutoff-freq cycles in filter
    min_filtorder  = 15   # minimum filter length
    trans          = 0.15 # fractional width of transition zones


    if locutoff>0 and hicutoff > 0 and locutoff > hicutoff and not not locutoff and not not hicutoff:
        raise ValueError('locutoff > hicutoff ??\n')
    if locutoff < 0 or hicutoff < 0 and not not locutoff and not not hicutoff:
        raise ValueError('locutoff | hicutoff < 0 ???\n')
    if locutoff>nyq and not not locutoff:
        raise ValueError('Low cutoff frequency cannot be > srate/2')
    if hicutoff>nyq and not not hicutoff:
        raise ValueError('High cutoff frequency cannot be > srate/2')

    #Gather filter parameters if specified in args
    if len(arg) < 2:
        filtorder = 0
    else:
        filtorder = arg[1]
    if len(arg) < 3:
        revfilt = 0
    else:
        revfilt = arg[2]
    if len(arg) < 4:
        firtype = 'firls'
    else:
        firtype = arg[3]
    if len(arg) < 5:
        causal = 0
    else:
        causal = arg[4]

    if not filtorder:
        if locutoff > 0 and not not locutoff:
            filtorder = minfac*int(srate/locutoff)
        elif hicutoff > 0 and not not hicutoff:
            filtorder = minfac*int(srate/hicutoff)

        if filtorder < min_filtorder:
            filtorder = min_filtorder


    if len(arg) < 1:
        epochframes = 0
    if epochframes == 0:
        epochframes = frames

    epochs = int(frames/epochframes)

    if not epochs*epochframes == frames:
        raise ValueError('epochframes does not divide frames.\n')

    if filtorder*3 > epochframes:
        print 'eegfilt(): filter order is %d.' % (filtorder)
        raise ValueError('eochframes must be at least 3 times the filtorder.')

    if not not hicutoff and (1+trans)*hicutoff/nyq > 1:
        raise ValueError('high cutoff frequency too close to Nyquist frequency')

    if locutoff > 0 and hicutoff > 0 and not not locutoff and not not hicutoff: #bandpass filter
        if revfilt:
            print 'eegfilt() - performing %d-point notch filter.\n' % (filtorder)
        else:
            print 'eegfilt() - performing %d-point bandpass filter.\n' % (filtorder)
        print('            If a message, ''Matrix is close to singular or badly scaled,'' appears,\n')
        print('            then Python has failed to design a good filter. As a workaround, \n')
        print('            for band-pass filtering, first highpass the data, then lowpass it.\n')

        if firtype == 'firls':
            f = [MINFREQ, round((1-trans)*locutoff/nyq,5), round(locutoff/nyq,5), round(hicutoff/nyq,5), round((1+trans)*hicutoff/nyq, 1,5)]
            print 'eegfilt() - low transition band width is %e Hz; high trans. band width, %e Hz.\n' % ((f[2]-f[1])*srate/2.0, (f[4]-f[3])*srate/2.0)
            m = [0,       0,                      1,           1,            0,                     0]
        #elif firtype == 'firl':
             #filtwts = fir1(filtorder, [locutoff, hicutoff]./(srate/2));

    elif locutoff > 0 and not not locutoff: # highpass filter
        if locutoff/nyq < MINFREQ:
            raise ValueError('eegfilt() - highpass cutoff freq must be > %e Hz\n\n' % (MINFREQ*nyq))
   #     print 'eegfilt() - performing %d-point highpass filtering.\n' % filtorder

        if firtype == 'firls':
            f=[MINFREQ, round(locutoff*(1-trans)/nyq,5), round(locutoff/nyq,5), 1]
   #         print 'eegfilt() - highpass transition band width is %e Hz.\n' %((f[2]-f[1])*srate/2.0)
            m=[   0,             0,                   1,      1]
        #elseif strcmp(firtype, 'fir1')
        #    filtwts = fir1(filtorder, locutoff./(srate/2), 'high');

    elif hicutoff > 0 and not not hicutoff: # lowpass filter
        if hicutoff/nyq < MINFREQ:
            raise ValueError('eegfilt() - lowpass cutoff freq must be > %g Hz' % (MINFREQ*nyq))
    #    print 'eegfilt() - performing %d-point lowpass filtering.\n' % filtorder
        if firtype == 'firls':
            f=[MINFREQ, round(hicutoff/nyq,5), round(hicutoff*(1+trans)/nyq,5), 1]
    #        print 'eegfilt() - lowpass transition band width is %e Hz.\n' % ((f[2]-f[1])*srate/2.0)
            m=[     1,           1,              0 ,                0]
        #elseif strcmp(firtype, 'fir1')
        #    filtwts = fir1(filtorder, hicutoff./(srate/2));

    else:
        raise ValueError('You must provide a non-0 low or high cut-off frequency')

    if revfilt:
        m = map(lambda x:int(not x),m) #MATLAB: m = ~m;

    filtwt = firls(filtorder,f,m) # get FIR filter coefficients

       
    smoothdata = np.zeros([chans,frames])
    for e in range(0,epochs):
        for c in range(0,chans):
            #if causal:
                #smoothdata
            if chans == 1:
                smoothdata[e*epochframes:(e+1)*epochframes] = sig.filtfilt(filtwt,[1],data[e*epochframes:(e+1)*epochframes])
            else:
                smoothdata[c,e*epochframes:(e+1)*epochframes] = sig.filtfilt(filtwt,[1],data[c,e*epochframes:(e+1)*epochframes])

    return smoothdata,filtwt


