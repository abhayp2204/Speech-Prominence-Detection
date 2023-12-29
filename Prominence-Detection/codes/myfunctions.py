#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:09:22 2022

@author: iiit
"""
import numpy
import numpy as np
import wave
import struct
import math
from scipy.signal import butter, lfilter


#Auxilliary functions for feature computation
def spectral_selection(x, n):
    y = x.shape
    row = y[0]
    col = y[1]
    xx = []
    for i in range(0,col,1):
        v = x[:,i]
        v = numpy.array([v])
        v = v.T
        t = numpy.array(numpy.arange(1,row+1)).reshape(-1,1)
        v = numpy.hstack((v, t))
        v_sort = v[v[:,0].argsort(),]
        v_sort_sel = v_sort[row-n:row, :]
        vv = v_sort_sel[v_sort_sel[:,1].argsort(),]
        #tt = numpy.array([vv[:,0]])
        if i!=0:
            if i==1:
                pp = numpy.array([xx])
                pp = pp.T
            else:
                pp = xx
            pp2 = numpy.array([vv[:,0]])
            pp2 = pp2.T
            xx = numpy.hstack((pp, pp2))
        else:
            xx = numpy.concatenate((xx, vv[:,0]))
    return xx


def temp_vec_corr(x2, t_sigma):
    from scipy.stats import norm
    y = x2.shape
    row = y[0]
    col = y[1]
    wn = norm.pdf(np.arange(1,col+1,1), (col+1)/2, t_sigma)
    # NOTE: if we use continue to manipulate the variable x2, (the function argument), then it gets reflected back in
    # in the parent function. (No idea why). So create a copy of x2 and work with that.
    x3 = np.zeros((row,col))
    for i in range(0,row,1):
        x3[i,:] = np.multiply(x2[i,:],wn)
    s=0
    for i in range(0,col-1,1):
        for j in range(i+1,col,1):
            s+= np.multiply(x3[:,i], x3[:,j])
    if col!=1:
        s = np.sqrt(np.divide(s, (col-1)*col/2))
    else:
        s = x3
    return s

def temporal_corr(x, win, t_sigma):
    hwin = (win-1)/2
    yy = x.shape
    row = yy[0]
    col = yy[1]

    row = int(row)
    hwin = int(hwin)

    x = np.array([np.concatenate((np.zeros((row,hwin)), x, np.zeros((row, hwin))), axis = 1)])
    y = []
    for i in range(hwin,col+hwin,1):
        temp2 = x[0,:,i-hwin:i+hwin+1]
        z = temp_vec_corr(temp2, t_sigma)
        z = np.array([z]).T
        if i==hwin:
            y = np.concatenate((y, z[:,0]))
        else:
            if i==hwin+1:
                y = np.array([y]).T
            y = np.hstack((y, z))
    return y

def spectral_corr(x):
    yy = x.shape
    row = yy[0]
    col = yy[1]

    s = np.zeros((1, col))
    for i in range(0, row-1, 1):
        for j in range(i+1, row, 1):
            s = s+np.multiply(x[i,:], x[j,:])

    if row!=1:
        s = np.sqrt(np.divide(s, (row*(row-1)/2)))
    else:
        s = x
    return s

def statFunctions_Syl(t):
    from scipy.stats.mstats import gmean
    if np.min(t)<0:
        t = np.subtract(t,min(t[0]))
        #out = []
        #return out
    out = np.array([np.median(t[0]), np.mean(t[0]), gmean(np.absolute(t[0])), np.max(t[0])-np.min(t[0]), np.std(t[0])])
    out = np.array([out]).T
    t = np.subtract(t,np.min(t[0]))
    t = np.divide(t, np.sum(t[0]))
    tempArr = np.array([np.arange(1,len(t[0])+1)])
    temporalMean = np.sum(np.multiply(tempArr,t)[0])
    temporalStd = np.sqrt(np.sum(np.multiply(np.power(np.subtract(np.array([np.arange(1,len(t[0])+1)]),temporalMean),2),t[0])))
    temporalSkewness = np.sum(np.divide(np.multiply(np.power(np.subtract(np.array([np.arange(1,len(t[0])+1)]),temporalMean),3),t[0]),np.power(temporalStd,3)))
    temporalKurthosis = np.sum(np.divide(np.multiply(np.power(np.subtract(np.array([np.arange(1,len(t[0])+1)]),temporalMean),4),t[0]),np.power(temporalStd,4)))
    arr1 = np.array([np.array([temporalStd, temporalSkewness, temporalKurthosis])]).T
    out = np.vstack((out,arr1))
    return out

def statFunctions_Vwl(t):
    if np.min(t)<0:
        t = np.subtract(t,min(t[0]))
        #out = []
        #eturn out
    out = np.array([np.median(t[0]), np.mean(t[0]), np.max(t[0])-np.min(t[0]), np.std(t[0])])
    out = np.array([out]).T
    t = np.subtract(t,np.min(t[0]))
    t = np.divide(t, np.sum(t[0]))
    tempArr = np.array([np.arange(1,len(t[0])+1)])
    temporalMean = np.sum(np.multiply(tempArr,t)[0])
    temporalStd = np.sqrt(np.sum(np.multiply(np.power(np.subtract(np.array([np.arange(1,len(t[0])+1)]),temporalMean),2),t[0])))
    temporalSkewness = np.sum(np.divide(np.multiply(np.power(np.subtract(np.array([np.arange(1,len(t[0])+1)]),temporalMean),3),t[0]),np.power(temporalStd,3)))
    temporalKurthosis = np.sum(np.divide(np.multiply(np.power(np.subtract(np.array([np.arange(1,len(t[0])+1)]),temporalMean),4),t[0]),np.power(temporalStd,4)))
    arr1 = np.array([np.array([temporalStd, temporalSkewness, temporalKurthosis])]).T
    out = np.vstack((out,arr1))
    return out

def smooth(t_cor, swin, sigma):
    from scipy.stats import norm
    ft = norm.pdf(np.arange(1,swin+1), (swin+1)/2, sigma)
    ft = np.array([ft])
    t_cor = np.array([t_cor])
    convRes = np.zeros((1, t_cor.shape[2]+ft.shape[1]-1))
    convRes = np.convolve(t_cor[0,0,:], ft[0,:])
    y = convRes[np.arange((swin+1)//2-1, len(convRes)-(swin-1)//2, 1)]
    return y

def get_labels(lab_list,fa,fileName):
        L=[]; fb=fa; filenm=[];
        
        for num in range(0,len(lab_list)):
            if str((lab_list[num][0].tolist())[0]) == str('P'):
                L.append(1)
                filenm.append(fileName)           
            else:
                L.append(0)
                filenm.append(fileName)
        fb = np.vstack((fa,L))
#        fb = np.vstack((fb,np.asarray(filenm,object)))
        return fb,filenm

def get_labels_seq2seq(lab_list):
        L=[];# filenm=[];
        
        for num in range(0,len(lab_list)):
            if str((lab_list[num][0].tolist())[0]) == str('P'):
                L.append(1)
#                filenm.append(fileName)           
            else:
                L.append(0)
                #filenm.append(fileName)
        #fb = np.vstack((fa,L))
#        fb = np.vstack((fb,np.asarray(filenm,object)))
        return L
    
def vocoder_func(wavPath):

    # FILTER DEFINITIONS

    def butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5*fs
        low = float(lowcut) / nyq
        high = float(highcut) / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_lowpass(lowcut, fs, order):
        nyq = 0.5*fs
        low = float(lowcut) / nyq
        b ,a = butter(order, low, btype='lowpass')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_lowpass_filter(data, lowcut, fs, order):
        b, a = butter_lowpass(lowcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    # FUNCTION TO READ A .wav FILE MATLAB STYLE

    def readWav(wavPath):
        waveFile = wave.open(wavPath)
        fs = waveFile.getframerate()
        length = waveFile.getnframes()
        data = []
        for i in range(0, length):
            waveData = waveFile.readframes(1)
            data.append(struct.unpack("<h", waveData))
        waveFile.close()
        data = np.array([data])
        data = data.astype(float)/np.max(np.abs(data))
        data = data[0]
        return data, fs, length

    # BUFFER FUNCTION AS DEFINED IN MATLAB

    def buffer(x, n, p=0, opt=None):
        import numpy
        if p >= n:
            raise ValueError('p ({}) must be less than n ({}).'.format(p,n))
        cols = int(numpy.ceil(len(x)/float(n-p)))+1
        if opt == 'nodelay':
            cols += 1
        elif opt != None:
            raise SystemError('Only `None` (default initial condition) and '
                              '`nodelay` (skip initial condition) have been '
                              'implemented')
        b = numpy.zeros((n, cols))
        j = 0
        for i in range(cols):
            if i == 0 and opt == 'nodelay':
                b[0:n,i] = x[0:n]
                continue
            elif i != 0 and p != 0:
                b[:p, i] = b[-p:, i-1]
            else:
                b[:p, i] = 0
            k = j + n - p
            n_end = p+len(x[j:k])
            b[p:n_end,i] = x[j:k,0]
            j = k
        return b

    fltcF= np.array([240,360,480,600,720,840,1000,1150,1300,1450,1600,1800,2000,2200,2400,2700,3000,3300,3750])
    fltBW= np.array([120,120,120,120,120,120,150,150,150,150,150,200,200,200,200,300,300,300,500])

    fltFc= np.array([np.subtract(fltcF,np.divide(fltBW,2)),np.add(fltcF,np.divide(fltBW,2))])
    fltLpFc= 50

    sig, Fs, length = readWav(wavPath)

    # Saving the audio in a txt file
    xx = np.append(Fs,sig)

    nWndw = int(round(Fs*0.02))
    nOverlap = int(round(Fs*0.01))
    sig = 0.99*sig/max(abs(sig))

    # Windowing first and filtering next
    sigFrames= buffer(sig*32768,nWndw,nOverlap)
    subBandEnergies= np.zeros([19,sigFrames.shape[1]])

    for j in range(0,sigFrames.shape[1]):
        currFrame = np.array([sigFrames[:,j]])
        for i in range(0,fltFc.shape[1]):
            fltFrame = butter_bandpass_filter(currFrame[0], fltFc[0][i], fltFc[1][i], Fs, 2); fltFrame = fltFrame.T
            rectFrame = np.abs(fltFrame[0:nWndw])
            lpFltFrame = butter_lowpass_filter(rectFrame, float(fltLpFc), Fs, 2)
            currEnergy = lpFltFrame[nWndw-1]
            if currEnergy < 1:
                currEnergy = 0.5
            subBandEnergies[i,j] = math.exp(2*math.log(currEnergy)/math.log(10))
    subBandEnergies = np.concatenate((np.exp(0.5*np.ones((19,1))),subBandEnergies[:,0:-2]),axis=1).T

    return subBandEnergies, xx