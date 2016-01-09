def scan_freq(subj,block,phase_elec,amp_elec,surrogate_analysis):
    import scipy.io as sio
    from numpy import arange,array,append,zeros,pi,angle,logical_and,mean,roll,save #for efficiency
    from eegfilt import eegfilt
    from scipy.signal import hilbert
    from math import log
    import pickle
    import os.path
    from random import randint

    #data_path = '/home/jcase/data/' + subj + block + '/' + subj + '_' + block + '_data.mat'
    #subglo_path = '/home/jcase/data/subj_globals.mat'
    #MI_output_path = '/home/jcase/data/' + subj + block + '/MI/e' + str(phase_elec) + '_e' + str(amp_elec)

    data_path = '/Users/johncase/Documents/UCSF Data/' + subj + '/' + subj + block + '/data/' + subj + '_' + block + '_data.mat'
    subglo_path = '/Users/johncase/Documents/UCSF Data/subj_globals.mat'
    MI_output_path = '/Users/johncase/Documents/UCSF Data/' + subj + '/' + subj + block + '/analysis/MI/e' + str(phase_elec) + '_e' + str(amp_elec)

    

    #Load ECOG Data
    ecog_data = sio.loadmat(data_path)['ecogData']
    amp_raw_data = ecog_data[amp_elec-1,:]
    pha_raw_data = ecog_data[phase_elec-1,:]
    del ecog_data

    #Load subject globals
    all_subj_data = sio.loadmat(subglo_path,struct_as_record=False, squeeze_me=True)['subj_globals']
    subj_data = getattr(getattr(all_subj_data,subj),block)
    srate = subj_data.srate
    per_chan_bad_epochs = subj_data.per_chan_bad_epochs
    allstimtimes = subj_data.allstimtimes
 
    #Surrogate Runs
    kruns = 10000

    #Phase-providing frequency
    #fp = arange(1,15.1,0.1)
    #fp_bandwidth = arange(0.5,5.1,0.1)

    fp = arange(1,21,1)
    fp_bandwidth = arange(1,11,1)

    #fp = arange(2,3)
    #fp_bandwidth = arange(2,3)

    #Amplitude-providing frequency
    fa = array([70,150])

    #Define phase bins
    n_bins = 20
    bin_size = 2*pi/n_bins
    bins = arange(-pi,pi-bin_size,bin_size)

    #Define time_window (roughly entire block, exclude artifacts samples later)
    t_0 = int(round(allstimtimes[0,0]*srate))
    t_end = int(round((allstimtimes[-1,1]+3) *srate))
    t_win = arange(t_0,t_end)

    #Determine samples with artifacts
    bad_samp = array([])
    if per_chan_bad_epochs[phase_elec-1].size == 2:
        bad_samp = append(bad_samp,arange(srate*per_chan_bad_epochs[phase_elec-1][0],srate*per_chan_bad_epochs[phase_elec-1][1]))
    else:
        for epoch in per_chan_bad_epochs[phase_elec-1]:
            bad_samp = append(bad_samp,arange(srate*epoch[0],srate*epoch[1]))

    if not phase_elec == amp_elec:
        if per_chan_bad_epochs[amp_elec-1].size == 2:
            bad_samp = append(bad_samp,arange(srate*per_chan_bad_epochs[amp_elec-1][0],srate*per_chan_bad_epochs[amp_elec-1][1]))
        else:
            for epoch in per_chan_bad_epochs[amp_elec-1]:
                bad_samp = append(bad_samp,arange(srate*epoch[0],srate*epoch[1]))

    #good_samps
    good_samps = list(set(t_win)-set(bad_samp))

    #Do high-gamma filtering
    pow,filtwt = eegfilt(amp_raw_data,srate,fa[0],[])
    pow,filtwt = eegfilt(pow[0],srate,[],fa[1])
    pow = abs(hilbert(pow[0][0:len(pow[0])-1]))
    pow = pow[good_samps] #exclude bad` samples

    #Calculate MI for each phase-providing central-frequencies / bandwidths
    MI = zeros([len(fp),len(fp_bandwidth)])
    MI_surrogate = zeros([len(fp),len(fp_bandwidth),kruns])

    for iFreq,freq in enumerate(fp):
        for iBand,band in enumerate(fp_bandwidth):

            if freq-(band/2) < 0.5:
                MI[iFreq,iBand] = 0
                continue

            print('freq = ' + str(freq) + ', bw = ' + str(band))

            #Do phase-providing phase filtering

            pha = zeros([1,len(amp_raw_data)])
            pha = eegfilt(pha_raw_data,srate,[],freq+(band/2))[0][0]
            pha = eegfilt(pha,srate,freq-(band/2),[])[0][0]
            pha = angle(hilbert(pha[0:len(pha)-1]))
            pha = pha[good_samps] #exclude bad samples

            #Calculate mean amplitude within each phase bin to yield a
            #distribution of amplitude(phase)
            bin_dist = zeros([len(bins)])
            for iBin in range(len(bins)):
                ind = logical_and(pha>=bins[iBin],pha<bins[iBin]+bin_size)
                bin_dist[iBin] = mean(pow[ind])

            #Normalize distribution to yield pseudo "probability density function" (PDF)
            bin_dist = bin_dist / sum(bin_dist)

            #Calculate Shannon entropy of PDF
            h_p = 0
            for iBin,mybin in enumerate(bin_dist):
                h_p = h_p - mybin * log(mybin)

            #MI = (Kullback-Leibler distance between h_p and uniform
            #distribution) / (Entropy of uniform distribution) (see
            #http://jn.physiology.org/content/104/2/1195)
            MI[iFreq,iBand] = (log(len(bins)) - h_p) / log(len(bins))

            if surrogate_analysis == 1:


                for iRun in range(kruns):

                    #if iRun%10 == 0:
                    #    print '{}\r'.format('Run ' + str(iRun+10)),

                    shift_factor = randint(0,len(pow))
                    pow_shifted = roll(pow,shift_factor)

                    #Calculate mean amplitude within each phase bin to yield a
                    #distribution of amplitude(phase)
                    bin_dist = zeros([len(bins)])
                    for iBin in range(len(bins)):
                        ind = logical_and(pha>=bins[iBin],pha<bins[iBin]+bin_size)
                        bin_dist[iBin] = mean(pow_shifted[ind])

                    #Normalize distribution to yield pseudo "probability density function" (PDF)
                    bin_dist = bin_dist / sum(bin_dist)

                    #Calculate Shannon entropy of PDF
                    h_p = 0
                    for iBin,mybin in enumerate(bin_dist):
                        h_p = h_p - mybin * log(mybin)

                    MI_surrogate[iFreq,iBand,iRun] = (log(len(bins)) - h_p) / log(len(bins))


    save(open(MI_output_path,'wb'),MI)
    if surrogate_analysis == 1:
        save(open(MI_output_path+'_surrogate','wb'),MI_surrogate)
