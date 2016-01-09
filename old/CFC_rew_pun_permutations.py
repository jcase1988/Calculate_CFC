def CFC_rew_pun_permutation(subj,block,phase_elec,amp_elec,surrogate_analysis):
    import scipy.io as sio
    from numpy import arange,array,append,zeros,pi,angle,logical_and,mean,roll,save,where,empty,delete,in1d,random,rint #for efficiency
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
    MI_output_path = '/Users/johncase/Documents/UCSF Data/' + subj + '/' + subj + block + '/analysis/MI/rew_pun/e' + str(phase_elec) + '_e' + str(amp_elec)

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
    stimID = subj_data.stimID

    #Trial windows
    bl = 0
    ps = 1 #in seconds

    #Surrogate Runs
    kruns = 200

    #Phase-providing frequency
    #fp = arange(1,15.1,0.1)
    #fp_bandwidth = arange(0.5,5.1,0.1)

    #fp = arange(1,21,1)
    #fp_bandwidth = arange(1,11,1)

    fp = arange(2,3)
    fp_bandwidth = arange(2,3)

    #Amplitude-providing frequency
    fa = array([70,150])

    #Define phase bins
    n_bins = 20
    bin_size = 2*pi/n_bins
    bins = arange(-pi,pi-bin_size,bin_size)

    #Define time_window (roughly entire block, exclude artifacts samples later)
    #t_0 = int(round(allstimtimes[0,0]*srate))
    #t_end = int(round((allstimtimes[-1,1]+3) *srate))
    #t_win = arange(t_0,t_end)

    MI_stim = empty((len(fp),len(fp_bandwidth),3))
    MI_diff = empty((len(fp),len(fp_bandwidth)))
    if surrogate_analysis == 1:
        MI_diff_surrogate = empty((len(fp),len(fp_bandwidth),kruns))



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



    #Do high-gamma filtering
    pow,filtwt = eegfilt(amp_raw_data,srate,fa[0],[])
    pow,filtwt = eegfilt(pow[0],srate,[],fa[1])
    pow = abs(hilbert(pow[0][0:len(pow[0])-1]))
   # pow = pow[good_samps] #exclude bad` samples

    #Calculate MI for each phase-providing central-frequencies / bandwidths

    for iFreq,freq in enumerate(fp):
        for iBand,band in enumerate(fp_bandwidth):

            if freq-(band/2) < 0.5:
                MI_diff[iFreq,iBand] = 0
                continue

            print('freq = ' + str(freq) + ', bw = ' + str(band))

            #Do phase-providing phase filtering

            pha = zeros([1,len(amp_raw_data)])
            pha = eegfilt(pha_raw_data,srate,[],freq+(band/2))[0][0]
            pha = eegfilt(pha,srate,freq-(band/2),[])[0][0]
            pha = angle(hilbert(pha[0:len(pha)-1]))



            for iStim,iStimID in enumerate(range(1,4)):

                trl_onsets = rint(allstimtimes[where(stimID==iStimID)[0],0]*srate).astype(int)

                trl_samps = array([]).astype(int)
                for trl in trl_onsets:
                    trl_samps = append(trl_samps,arange(int(trl+bl*srate),int(trl+ps*srate)))

                #exclude bad samples
                trl_samps = trl_samps[~in1d(trl_samps,bad_samp)]

                #keep phase/pow info only within iStim trl windows
                pha_istim = pha[trl_samps]
                pow_istim = pow[trl_samps]

                #Calculate mean amplitude within each phase bin to yield a
                #distribution of amplitude(phase)
                bin_dist = zeros([len(bins)])
                for iBin in range(len(bins)):
                    ind = logical_and(pha_istim>=bins[iBin],pha_istim<bins[iBin]+bin_size)
                    bin_dist[iBin] = mean(pow_istim[ind])

                #Normalize distribution to yield pseudo "probability density function" (PDF)
                bin_dist = bin_dist / sum(bin_dist)

                #Calculate Shannon entropy of PDF
                h_p = 0
                for iBin,mybin in enumerate(bin_dist):
                    h_p = h_p - mybin * log(mybin)

                #MI = (Kullback-Leibler distance between h_p and uniform
                #distribution) / (Entropy of uniform distribution) (see
                #http://jn.physiology.org/content/104/2/1195)
                MI_stim[iFreq,iBand,iStim] = (log(len(bins)) - h_p) / log(len(bins))

            # difference statistic = abs(A-B) + abs(A-C) + abs(B-C)
            MI_diff[iFreq,iBand] = abs(MI_stim[iFreq,iBand,0]-MI_stim[iFreq,iBand,1]) + abs(MI_stim[iFreq,iBand,0]-MI_stim[iFreq,iBand,2]) + abs(MI_stim[iFreq,iBand,1]-MI_stim[iFreq,iBand,2])

            if surrogate_analysis == 1:

                stim_n = zeros(3)
                stim_n[0] = len(where(stimID==1)[0])
                stim_n[1] = len(where(stimID==2)[0])
                stim_n[2] = len(where(stimID==3)[0])
                shuffle_ind = arange(sum(stim_n)).astype(int)


                for iRun in range(kruns):

                    if iRun%10 == 0:
                        print '{}\r'.format('Run ' + str(iRun+10)),

                    MI_stim_surrogate = zeros(3)

                    random.shuffle(shuffle_ind)

                    cnt = 0
                    for iStim,iStimID in enumerate(range(1,4)):

                        phase_trl_onsets = rint(allstimtimes[where(stimID==iStimID)[0],0]*srate).astype(int)
                        amp_trl_onsets =  rint(allstimtimes[shuffle_ind[cnt:cnt+len(phase_trl_onsets)],0]*srate).astype(int)
                        cnt = cnt + len(phase_trl_onsets)


                        bin_dist = zeros([len(bins)])
                        for iBin in range(len(bins)):

                            bin_power_list = array([])

                            for iTrial,trl in enumerate(phase_trl_onsets):

                                #find sample of onset to 1s + onset
                                phase_trl = arange(trl,trl+ps*srate).astype(int)

                                #find samples within phase bin (indices relative to stimulus onset (i.e. 0-400))
                                ind = where(logical_and.reduce((pha[phase_trl]>=bins[iBin],pha[phase_trl]<bins[iBin]+bin_size,~in1d(phase_trl,bad_samp))))[0]

                                #find amp samples with the same post-stimulus latency as "inds"
                                amp_samples = amp_trl_onsets[iTrial]+ind
                                amp_samples = amp_samples[~in1d(amp_trl_onsets[iTrial]+ind,bad_samp)]

                                #grow list of power during phase bin (power from random trial but with same post-stim indices)
                                bin_power_list = append(bin_power_list,pow[amp_samples])

                            bin_dist[iBin] = mean(bin_power_list)

                        #Normalize distribution to yield pseudo "probability density function" (PDF)
                        bin_dist = bin_dist / sum(bin_dist)

                        #Calculate Shannon entropy of PDF
                        h_p = 0
                        for iBin,mybin in enumerate(bin_dist):
                            h_p = h_p - mybin * log(mybin)

                        MI_stim_surrogate[iStim] = (log(len(bins)) - h_p) / log(len(bins))
                    MI_diff_surrogate[iFreq,iBand,iRun] = abs(MI_stim_surrogate[0]-MI_stim_surrogate[1]) + abs(MI_stim_surrogate[0]-MI_stim_surrogate[2]) + abs(MI_stim_surrogate[1]-MI_stim_surrogate[2])



    save(open(MI_output_path+'_diff','wb'),MI_diff)
    save(open(MI_output_path+'_stim','wb'),MI_stim)
    if surrogate_analysis == 1:
        save(open(MI_output_path+'_diff_surrogate','wb'),MI_diff_surrogate)
