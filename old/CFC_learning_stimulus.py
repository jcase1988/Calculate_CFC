#@profile
def CFC_learning_stimulus(subj,block,phase_elec,amp_elec,method,stim_method,surrogate_analysis):
    import scipy.io as sio
    from numpy import concatenate,exp,arange,intersect1d,array,append,zeros,pi,angle,logical_and,mean,roll,save,where,empty,delete,in1d,random,rint #for efficiency
    from eegfilt import eegfilt
    from scipy.signal import hilbert
    from math import log
    import pickle
    import os
    from random import randint
    from collections import defaultdict

    #stim_method: 0 for all stimuli (60 vs 60 trials)
    #             1 for 1st offered (30 vs 30 trials)
    #             2 for 2nd offered (30 vs 30 trials)
    #             3 for feedback    (30 vs 30 trials)

    s_methods = {0:'all',1:'first',2:'second',3:'feedback'}
    
   # subglo_path = '/home/jcase/data/subj_globals.mat'
   # data_path = '/home/jcase/data/' + subj + block + '/' + subj + '_' + block + '_data.mat'
   # MI_output_path = '/home/jcase/data/' + subj + block + '/' + method + '/' + s_methods[stim_method] + '_e' + str(phase_elec) + '_e' + str(amp_elec)

    subglo_path = '/Users/johncase/Documents/UCSF Data/subj_globals.mat'
    data_path = '/Users/johncase/Documents/UCSF Data/' + subj + '/' + subj + block + '/data/' + subj + '_' + block + '_data.mat'
    MI_output_path = '/Users/johncase/Documents/UCSF Data/' + subj + '/' + subj + block + '/analysis/' + method + '/' + s_methods[stim_method] + '_e' + str(phase_elec) + '_e' + str(amp_elec)

    amp_raw_data = {}
    pha_raw_data = {}
    per_chan_bad_epochs = {}
    stimID = {}

    #Trial windows
    bl = 0
    ps = 1 #in seconds

    #Surrogate Runs
    kruns = 200

    #Amplitude-providing frequency
    fa = array([70,150])

    #Define phase bins
    n_bins = 18
    bin_size = 2*pi/n_bins
    bins = arange(-pi,pi-bin_size,bin_size)

    MI_block = zeros((len(fp),len(fp_bandwidth),2))
    MI_diff = zeros((len(fp),len(fp_bandwidth)))
    if surrogate_analysis == 1:
        MI_diff_surrogate = zeros((len(fp),len(fp_bandwidth),kruns))


    #Determine samples with artifacts
    bad_samp = defaultdict(list)
    allstimtimes = []
    pow = {}

    ecog_data = sio.loadmat(data_path)['ecogData']
    amp_raw_data = ecog_data[amp_elec-1,:]
    pha_raw_data = ecog_data[phase_elec-1,:]
    del ecog_data

    #Load subject globals
    all_subj_data = sio.loadmat(subglo_path,struct_as_record=False, squeeze_me=True)['subj_globals']
    subj_data = getattr(getattr(all_subj_data,subj),block)
    srate = subj_data.srate

    per_chan_bad_epochs = subj_data.per_chan_bad_epochs

    offered = subj_data.offered
    samp1_onsets = subj_data.sample1times[:,0]
    samp2_onsets = subj_data.sample2times[:,0]
    feedback_onsets = subj_data.feedbacktimes[:,0]

    #Identify bad samples
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
    pow,filtwt = eegfilt(amp_raw_data,srate,[],fa[1])
    pow,filtwt = eegfilt(pow[0],srate,fa[0],[])
    pow = abs(hilbert(pow[0][0:len(pow[0])-1]))

    if stim_method == 0: #all stimuli
        all_onsets = append(samp1_onsets,samp2_onsets)
    elif stim_method == 1:
        all_onsets = samp1_onsets
    elif stim_method == 2:
        all_onsets = samp2_onsets
    elif stim_method == 3:
        all_onsets = feedback_onsets

    #make each onset a tuple containing the block name, phase, and amp data
    for onset in all_onsets:

        halfway = samp1_onsets[30]
        if onset < halfway:
            blk = 0
        else:
            blk = 1

        trl = arange(rint(onset*srate),rint((onset+ps)*srate)).astype(int)
        #if onset does not overlap with an artifact
        if not any(intersect1d(trl,bad_samp)):
            allstimtimes.extend([(blk,trl,pow[trl])])


    #Phase-providing frequency
    fp = arange(1,21,1)
    fp_bandwidth = arange(1,11,1)

    fp_good = arange(4,18)
    fp_bandwidth_good = arange(4,10)

    #Calculate MI for each phase-providing central-frequencies / bandwidths
    for iFreq,freq in enumerate(fp):
        for iBand,band in enumerate(fp_bandwidth):

            if freq-(band/2) < 0.5:
                MI_diff[iFreq,iBand] = 0
                continue

            if freq not in fp_good or band not in fp_bandwidth_good:
                MI_diff[iFreq,iBand] = 0
                continue

            print('freq = ' + str(freq) + ', bw = ' + str(band))

            pha = {}

            stim_n = zeros(2)

            #Do phase-providing phase filtering
            pha = zeros([1,len(amp_raw_data)])
            pha = eegfilt(pha_raw_data,srate,[],freq+(band/2))[0][0]
            pha = eegfilt(pha,srate,freq-(band/2),[])[0][0]
            pha = angle(hilbert(pha[0:len(pha)-1]))

            for blk in range(0,2):
                #add phase information
                for iTrial,trial in enumerate(allstimtimes):
                    if trial[0] == blk:
                        allstimtimes[iTrial] = allstimtimes[iTrial][:3] + (pha[trial[1]],)

                #count number of trials in each block (for surrogate analysis)
                stim_n[blk] = int(len([trial for trial in allstimtimes if trial[0] == blk]))

            for blk in range(0,2):
                #retrieve power from allstimtimes
                pow_istim = [trial[2] for trial in allstimtimes if trial[0] == blk]
                pow_istim = array([item for sublist in pow_istim for item in sublist])

                #retrieve phase from allstimtimes
                pha_istim = [trial[3] for trial in allstimtimes if trial[0] == blk]
                pha_istim = array([item for sublist in pha_istim for item in sublist])

                if method == 'MI':

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
                    MI_block[iFreq,iBand,iBlock] = (log(len(bins)) - h_p) / log(len(bins))

                elif method == 'dPAC':

                    MI_block[iFreq,iBand,blk] = abs(mean(pow_istim*(exp(1j*pha_istim) - mean(exp(1j*pha_istim)))))


            # difference statistic = Block 2 - Block 1
            MI_diff[iFreq,iBand] = MI_block[iFreq,iBand,1]-MI_block[iFreq,iBand,0]

            if surrogate_analysis == 1:

                phase_trl_onsets_all = [trial[3] for trial in allstimtimes]
                amp_trl_onsets_all = [trial[2] for trial in allstimtimes]

                #phases do no shuffle between runs, so predetermine them
                pha_istims = [None]*2
                cnt = 0
                for iBlock in range(2):
                    phase_trl_onsets = phase_trl_onsets_all[cnt:cnt+int(stim_n[iBlock])]
                    cnt = cnt + int(stim_n[iBlock])
                    pha_istims[iBlock] = array([item for sublist in phase_trl_onsets for item in sublist])



                for iRun in range(kruns):

                    if iRun%10 == 0:
                        print '{}\r'.format('Run ' + str(iRun+10)),

                    MI_block_surrogate = zeros(2)

                    random.shuffle(amp_trl_onsets_all)

                    cnt = 0
                    for iBlock in range(2):

                        amp_trl_onsets = amp_trl_onsets_all[cnt:cnt+int(stim_n[iBlock])]
                        cnt = cnt + int(stim_n[iBlock])

                        if method == 'MI':
                            bin_dist = zeros([len(bins)])
                            for iBin in range(len(bins)):

                                bin_power_list = array([])

                                for iTrial,trl in enumerate(phase_trl_onsets):

                                    #find samples within phase bin (indices relative to stimulus onset (i.e. 0-400))
                                    ind = where(logical_and(trl>=bins[iBin],trl<bins[iBin]+bin_size))[0]

                                    #grow list of power during phase bin (power from random trial but with same post-stim indices)
                                    if any(ind):
                                        bin_power_list = append(bin_power_list,amp_trl_onsets[iTrial][ind])

                                bin_dist[iBin] = mean(bin_power_list)

                            #Normalize distribution to yield pseudo "probability density function" (PDF)
                            bin_dist = bin_dist / sum(bin_dist)

                            #Calculate Shannon entropy of PDF
                            h_p = 0
                            for iBin,mybin in enumerate(bin_dist):
                                h_p = h_p - mybin * log(mybin)

                            MI_block_surrogate[iBlock] = (log(len(bins)) - h_p) / log(len(bins))

                        elif method == 'dPAC':
                            pow_istim = array([item for sublist in amp_trl_onsets for item in sublist])
                            MI_block_surrogate[iBlock] = abs(mean(pow_istim*(exp(1j*pha_istims[iBlock]) - mean(exp(1j*pha_istims[iBlock])))))



                    MI_diff_surrogate[iFreq,iBand,iRun] = MI_block_surrogate[1]-MI_block_surrogate[0]




    save(open(MI_output_path+'_block_diff','wb'),MI_diff)
    save(open(MI_output_path+'_block','wb'),MI_block)
    if surrogate_analysis == 1:
        save(open(MI_output_path+'_block_diff_surrogate','wb'),MI_diff_surrogate)

#CFC_learning_stimulus('EC82','B48',307,102,'dPAC',1,1)

