#@profile
def CFC_low_high_pre_post(subj,blocks,phase_elec,amp_elec,surrogate_analysis):
    import scipy.io as sio
    from numpy import arange,intersect1d,array,append,zeros,pi,angle,logical_and,mean,roll,save,where,empty,delete,in1d,random,rint #for efficiency
    from eegfilt import eegfilt
    from scipy.signal import hilbert
    from math import log
    import pickle
    import os
    from random import randint
    from collections import defaultdict

    #subglo_path = '/home/jcase/data/subj_globals.mat'
    #MI_output_path = '/home/jcase/data/' + subj + ''.join(blocks) + '/MI/e' + str(phase_elec) + '_e' + str(amp_elec)

    subglo_path = '/Users/johncase/Documents/UCSF Data/subj_globals.mat'
    MI_output_path = '/Users/johncase/Documents/UCSF Data/' + subj + '/' + subj + ''.join(blocks) + '/analysis/MI_ap/e' + str(phase_elec) + '_e' + str(amp_elec)

    amp_raw_data = {}
    pha_raw_data = {}

    per_chan_bad_epochs = {}
    stimID = {}

    #Trial windows
    bl = 0
    ps = 1 #in seconds

    #Surrogate Runs
    kruns = 200

    #Phase-providing frequency
    #fp = arange(1,15.1,0.1)
    #fp_bandwidth = arange(0.5,5.1,0.1)

    fp = [(i,i+2) for i in arange(3,50,2)]
#    fp_bandwidth = arange(1,11,1)

    #fp = arange(2,3)
    #fp_bandwidth = arange(2,3)

    #Amplitude-providing frequency
    fa = [(i,i+4) for i in arange(50,160,4)]

    #Define phase bins
    n_bins = 18
    bin_size = 2*pi/n_bins
    bins = arange(-pi,pi-bin_size,bin_size)

    MI_block = zeros((len(fa),len(fp),2))
    MI_diff = zeros((len(fa),len(fp)))
    if surrogate_analysis == 1:
        MI_diff_surrogate = zeros((len(fa),len(fp),kruns))


    #Determine samples with artifacts
    bad_samp = defaultdict(list)
    allstimtimes = []
    pow = {}

    for blk in blocks:
        #Load ECOG Data

        #data_path = '/home/jcase/data/' + subj + blk + '/' + subj + '_' + blk + '_data.mat'

        data_path = '/Users/johncase/Documents/UCSF Data/' + subj + '/' + subj + blk + '/data/' + subj + '_' + blk + '_data.mat'

        ecog_data = sio.loadmat(data_path)['ecogData']
        amp_raw_data[blk] = ecog_data[amp_elec-1,:]
        pha_raw_data[blk] = ecog_data[phase_elec-1,:]
        del ecog_data

        #Load subject globals
        all_subj_data = sio.loadmat(subglo_path,struct_as_record=False, squeeze_me=True)['subj_globals']
        subj_data = getattr(getattr(all_subj_data,subj),blk)
        srate = subj_data.srate
        per_chan_bad_epochs[blk] = subj_data.per_chan_bad_epochs
        all_onsets = subj_data.allstimtimes[:,0]
        stimID[blk] = subj_data.stimID
        all_onsets = delete(all_onsets,where(stimID[blk]==10),0) #delete clicks
        stimID[blk] = delete(stimID[blk],where(stimID[blk]==10)) #delete clicks

        #Identify bad samples for each block
        if per_chan_bad_epochs[blk][phase_elec-1].size == 2:
            bad_samp[blk] = append(bad_samp[blk],arange(srate*per_chan_bad_epochs[blk][phase_elec-1][0],srate*per_chan_bad_epochs[blk][phase_elec-1][1]))
        else:
            for epoch in per_chan_bad_epochs[blk][phase_elec-1]:
                bad_samp[blk] = append(bad_samp[blk],arange(srate*epoch[0],srate*epoch[1]))

        if not phase_elec == amp_elec:
            if per_chan_bad_epochs[blk][amp_elec-1].size == 2:
                bad_samp[blk] = append(bad_samp[blk],arange(srate*per_chan_bad_epochs[blk][amp_elec-1][0],srate*per_chan_bad_epochs[blk][amp_elec-1][1]))
            else:
                for epoch in per_chan_bad_epochs[blk][amp_elec-1]:
                    bad_samp[blk] = append(bad_samp[blk],arange(srate*epoch[0],srate*epoch[1]))

    for iFreq_a,freq_a in enumerate(fa):

        allstimtimes = []

        for iBlock,blk in enumerate(blocks):
            #Do high-gamma filtering
            pow[blk],filtwt = eegfilt(amp_raw_data[blk],srate,freq_a[0],[])
            pow[blk],filtwt = eegfilt(pow[blk][0],srate,[],freq_a[1])
            pow[blk] = abs(hilbert(pow[blk][0][0:len(pow[blk][0])-1]))

            #make each onset a tuple containing the block name, phase, and amp data

            for onset in all_onsets:

                trl = arange(rint(onset*srate),rint((onset+ps)*srate)).astype(int)
                #if onset does not overlap with an artifact
                if not any(intersect1d(trl,bad_samp[blk])):
                    allstimtimes.extend([(blk,trl,pow[blk][trl])])

        for iFreq_p,freq_p in enumerate(fp):


            print('amp = ' + str(int(mean([freq_a[0],freq_a[1]]))) + ', pha  = ' + str(int(mean([freq_p[0],freq_p[1]]))))

            pha = {}

            stim_n = zeros(2)
            for iBlock,blk in enumerate(blocks):

                #Do phase-providing phase filtering
                pha[blk] = zeros([1,len(pha_raw_data)])
                pha[blk] = eegfilt(pha_raw_data[blk],srate,[],freq_p[1])[0][0]
                pha[blk] = eegfilt(pha[blk],srate,freq_p[0],[])[0][0]
                pha[blk] = angle(hilbert(pha[blk][0:len(pha[blk])-1]))

                #add phase information
                for iTrial,trial in enumerate(allstimtimes):
                    if trial[0] == blk:
                        allstimtimes[iTrial] = allstimtimes[iTrial][:3] + (pha[blk][trial[1]],)

                #count number of trials in each block (for surrogate analysis)
                stim_n[iBlock] = int(len([trial for trial in allstimtimes if trial[0] == blk]))

                #retrieve power from allstimtimes
                pow_istim = [trial[2] for trial in allstimtimes if trial[0] == blk]
                pow_istim = array([item for sublist in pow_istim for item in sublist])

                #retrieve phase from allstimtimes
                pha_istim = [trial[3] for trial in allstimtimes if trial[0] == blk]
                pha_istim = array([item for sublist in pha_istim for item in sublist])


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
                MI_block[iFreq_a,iFreq_p,iBlock] = (log(len(bins)) - h_p) / log(len(bins))

            # difference statistic = Block 2 - Block 1
            MI_diff[iFreq_a,iFreq_p] = MI_block[iFreq_a,iFreq_p,1]-MI_block[iFreq_a,iFreq_p,0]

            if surrogate_analysis == 1:

                phase_trl_onsets_all = [trial[3] for trial in allstimtimes]
                amp_trl_onsets_all = [trial[2] for trial in allstimtimes]

                for iRun in range(kruns):

                    if iRun%10 == 0:
                        print '{}\r'.format('Run ' + str(iRun+10)),

                    MI_block_surrogate = zeros(2)

                    random.shuffle(amp_trl_onsets_all)

                    cnt = 0
                    for iBlock,blk in enumerate(blocks):

                        phase_trl_onsets = phase_trl_onsets_all[cnt:cnt+int(stim_n[iBlock])]
                        amp_trl_onsets = amp_trl_onsets_all[cnt:cnt+int(stim_n[iBlock])]
                        cnt = cnt + int(stim_n[iBlock])

                        bin_dist = zeros([len(bins)])
                        for iBin in range(len(bins)):

                            bin_power_list = array([])

                            for iTrial,trl in enumerate(phase_trl_onsets):

                                #find samples within phase bin (indices relative to stimulus onset (i.e. 0-400))
                                ind = where(logical_and(trl>=bins[iBin],trl<bins[iBin]+bin_size))[0]

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
                    MI_diff_surrogate[iFreq_a,iFreq_p,iRun] = MI_block_surrogate[1]-MI_block_surrogate[0]

    save(open(MI_output_path+'_block_diff','wb'),MI_diff)
    save(open(MI_output_path+'_block','wb'),MI_block)
    if surrogate_analysis == 1:
        save(open(MI_output_path+'_block_diff_surrogate','wb'),MI_diff_surrogate)

CFC_low_high_pre_post('EC71',['B2','B11'],1,90,1)