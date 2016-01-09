@profile
def CFC_mean_vec_pre_post(subj,blocks,phase_elec,amp_elec,surrogate_analysis):
    import scipy.io as sio
    from numpy import arange,intersect1d,array,append,zeros,pi,angle,logical_and,mean,roll,save,where,empty,delete,in1d,random,rint,exp #for efficiency
    from eegfilt import eegfilt
    from scipy.signal import hilbert
    from math import log
    import pickle
    import os
    from random import randint
    from collections import defaultdict
    from itertools import chain

    #subglo_path = '/home/jcase/data/subj_globals.mat'
    #MI_output_path = '/home/jcase/data/' + subj + ''.join(blocks) + '/MI/e' + str(phase_elec) + '_e' + str(amp_elec)

    subglo_path = '/Users/johncase/Documents/UCSF Data/subj_globals.mat'
    MI_output_path = '/Users/johncase/Documents/UCSF Data/' + subj + '/' + subj + ''.join(blocks) + '/analysis/mean_vec/e' + str(phase_elec) + '_e' + str(amp_elec)

#    if not os.path.exists(MI_output_path):
#        os.makedirs(MI_output_path)

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
    #t_0 = int(round(allstimtimes[0,0]*srate))
    #t_end = int(round((allstimtimes[-1,1]+3) *srate))
    #t_win = arange(t_0,t_end)

    MI_block = zeros((len(fp),len(fp_bandwidth),2),complex)
    MI_diff = zeros((len(fp),len(fp_bandwidth)))
    if surrogate_analysis == 1:
        MI_diff_surrogate = zeros((len(fp),len(fp_bandwidth),kruns))


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

        #Do high-gamma filtering
        pow[blk],filtwt = eegfilt(amp_raw_data[blk],srate,fa[0],[])
        pow[blk],filtwt = eegfilt(pow[blk][0],srate,[],fa[1])
        pow[blk] = abs(hilbert(pow[blk][0][0:len(pow[blk][0])-1]))


        #make each onset a tuple containing the block name, phase, and amp data
        for onset in all_onsets:

            trl = arange(rint(onset*srate),rint((onset+ps)*srate)).astype(int)
            #if onset does not overlap with an artifact
            if not any(intersect1d(trl,bad_samp[blk])):
                allstimtimes.extend([(blk,trl,pow[blk][trl])])

    #Calculate MI for each phase-providing central-frequencies / bandwidths

    for iFreq,freq in enumerate(fp):
        for iBand,band in enumerate(fp_bandwidth):

            if freq-(band/2) < 0.5:
                MI_diff[iFreq,iBand] = 0
                continue

            print('freq = ' + str(freq) + ', bw = ' + str(band))

            pha = {}

            stim_n = zeros(2)
            for iBlock,blk in enumerate(blocks):

                #Do phase-providing phase filtering
                pha[blk] = zeros([1,len(amp_raw_data)])
                pha[blk] = eegfilt(pha_raw_data[blk],srate,[],freq+(band/2))[0][0]
                pha[blk] = eegfilt(pha[blk],srate,freq-(band/2),[])[0][0]
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

                #Calculate mean vector of composite signal
                MI_block[iFreq,iBand,iBlock] = mean(pow_istim * exp(1j * pha_istim))

            # difference statistic = Block 2 - Block 1
            MI_diff[iFreq,iBand] = abs(MI_block[iFreq,iBand,1])-abs(MI_block[iFreq,iBand,0])

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

                        phase_trls = phase_trl_onsets_all[cnt:cnt+int(stim_n[iBlock])]
                        phase_trls = array([item for sublist in phase_trls for item in sublist])

                        amp_trls = amp_trl_onsets_all[cnt:cnt+int(stim_n[iBlock])]
                        amp_trls = array([item for sublist in amp_trls for item in sublist])

                        cnt = cnt + int(stim_n[iBlock])

                        MI_block_surrogate[iBlock] = abs(mean(amp_trls * exp(1j * phase_trls)))
                    MI_diff_surrogate[iFreq,iBand,iRun] = MI_block_surrogate[1]-MI_block_surrogate[0]



    save(open(MI_output_path+'_block_diff','wb'),MI_diff)
    save(open(MI_output_path+'_block','wb'),MI_block)
    if surrogate_analysis == 1:
        save(open(MI_output_path+'_block_diff_surrogate','wb'),MI_diff_surrogate)

CFC_mean_vec_pre_post('EC71',['B2','B11'],1,90,1)