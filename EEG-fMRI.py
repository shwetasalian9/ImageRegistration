# -*- coding: utf-8 -*-
"""
Source Code for Implementation of Assignment 3

@author: Shweta Salian
"""

import mne as mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.signal import find_peaks
from mne.preprocessing import ICA
import nibabel as nib
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
import scipy.stats as stats

"""
Step 2a : Load EEG dataset into python using mne
""" 

 
# Load eeg header file - Alex scan retino_gamma_01
# eeg = mne.io.read_raw_brainvision('eegfmri/alex/alex_retino_gamma_01.vhdr', preload=True)

# Load eeg header file  - Alex scan retino_gamma_02
# eeg = mne.io.read_raw_brainvision('eegfmri/alex/retino_gamma_02.vhdr', preload=True)

# Load eeg header file - Alex scan retino_rest
# eeg = mne.io.read_raw_brainvision('eegfmri/alex/alex_rest.vhdr', preload=True)

# Load eeg header file - Genevieve scan retino_gamma_01
# eeg = mne.io.read_raw_brainvision('eegfmri/genevieve/retino_gamma_01.vhdr', preload=True)

# Load eeg header file - Genevieve scan retino_gamma_02
# eeg = mne.io.read_raw_brainvision('eegfmri/genevieve/retino_gamma_02.vhdr', preload=True)

# Load eeg header file - Genevieve scan retino_rest
# eeg = mne.io.read_raw_brainvision('eegfmri/genevieve/genevieve_rest.vhdr', preload=True)

# Load eeg header file - Russell scan 2-gamma
# eeg = mne.io.read_raw_brainvision(r'eegfmri/russell/2-gamma.vhdr', preload=True)

# Load eeg header file - Russell scan 4-gamma
# eeg = mne.io.read_raw_brainvision(r'eegfmri/russell/4 gamma.vhdr', preload=True)

eeg = mne.io.read_raw_brainvision(r'eegfmri/russell/6 rest.vhdr', preload=True)

# Read data from eeg file
eeg_data = eeg.get_data()

# Read mne events
events = mne.events_from_annotations(eeg)

# Plotting the time series eeg data
plt.plot(eeg_data[3, :])


"""
Step 2b : Removal of gradient artifact and ballistocardiogram artifact
"""     

# Removal of gradient artifact
volume_trigs = events[1]['Response/R128']
volume_inds = np.argwhere(events[0][:,2]==volume_trigs)
volume_times = events[0][volume_inds,0].astype(int)
    
# sanity check
trig_ts = np.zeros([eeg_data.shape[1]])
trig_ts[volume_times] = 1
plt.plot(stats.zscore(trig_ts)); plt.plot(stats.zscore(eeg_data[3,:]))


gradlen = 3465

gradient_epochs = np.reshape(eeg_data[:,volume_times[0,0]:volume_times[-2,0]+gradlen],[64,volume_times.shape[0]-1,gradlen])

sub_grad_artifact = gradient_epochs - np.expand_dims(np.mean(gradient_epochs,axis=1), axis=1)
sub_grad_artifact = np.reshape(sub_grad_artifact, eeg_data[:,volume_times[0,0]:volume_times[-2,0]+gradlen].shape)

plt.plot(sub_grad_artifact[3,:])
    

# Removal of ballistocardiogram artifact

# downsample
resamp = resample(sub_grad_artifact,int(sub_grad_artifact.shape[1]/(5000/250)),axis=1)
plt.plot(resamp[3,:])

    
pks = find_peaks(resamp[31,:],distance=180)
pkvis = np.zeros([resamp.shape[1]])
pkvis[pks[0]] = 1
plt.plot(stats.zscore(pkvis)/3); plt.plot(stats.zscore(resamp[31,:]));


# epoch the data using peaks
eps = np.zeros([64,pks[0].shape[0],210])
for i in np.arange(1,pks[0].shape[0]-1):
    eps[:,i,:] = resamp[:,pks[0][i]-105:pks[0][i]+105]

meanbcg = np.mean(eps,axis=1)
    
sub_bal_artifact = np.copy(resamp)
    
for i in np.arange(1,pks[0].shape[0]-1):
    sub_bal_artifact[:,pks[0][i]-105:pks[0][i]+105] = resamp[:,pks[0][i]-105:pks[0][i]+105] - meanbcg

plt.plot(resamp[3,:]); plt.plot(sub_bal_artifact[3,:])  


"""
Step 2c : Run ICA and plot ICA Components
"""  

eeg.resample(250)
raw = eeg.get_data()
events = mne.events_from_annotations(eeg)
volume_trigs = events[1]['Response/R128']
volume_inds = np.argwhere(events[0][:,2] == volume_trigs)
volume_times = events[0][volume_inds,0].astype(int)
start = volume_times[0][0]
    

raw[:,start:start+sub_bal_artifact.shape[1]] = sub_bal_artifact
raw[:,0:start] = 0
raw[:,start+sub_bal_artifact.shape[1]:] = 0
# overwrite eeg data with denoised data
eeg._data = raw
    
montage = mne.channels.make_standard_montage('standard_1020')
eeg.set_montage(montage, on_missing='ignore')

# high pass filter
eeg.filter(1,124)
ica = ICA(n_components=60)
good_channels = np.arange(0, 63); good_channels = np.delete(good_channels, 31)
ica.fit(eeg, picks=good_channels)

# visualize components
ica.plot_components()

# visualize sources 
ica.plot_sources(eeg)

# get the sources
sources = ica.get_sources(eeg)._data

psd = welch(sources, fs=250, nperseg=500)

for i in np.arange(0,60):
    plt.subplot(6,10,i+1)
    plt.plot(psd[0][0:120], np.log(psd[1][i,0:120]), 'r')
    plt.axvline(8); plt.axvline(13)
    plt.title(i)
    
plt.subplots_adjust(hspace=1)
plt.subplots_adjust(wspace=0.5)


"""
Step 2d : Identify good ICA Components
"""  
# Good components for alex - retino_gamma_01
# goodcomps = sources[[8,14,15,22],:]

# Good components for alex - retino_gamma_02
# goodcomps = sources[[17,19,20,25],:]

# Good components for alex - retino_rest
# goodcomps = sources[[8,13,16],:]

# Good components for genevieve - retino_gamma_01
# goodcomps = sources[[14,15,21,24,25],:]

# Good components for genevieve - retino_gamma_02
# goodcomps = sources[[12,18,21,30],:]

# Good components for genevieve - retino_rest
# goodcomps = sources[[5,9,14,17,27,28,29],:]

# Good components for russell - 2-gamma
# goodcomps = sources[[12,17,18,16,19],:]

# Good components for russell - 4 gamma
# goodcomps = sources[[3,19,18,33],:]

# Good components for russell - 4 gamma
goodcomps = sources[[8,12,31,9,14,17],:]

# correlate with bold

# Bold image - Alex scan retino_gamma_01 
# bold = nib.load('eegfmri/alex/scan1/bp_retino_gamma_01.nii.gz').get_fdata()

# Bold image - Alex scan retino_gamma_02
# bold = nib.load('eegfmri/alex/scan2/bp_retino_gamma_02.nii.gz').get_fdata()

# Bold image - Alex scan retino_rest
# bold = nib.load('eegfmri/alex/scan3/bp_retino_rest.nii.gz').get_fdata()

# Bold image - Genevieve scan retino_gamma_01
# bold = nib.load('eegfmri/genevieve/scan1/bp_retino_gamma_01.nii.gz').get_fdata()

# Bold image - Genevieve scan retino_gamma_02
# bold = nib.load('eegfmri/genevieve/scan2/bp_retino_gamma_02.nii.gz').get_fdata()

# Bold image - Genevieve scan retino_rest
# bold = nib.load('eegfmri/genevieve/scan3/bp_retino_rest.nii.gz').get_fdata()

# Bold image - Russell scan 2-gamma
# bold = nib.load('eegfmri/russell/scan1/bp_retino_gamma_01.nii.gz').get_fdata()

# Bold image - Russell scan 4 gamma
# bold = nib.load('eegfmri/russell/scan2/bp_retino_gamma_02.nii.gz').get_fdata()


bold = nib.load('eegfmri/russell/scan3/bp_retino_rest.nii.gz').get_fdata()


"""
Step 2e : Bandpass filter the denoised EEG data to alpha band (8-13 Hz)
"""  

# Alpha Comps

alpha_comps = mne.filter.filter_data(goodcomps, 250,8,13)
alpha_comps = np.abs(alpha_comps)

smooth_alpha_comps = gaussian_filter1d(alpha_comps, 250, axis =1)
trimalpha = np.mean(smooth_alpha_comps[:, start:], axis=0)

plt.plot(trimalpha)    


"""
Step 4a :  Downsample to match the fMRI temporal resolution
""" 

# downsample alpha
n_timepoints = volume_times.shape[0]
resampled_alpha = resample(trimalpha,n_timepoints)  


"""
Step 4b :  Convolve the resampled, bandpass-filtered EEG power with the canonical hemodynamic response function
""" 


shift = int(5/0.693)
shift_alpha = np.roll(resampled_alpha,shift)

shift_alpha = mne.filter.filter_data(shift_alpha, 1/0.693, 0.01, 0.7)

alpha4d = np.zeros(bold.shape)

alpha4d = alpha4d[:,:,:,0:shift_alpha.shape[0]] + shift_alpha

# selected range differently for different scans based on n_timepoints value
timepoints = np.arange(30,450)



"""
Step 4c :  Correlate the convolved EEG alpha (8-13 Hz) power with fMRI in each voxel
"""

#correlation for alpha
corrmap = np.sum((bold[:,:,:,timepoints] - np.mean(bold[:,:,:,timepoints],axis=3,keepdims=True)) *
                  (alpha4d[:,:,:,timepoints] - np.mean(alpha4d[:,:,:,timepoints],axis=3,keepdims=True)), axis=3)\
    / (np.sqrt(np.sum((bold[:,:,:,timepoints] - np.mean(bold[:,:,:,timepoints], axis=3, keepdims= True))**2, axis=3))
        * np.sqrt(np.sum((alpha4d[:,:,:,timepoints] - np.mean(alpha4d[:,:,:,timepoints],axis=3,keepdims=True))**2,axis=3)))


for i in np.arange(0,33):
    plt.subplot(3,11,i+1)
    plt.imshow(np.rot90(corrmap[:,:,i]),vmin=-0.3, vmax=0.3)
    plt.title(i)
    
    
    
# meanbold = nib.load('eegfmri/alex/retino_gamma_01.nii.gz')
# corrmapnii = nib.Nifti1Image(corrmap, meanbold.affine, meanbold.header)
# nib.save(corrmapnii, 'eegfmri/alex/scan1/alex_alpha_corrmap.nii.gz')

# meanbold = nib.load('eegfmri/alex/retino_gamma_02.nii.gz')
# corrmapnii = nib.Nifti1Image(corrmap, meanbold.affine, meanbold.header)
# nib.save(corrmapnii, 'eegfmri/alex/scan2/alex_alpha_corrmap.nii.gz')

# meanbold = nib.load('eegfmri/alex/retino_rest.nii.gz')
# corrmapnii = nib.Nifti1Image(corrmap, meanbold.affine, meanbold.header)
# nib.save(corrmapnii, 'eegfmri/alex/scan3/alex_alpha_corrmap.nii.gz')

# meanbold = nib.load('eegfmri/genevieve/retino_gamma_01.nii.gz')
# corrmapnii = nib.Nifti1Image(corrmap, meanbold.affine, meanbold.header)
# nib.save(corrmapnii, 'eegfmri/genevieve/scan1/genevieve_alpha_corrmap.nii.gz')

# meanbold = nib.load('eegfmri/genevieve/retino_gamma_02.nii.gz')
# corrmapnii = nib.Nifti1Image(corrmap, meanbold.affine, meanbold.header)
# nib.save(corrmapnii, 'eegfmri/genevieve/scan2/genevieve_alpha_corrmap.nii.gz')

# meanbold = nib.load('eegfmri/genevieve/retino_rest.nii.gz')
# corrmapnii = nib.Nifti1Image(corrmap, meanbold.affine, meanbold.header)
# nib.save(corrmapnii, 'eegfmri/genevieve/scan3/genevieve_alpha_corrmap.nii.gz')

# meanbold = nib.load('eegfmri/russell/retino_gamma_01.nii.gz')
# corrmapnii = nib.Nifti1Image(corrmap, meanbold.affine, meanbold.header)
# nib.save(corrmapnii, 'eegfmri/russell/scan1/russell_alpha_corrmap.nii.gz')

# meanbold = nib.load('eegfmri/russell/retino_gamma_02.nii.gz')
# corrmapnii = nib.Nifti1Image(corrmap, meanbold.affine, meanbold.header)
# nib.save(corrmapnii, 'eegfmri/russell/scan2/russell_alpha_corrmap.nii.gz')

meanbold = nib.load('eegfmri/russell/retino_rest.nii.gz')
corrmapnii = nib.Nifti1Image(corrmap, meanbold.affine, meanbold.header)
nib.save(corrmapnii, 'eegfmri/russell/scan3/russell_alpha_corrmap.nii.gz')



"""
Step 2e : Bandpass filter the denoised EEG data to gamma band (40-70 Hz)
"""
  
# Gamma Comps
gamma_comps = mne.filter.filter_data(goodcomps, 250,40,70)
gamma_comps = np.abs(gamma_comps)

smooth_gamma_comps = gaussian_filter1d(gamma_comps, 250, axis =1)
trimgamma = np.mean(smooth_gamma_comps[:, start:], axis=0)

plt.plot(trimgamma)   


"""
Step 4a :  Downsample to match the fMRI temporal resolution
""" 

# downsample gamma
n_timepoints = volume_times.shape[0]
resampled_gamma = resample(trimgamma,n_timepoints)  


"""
Step 4b :  Convolve the resampled, bandpass-filtered EEG power with the canonical hemodynamic response function
""" 

shift = int(5/0.693)
shift_gamma = np.roll(resampled_gamma,shift)

shift_gamma = mne.filter.filter_data(shift_gamma, 1/0.693, 0.01, 0.7)

gamma4d = np.zeros(bold.shape)

gamma4d = gamma4d[:,:,:,0:shift_gamma.shape[0]] + shift_gamma

timepoints = np.arange(30,450)


"""
Step 4c :  Correlate the convolved EEG alpha (40-70 Hz) power with fMRI in each voxel
"""

#correlation for gamma
corrmap_gamma = np.sum((bold[:,:,:,timepoints] - np.mean(bold[:,:,:,timepoints],axis=3,keepdims=True)) *
                  (gamma4d[:,:,:,timepoints] - np.mean(gamma4d[:,:,:,timepoints],axis=3,keepdims=True)), axis=3)\
    / (np.sqrt(np.sum((bold[:,:,:,timepoints] - np.mean(bold[:,:,:,timepoints], axis=3, keepdims= True))**2, axis=3))
        * np.sqrt(np.sum((gamma4d[:,:,:,timepoints] - np.mean(gamma4d[:,:,:,timepoints],axis=3,keepdims=True))**2,axis=3)))


for i in np.arange(0,33):
    plt.subplot(3,11,i+1)
    plt.imshow(np.rot90(corrmap_gamma[:,:,i]),vmin=-0.3, vmax=0.3)
    plt.title(i)
    

# corrmapnii_gamma = nib.Nifti1Image(corrmap_gamma, meanbold.affine, meanbold.header)
# nib.save(corrmapnii_gamma, f'eegfmri/alex/scan1/alex_gamma_corrmap.nii.gz')

# corrmapnii_gamma = nib.Nifti1Image(corrmap_gamma, meanbold.affine, meanbold.header)
# nib.save(corrmapnii_gamma, f'eegfmri/alex/scan2/alex_gamma_corrmap.nii.gz')

# corrmapnii_gamma = nib.Nifti1Image(corrmap_gamma, meanbold.affine, meanbold.header)
# nib.save(corrmapnii_gamma, f'eegfmri/alex/scan3/alex_gamma_corrmap.nii.gz')

# corrmapnii_gamma = nib.Nifti1Image(corrmap_gamma, meanbold.affine, meanbold.header)
# nib.save(corrmapnii_gamma, f'eegfmri/genevieve/scan1/genevieve_gamma_corrmap.nii.gz')

# corrmapnii_gamma = nib.Nifti1Image(corrmap_gamma, meanbold.affine, meanbold.header)
# nib.save(corrmapnii_gamma, f'eegfmri/genevieve/scan2/genevieve_gamma_corrmap.nii.gz')

# corrmapnii_gamma = nib.Nifti1Image(corrmap_gamma, meanbold.affine, meanbold.header)
# nib.save(corrmapnii_gamma, f'eegfmri/genevieve/scan3/genevieve_gamma_corrmap.nii.gz')

# corrmapnii_gamma = nib.Nifti1Image(corrmap_gamma, meanbold.affine, meanbold.header)
# nib.save(corrmapnii_gamma, 'eegfmri/russell/scan1/russell_gamma_corrmap.nii.gz')

# corrmapnii_gamma = nib.Nifti1Image(corrmap_gamma, meanbold.affine, meanbold.header)
# nib.save(corrmapnii_gamma, 'eegfmri/russell/scan2/russell_gamma_corrmap.nii.gz')

corrmapnii_gamma = nib.Nifti1Image(corrmap_gamma, meanbold.affine, meanbold.header)
nib.save(corrmapnii_gamma, 'eegfmri/russell/scan3/russell_gamma_corrmap.nii.gz')





