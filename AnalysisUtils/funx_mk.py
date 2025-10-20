from roifile import ImagejRoi
import numpy as np
import cv2 
import pynapple as nap

from scipy import signal
import pandas as pd
import scipy
import scipy.cluster.hierarchy as sch
import matread as mread
from scipy.signal import find_peaks

from colorprint import cprint
def create_mask(roifile,imgsize = np.array([512,512])):
    """Function to create a mask from imageJ roi files

    Input:
        roifile: .zip or .roi file from ImageJ
        imgsize: img size for tiff stacks, (512,512) or(256,256)

    Output:
        Pat_Masks: binary mask images, (imgsize x Nrois)
        Maskcenter: center of eack mask, [Nrois x 2 ]
    """
    maskrois = ImagejRoi.fromfile(roifile)
    Nrois = len(maskrois)
    Pat_Masks = np.empty(np.append(imgsize,Nrois))
    Maskcenter= np.empty([Nrois,2])
    for i, roi_ in enumerate(maskrois):
        XYs = roi_.coordinates()     
        mask = np.zeros(imgsize, dtype=np.int8)
        if roi_.roitype==1:
            xmin = int(np.min(XYs[:,0]))
            xmax = int(np.max(XYs[:,0]))
            ymin = int(np.min(XYs[:,1]))
            ymax = int(np.max(XYs[:,1]))
            mask[ymin:ymax+1, xmin:xmax+1]=int(1)
        else:
            cv2.fillConvexPoly(mask, XYs.astype('int32'), 1)
        mcenter = np.where(mask>0)
        Maskcenter[i,:]  = np.mean(mcenter,axis=1)
        Pat_Masks[:,:,i] = mask
    return Pat_Masks, Maskcenter

def mask_edge_detection(cell_masks):
    ncell = np.size(cell_masks,axis=1)
    Nx = np.sqrt(np.size(cell_masks,axis=0)).astype(int)
    edges = np.zeros([Nx,Nx,ncell])
    for i in range(ncell):
        img_ = np.uint8(np.reshape((cell_masks[:,i]>0),[Nx,Nx]).T)
        img_ = cv2.GaussianBlur(img_, (3,3),0)
        img = np.uint8(img_)
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        grad = np.sqrt(grad_x**2 + grad_y**2)
        edges[:,:,i] = (grad * 255 / grad.max()).astype(np.uint8)
    return np.mean(edges,axis=2)
def Kalman_Stack_Filter(imageStack, gain=0.5, percentvar=0.05):
    # Process input arguments
    if gain > 1.0 or gain < 0.0:
        gain = 0.8
    if percentvar > 1.0 or percentvar < 0.0:
        percentvar = 0.05

    # Copy the last frame onto the end so that we filter the whole way through
    imageStack = np.append(imageStack, imageStack[-1,:,:][np.newaxis,:,:], axis=0)

    # Set up variables
    stacksize,width, height= imageStack.shape
    tmp = np.ones((width, height))

    # Set up priors
    predicted = imageStack[0,:, :]
    predictedvar = tmp * percentvar
    noisevar = predictedvar

    # Now conduct the Kalman-like filtering on the image stack
    for i in range(1, stacksize-1):
        stackslice = imageStack[i+1,:, :]
        observed = stackslice

        Kalman = predictedvar / (predictedvar + noisevar)
        corrected = gain*predicted + (1.0-gain)*observed + Kalman*(observed-predicted)
        correctedvar = predictedvar*(tmp - Kalman)

        predictedvar = correctedvar
        predicted = corrected
        imageStack[i,:, :] = corrected
    imageStack = np.delete(imageStack,-1,axis=0)
    return imageStack
def kalman_filter_denoise( stack, gain = 0.5, variance = 0.5):
    """
    Performs Kalman denoising of a 3D array (time, width, height)
    adopted from python conversion  Diego Asua
    """
    assert len(stack) != 0, "Stack is empty."
    assert len(stack) != 1, "Stack must contain more than one element."
    stack = np.concatenate((stack, np.expand_dims(stack[-1], 0)))
    _, width, height = stack.shape

    previous = stack[0]
    predicted = np.tile(variance, (width, height))
    noise = predicted

    ones = np.ones((width, height))
    denoised = np.zeros_like(stack)

    for i_frame, frame in enumerate(tqdm(stack[1:])):
        estimate = predicted / (predicted + noise)
        corrected = (
            gain * previous
            + (1 - gain) * frame
            + estimate * (frame - previous)
        )

        predicted = predicted * (ones - estimate)
        previous = corrected
        denoised[i_frame] = corrected

    # get rid of extra frame at the end
    denoised = denoised[:-1]

    return denoised

#def PCA_calc(X, n_components=3, zscore =True)
#    from sklearn.preprocessing import StandardScaler
#    if zscore:
#        ss = StandardScaler(with_mean=True, with_std=True)
#        Xz = ss.fit_transform(X.T).T
#    else:
#        Xz = X
    #pca = PCA(n_components=n_components)
    #Xa_p = pca.fit_transform(Xa.T).T


 
def findpeaks(arr, height=1, width=1, distance=1):
    peaks_dict = {}
    inds_pos,pos = find_peaks(arr, height=height, width=width, distance=distance)
    peaks_dict['pos_peaks']= pos['peak_heights']
    peaks_dict['pos_locs']= inds_pos
    inds_neg,neg = find_peaks(arr * -1, height=height, width=width, distance=distance)
    peaks_dict['neg_peaks']= neg['peak_heights']*-1
    peaks_dict['neg_locs']= inds_neg
    
    return peaks_dict

def calc_onset_time(odor_responses, odor_list, threshold1 = 4,pre=2000,post=4000,window_duration = 500,threshold2=3, baseline=None,isnormalize=False):
    """Function to to calculate odor onset time and generate data frames.

    Input:
        odor responses: time X odor X trials
        odor_list: 
        threshold: use values >4, it is used to determine whether max value of 
        odor response exceed threshold*standard deviation of baseline
        pre: 2000 ms, in prepocessing usually I took 2000 ms before inhalation onset
        post: 4000 ms, in prepocessing usually I took 4000 ms after inhalation onset
        window_duration: 500 ms (default), the window size to determine response values for odor window and baseline

    Output:
        r: response vector (roi , odor)
        dfdata: converted to date frames with odor ids
    """
    ncell = np.shape(odor_responses[0])[1]
    nodor = len(odor_responses)
    cell_latencies = np.zeros([ncell,nodor])
    cell_latencies_tofirst  = np.zeros([ncell,nodor])
    std_baseline = np.zeros([ncell,nodor])
    mean_baseline = np.zeros([ncell,nodor])
    max_val = np.zeros([ncell,nodor])    
    mean_val = np.zeros([ncell,nodor])
    std_val = np.zeros([ncell,nodor])
    max_ind = np.zeros([ncell,nodor])
    response_sign= np.zeros([ncell,nodor])
    roi_name= np.zeros([ncell,nodor])
    response_rank= np.zeros([ncell,nodor])

    odor_val = np.empty([ncell,nodor],dtype=object)
    df = {}
    for kk,od_res in enumerate(odor_responses):
        if baseline is not None:
            odResMean = np.mean(od_res,axis=2)
            if np.size(od_res,axis=0)>200:  # for glom recording with 60Hz
                ids = find_responsiveness(odResMean-baseline,res_interval=np.arange(116,176), baseline_interval=np.arange(84,114),std_threshold = 2,mean_threshold=0.1)
            else:
                ids = find_responsiveness(odResMean-baseline,res_interval=np.arange(59,89), baseline_interval=np.arange(44,54),std_threshold = 2,mean_threshold=0.1)
            cellid = np.where(ids>0)
            cellids = cellid[0]
        else:
            cellids = np.linspace(0,ncell-1,ncell) 
        for i in range(ncell):
            roi_name[i,kk] = i +1
            odor_val[i,kk] = odor_list[kk]
            dffup1 = signal.resample(np.squeeze(np.mean(od_res[:,i,:], axis=1)), pre+post) # each trial is from -2 to 4 sec
            dff_mean = np.mean(dffup1[pre:pre+window_duration])
            if isnormalize:
                dffup =dffup1/dff_mean
            else:
                dffup =dffup1
            dff_windowed= dffup[pre:pre+window_duration]  # use first 500 ms
            std_baseline[i,kk] = np.std(dffup[pre-500:pre-50]) # pre inhalation baseline
            mean_baseline[i,kk]= np.mean(dffup[pre-500:pre-50]) #
            max_val[i,kk] = np.max(np.abs(dff_windowed))
            mean_val[i,kk] = np.mean(dff_windowed)
            std_val[i,kk] = np.std(dff_windowed)
            max_ind[i,kk] = np.argmax(np.abs(dff_windowed))
            # check response_mean>3*std_baseline and  max value>threshold*std_baseline
            if i in cellids:
                if ((max_val[i,kk] - mean_baseline[i,kk]) > (std_baseline[i,kk] * threshold1)) and max_ind[i,kk]>10 and ((mean_val[i,kk] - mean_baseline[i,kk]) > (threshold2* std_baseline[i,kk])):

                    if max_val[i,kk] > mean_baseline[i,kk]:
                        lat = np.argwhere(dff_windowed >(std_baseline[i,kk] * threshold1 ))
                        if len(lat)>0:
                            response_sign[i,kk] = 1
                            cell_latencies[i, kk] = lat[0] + 1e-2*np.random.randn(1) # add small random number to make them unique
                        else:
                            response_sign[i,kk] = 0
                            cell_latencies[i, kk] = np.nan
                elif  ((max_val[i,kk] - mean_baseline[i,kk]) > (std_baseline[i,kk] * threshold1))  and max_ind[i,kk]>10 and mean_val[i,kk]<0 and ((mean_val[i,kk] - mean_baseline[i,kk]) < (threshold2* std_baseline[i,kk])):
                    lat = np.argwhere(dff_windowed <(-1*std_baseline[i,kk] * threshold2 ))
                    if len(lat)>0:
                        response_sign[i,kk] = -1
                        cell_latencies[i, kk] = -1*lat[0]+2e3 + 20*np.random.randn(1)# add small random number to make them unique
                    else:
                        response_sign[i,kk] = 0
                        cell_latencies[i, kk] = np.nan
                else:
                    response_sign[i,kk] = 0
                    cell_latencies[i, kk] = np.nan
            else:
                response_sign[i,kk] = 0
                cell_latencies[i, kk] = np.nan
        response_rank[np.argsort(cell_latencies[:, kk]),kk]= np.arange(1,ncell+1)
    max_val_temp = np.copy(max_val)
    max_val_temp[np.isnan(cell_latencies)] = np.nan
    mean_val_temp = np.copy(mean_val)
    mean_val_temp[np.isnan(cell_latencies)] = np.nan
    max_normalized = ((max_val_temp.T/np.nanmax(max_val_temp,axis=1)).T) + 0.01*max_val_temp # to make odor cases unique by giving bias to high responses
    mean_normalized = ((mean_val_temp.T/np.nanmax(mean_val_temp,axis=1)).T) + 0.01*mean_val_temp
    
    max_amp_rank= np.zeros([ncell,nodor])
    mean_amp_rank= np.zeros([ncell,nodor])
    for jj,od_res in enumerate(odor_responses):
        max_amp_rank[np.argsort(1-max_normalized[:, jj]),jj]= np.arange(1,ncell+1)
        mean_amp_rank[np.argsort(1-mean_normalized[:, jj]),jj]= np.arange(1,ncell+1)
        cell_latencies_tofirst[:,jj] = cell_latencies[:,jj] - np.nanmin(cell_latencies[:,jj])
    response_rank[np.isnan(cell_latencies)] =np.nan
    mean_amp_rank[np.isnan(cell_latencies)] =np.nan
    max_amp_rank[np.isnan(cell_latencies)] =np.nan
    df['roi_id'] = (roi_name.flatten()).astype(int)
    df['odor_val'] = odor_val.flatten()
    df['response_sign'] = (response_sign.flatten()).astype(int)
    df['std_baseline'] = std_baseline.flatten()
    df['mean_baseline'] = mean_baseline.flatten()
    df['max_val'] = max_val.flatten()
    df['mean_odor_resp'] = mean_val.flatten()
    df['mean_resp'] = mean_val_temp.flatten()
    df['max_resp'] = max_val_temp.flatten()
    df['std_odor_resp'] = std_val.flatten()
    df['latencies'] = cell_latencies.flatten()
    df['latency_rank'] = response_rank.flatten()
    df['max_normalized'] = max_normalized.flatten()
    df['mean_normalized'] = mean_normalized.flatten()
    df['max_amp_rank'] = max_amp_rank.flatten()
    df['mean_amp_rank'] = mean_amp_rank.flatten()
    df['latencies_tofirst'] = cell_latencies_tofirst.flatten()

    return pd.DataFrame(df)



def calc_onset_time2(odor_responses, odor_list, threshold1 = 4,pre=2000,post=4000,window_duration = 500,threshold2=3, baseline=None):
    """Function to to calculate odor onset time and generate data frames.

    Input:
        odor responses: time X odor X trials
        odor_list: 
        threshold: use values >4, it is used to determine whether max value of 
        odor response exceed threshold*standard deviation of baseline
        pre: 2000 ms, in prepocessing usually I took 2000 ms before inhalation onset
        post: 4000 ms, in prepocessing usually I took 4000 ms after inhalation onset
        window_duration: 500 ms (default), the window size to determine response values for odor window and baseline

    Output:
        r: response vector (roi , odor)
        dfdata: converted to date frames with odor ids
    """
    ncell = np.shape(odor_responses[0])[1]
    nodor = len(odor_responses)
    cell_latencies = np.zeros([nodor,ncell])
    cell_latencies_tofirst  = np.zeros([nodor,ncell])
    std_baseline = np.zeros([nodor,ncell])
    mean_baseline = np.zeros([nodor,ncell])
    max_val =np.zeros([nodor,ncell])   
    mean_val = np.zeros([nodor,ncell])
    std_val = np.zeros([nodor,ncell])
    max_ind = np.zeros([nodor,ncell])
    response_sign= np.zeros([nodor,ncell])
    roi_name= np.zeros([nodor,ncell])
    response_rank= np.zeros([nodor,ncell])

    odor_val = np.empty([nodor,ncell],dtype=object)
    df = {}
    for kk,od_res in enumerate(odor_responses):
        if baseline is not None:
            odResMean = np.mean(od_res,axis=2)
            if np.size(od_res,axis=0)>200:  # for glom recording with 60Hz
                ids = find_responsiveness(odResMean-baseline,res_interval=np.arange(116,176), baseline_interval=np.arange(84,114),std_threshold = 2,mean_threshold=0.1)
            else:
                ids = find_responsiveness(odResMean-baseline,res_interval=np.arange(59,89), baseline_interval=np.arange(44,54),std_threshold = 2,mean_threshold=0.1)
            cellid = np.where(ids>0)
            cellids = cellid[0]
        else:
            cellids = np.linspace(0,ncell-1,ncell) 
        for i in range(ncell):
            roi_name[kk,i] = i +1
            odor_val[kk,i] = odor_list[kk]
            dffup = signal.resample(np.squeeze(np.mean(od_res[:,i,:], axis=1)), pre+post) # each trial is from -2 to 4 sec
            dff_windowed= dffup[pre:pre+window_duration]  # use first 500 ms
            std_baseline[kk,i] = np.std(dffup[pre-window_duration:pre-50]) # pre inhalation baseline
            mean_baseline[kk,i]= np.mean(dffup[pre-window_duration:pre-50]) #
            max_val[kk,i] = np.max(np.abs(dff_windowed))
            mean_val[kk,i] = np.mean(dff_windowed)
            std_val[kk,i] = np.std(dff_windowed)
            max_ind[kk,i] = np.argmax(dff_windowed)
            # check response_mean>3*std_baseline and  max value>threshold*std_baseline
            if i in cellids:
                if ((max_val[kk,i] - mean_baseline[kk,i]) > (std_baseline[kk,i] * threshold1)) and max_ind[kk,i]>10 and ((mean_val[kk,i] - mean_baseline[kk,i]) > (threshold2* std_baseline[kk,i])):

                    if max_val[kk,i] > mean_baseline[kk,i]:
                        lat = np.argwhere(dff_windowed >(std_baseline[kk,i] * threshold1 ))
                        if len(lat)>0:
                            response_sign[kk,i] = 1
                            cell_latencies[kk,i] = lat[0] + 1e-2*np.random.randn(1) # add small random number to make them unique
                        else:
                            response_sign[kk,i] = 0
                            cell_latencies[kk,i] = np.nan
                elif ((max_val[kk,i] - mean_baseline[kk,i]) > (std_baseline[kk,i] * threshold1)) and max_ind[kk,i]>10 and (np.abs(mean_val[kk,i] - mean_baseline[kk,i]) > (threshold2* std_baseline[kk,i])):
                    lat = np.argwhere(dff_windowed <(std_baseline[kk,i] * threshold1 ))
                    if len(lat)>0:
                        response_sign[kk,i] = -1
                        cell_latencies[kk,i] = lat[0]+1e3 + 1e-2*np.random.randn(1)# add small random number to make them unique
                    else:
                        response_sign[kk,i] = 0
                        cell_latencies[kk,i] = np.nan
                else:
                    response_sign[kk,i] = 0
                    cell_latencies[kk,i] = np.nan
            else:
                response_sign[kk,i] = 0
                cell_latencies[kk,i] = np.nan
        response_rank[kk,np.argsort(cell_latencies[kk, :])]= np.arange(1,ncell+1)

    df['odor_val'] = (odor_val.T).flatten()
    df['roi_id'] = ((roi_name.T).flatten()).astype(int)
    df['response_sign'] = (response_sign.T.flatten()).astype(int)
    df['std_baseline'] = (std_baseline.T).flatten()
    df['mean_baseline'] = (mean_baseline.T).flatten()
    df['max_val'] = (max_val.T).flatten()
    df['mean_odor_resp'] = (mean_val.T).flatten()
    df['std_odor_resp'] = (std_val.T).flatten()
    df['latencies'] = (cell_latencies.T).flatten()

    return pd.DataFrame(df)

def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Input:
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Output:
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    
    Usage:
    sns.heatmap(cluster_corr(df.corr()))
    corr_array[idx, :][:, idx],idx
    """
    from scipy.spatial.distance import squareform
    dist = 1 - corr_array
    #pairwise_distances = sch.distance.pdist(1-corr_array)
    pairwise_distances = squareform(dist)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :],idx
    return corr_array[idx, :][:, idx],idx
def calc_odor_cell_matrix(odor_responses,odor_list,time_window=[60,90]):
    """Function to generate responses for each rois and odors in the dataframe.

    Input:
        odor responses: time X odor X trials
        odor_list: 
        time_window : [inh_frame   inh_frame + duration]
    Output:
        pandas date frames 
    """

    nodor = len(odor_responses)
    nroi = (odor_responses[0].shape)[1]
    df = {}
    r = np.zeros([nroi,nodor])
    for i, (odor_res, odorname) in enumerate(zip(odor_responses,odor_list)):
        r[:,i] = np.mean(np.mean(odor_res[time_window[0]:time_window[1],:,:], axis=2), axis=0)
        df[odorname] = r[:,i]
    dfdata = pd.DataFrame(df)
    return r,dfdata

def calc_odor_correlation(odor_responses,odor_list,time_window=[60,90]):
    """Function to generate responses for each rois and odors in the dataframe.

    Input:
        odor responses: time X odor X trials
        odor_list: 
        time_window : [inh_frame   inh_frame + duration]
    Output:
        pandas date frames 
    """

    nodor = len(odor_responses)
    if isinstance(odor_responses, list):
        nroi = (odor_responses[0].shape)[1]
        df = {}
        r = np.zeros([nroi,nodor])
        for i, (odor_res, odorname) in enumerate(zip(odor_responses,odor_list)):
            if odor_res.ndim == 3: #means we need to do trials average also
                r[:,i] = np.mean(np.mean(odor_res[time_window[0]:time_window[1],:,:], axis=2), axis=0)
            else:
                r[:,i] = np.mean(odor_res[time_window[0]:time_window[1],:], axis=0)
            df[odorname] = r[:,i]
    else:
        nroi = (odor_responses.shape)[0]
        df = {}
        r = np.zeros([nroi,nodor])
        for i, odorname in enumerate(odor_list):
            if odor_responses.ndim == 3:
                r[:,i] = np.mean(np.mean(odor_responses[time_window[0]:time_window[1],:,:], axis=2), axis=0)

            else:
                r[:,i] = np.mean(odor_responses[time_window[0]:time_window[1],:], axis=0)
            df[odorname] = r[:,i]
        
    dfdata = pd.DataFrame(df)
    return r,dfdata

def calc_cell_correlation(odor_responses,cell_list,time_window=[60,90]):
    """Function to generate responses for each rois and odors in the dataframe.

    Input:
        odor responses: time X odor X trials
        cell_list: 
        time_window : [inh_frame   inh_frame + duration]
    Output:
        pandas date frames 
    """

    ncell = len(cell_list)
    if len(odor_responses) > 1:
        nroi = (odor_responses[0].shape)[1]
    else:
        nroi = (odor_responses[0].shape)[0]
    df = {}
    r = np.zeros([nroi,ncell])
    for i, (odor_res, odorname) in enumerate(zip(odor_responses,cell_list)):
        r[:,i] = np.mean(np.mean(odor_res[time_window[0]:time_window[1],:,:], axis=2), axis=0)
        df[odorname] = r[:,i]
    dfdata = pd.DataFrame(df)
    return r,dfdata

def create_odorlist(odor_conds):

    """Function to create odor list.
            None od ids saved as -- for olfa1
                             xx for olfa2
                             zz for olfa3 
        at the end of function if od is None it will appear as empty in the name and concentation
    Input:
        odor_conds:  it is in the form of od1/od2/od3:conc1/conc2/con3 nM 

    Output:
        odor_names : [od1 od2 od3] 
        odor_concs:  [conc1 conc2 conc3]
        stim_conds:  [SL xx]
    """
    odor_names = [[] for x in range(len(odor_conds))]
    odor_concs = [[] for x in range(len(odor_conds))]
    stim_conds = [[] for x in range(len(odor_conds))]
    for kk,z in enumerate(odor_conds):
        odorid = np.char.rsplit(z,':').tolist()[0][0]
        odoridsplitted = odorid.split('/')
        conc = np.char.rsplit(z,':').tolist()[0][1]
        concsplitted= conc.split('/')
        for i,odid in enumerate(odoridsplitted):
            if odid =='--' or odid =='xx' or odid == 'zz':
                odid = ''
                c = ''
                odor_concs[kk].append(c)
            else:
                odid = odid.capitalize()
                odor_concs[kk].append(concsplitted[i][:-2])
            odor_names[kk].append(odid)
        if len(concsplitted)==5:
            stim_conds[kk].append(concsplitted[-1])
        
    return odor_names, odor_concs, stim_conds

def sort_latency(data1,res_interval=np.arange(59,89), baseline_interval=np.arange(30,57),std_threshold = 3,mean_threshold=0.2,fps=30, baseline_res = None):
    """Function to find latencies. Neg responses are have latency offset of 2e4, No responses 1e4

    Input:
        data1: Time X cell
        res_interval: 1 sec after inhalation onset
        baseline_interval: 1 sec before inhalation onset
        std_threshold: used to compare max response in res_interval with baseline_interval
        mean_threshold: value for mean responses during res_interval
        fps: imaging rate (Hz)
    Output:
        sorted_Cell: based on their latencies
        lats: latency (ms)
    """
    if   baseline_res is None:
        data = data1
    else:
        data = data1 - baseline_res
    baseline_std = np.std(data[baseline_interval,:],axis=0)
    response_mean = np.mean(data[res_interval,:],axis=0)
    baseline_mean= np.mean(data[baseline_interval,:],axis=0)
    Ncell = np.size(data[0,:])
    lats = np.ones(Ncell)*(1e4)
    res_sign = np.zeros(Ncell)
    for i in range(0,Ncell):
        a =np.logical_and(np.abs(data[res_interval,i]-baseline_mean[i])>(std_threshold*baseline_std[i]),(response_mean[i]-baseline_mean[i])>(mean_threshold))
        if a.any():
            lats[i] =np.argmax(a)*fps  - 2*np.max(data[res_interval,i])
            res_sign[i] = 1
        else:
            b =np.logical_and(np.abs(data[res_interval,i]-baseline_mean[i])>(std_threshold*baseline_std[i]),(response_mean[i]-baseline_mean[i])<(-1*mean_threshold/2))
            if b.any()>0:
                lats[i] = -1*np.argmax(b)*fps + np.random.randn() + 2e4 # add random small values to make every latency value unique
                res_sign[i] = -1
    return np.argsort(lats),lats,res_sign


def find_responsiveness(data,res_interval=np.arange(59,89), baseline_interval=np.arange(30,58),std_threshold = 3,mean_threshold=0.2):
    """Function to find response type, exc or inh.

    Input:
        data: Time X cell
        res_interval: 1 sec after inhalation onset
        baseline_interval: 1 sec before inhalation onset
        std_threshold: used to compare max response in res_interval with baseline_interval
        mean_threshold: value for mean responses during res_interval
    Output:
        responsiveness_ind (1: exc, -1:inh, 0:noresponse)
    """
    baseline_std = np.std(data[baseline_interval,:],axis=0)
    response_mean = np.mean(data[res_interval,:],axis=0)
    Ncell = np.size(data[0,:])
    responsiveness_ind = np.zeros(Ncell)
    for i in range(0,Ncell):
        a =np.logical_and(np.abs(data[res_interval,i])>(std_threshold*baseline_std[i]),response_mean[i]>mean_threshold,data[res_interval,i]>0)
        if a.any():
            responsiveness_ind[i] = 1
        else:
            b =np.logical_and(np.abs(data[res_interval,i])>(std_threshold*baseline_std[i]),response_mean[i]<(-1*mean_threshold),data[res_interval,i]<0)
            if b.any()>0:
                responsiveness_ind[i] = -1

    return responsiveness_ind


def read_session(sessionname):
    """Function to read V73 mat files generated from odor imaging preprocessing.

    Input:
        sessionname: mat file
    Output:
        session dictionary 
    """
    data_session= mread.loadmat(sessionname + '_S_v73.mat',use_attrdict=True)
    odor_conds= data_session['Session']['UniqueConds']
    odor_trials=data_session['Session']['OdorTrials']
    nodor= np.size(odor_conds)

    odor_names, odor_concs,stim_conds= create_odorlist(odor_conds)
    odor_list= [''.join(x +y+z) for x,y,z in zip(odor_names, odor_concs,stim_conds)]
    Fluo= data_session['Session']['F']
    odor_responses= data_session['Session']['OdorResponse']
    session_fps = data_session['Session']['Infos']['fps']

    print('Number of Stimulus Types: {}'.format(len(odor_responses)))
    try:
        print('Number of Trial per Stimulus: {}'.format(len(odor_responses[0][0,0,:])))
    except:
        print('Number of Trial per Stimulus: {}'.format(len(odor_responses[0][0,:])))
    print('Dimensions of single trial array (# time points by # neuron # Trial): {}'.format(odor_responses[0].shape))
    session = {}
    session['sniffs'] = data_session['Session']['Sniffs']
    session['odor_list'] = odor_list
    session['odor_responses'] = odor_responses
    session['fps'] = session_fps
    session['nodor'] = nodor
    try:
        session['ncell'] = np.size(odor_responses[0][0,:,0])
    except:
        session['ncell'] = np.size(odor_responses[0][0,:])
    if "pre_inhs" in data_session["Session"]["VoyeurData"]:
        trial_read = data_session['Session']["Infos"]["TrialsRead"]
        pre_inhs_ = data_session['Session']['VoyeurData']['pre_inhs']
        post_inhs_ = data_session['Session']['VoyeurData']['post_inhs']
        session["pre_inhs"] = [pre_inhs_[i] for i in range(len(trial_read)) if trial_read[i]]
        session["post_inhs"] = [post_inhs_[i] for i in range(len(trial_read)) if trial_read[i]]
    if "CellMask" in data_session["Session"]:
        session['cell_mask'] = data_session['Session']['CellMask']


    return session

def read_stimsession(sessionname,create_pynapple_objects=False):
    """Function to read V73 mat files generated from stim/odor imaging preprocessing.
    Odor_list will be raw compare to read_session function

    Input:
        sessionname: mat file
    Output:
        session dictionary 
    """
    data_session= mread.loadmat(sessionname + '_S_v73.mat',use_attrdict=True)
    odor_conds= data_session['Session']['UniqueConds']
    odor_trials=data_session['Session']['OdorTrials']
    #odor_names, odor_concs= create_odorlist(odor_conds)
    nodor= np.size(odor_conds)
    Fluo= data_session['Session']['F']
    odor_responses= data_session['Session']['OdorResponse']
    session_fps = data_session['Session']['Infos']['fps']

    print('Number of Stimulus Types: {}'.format(len(odor_responses)))
    try:
        cprint('Number of Trial per Stimulus: {}'.format(len(odor_responses[0][0,0,:])),style='success')
    except:
        print('Number of Trial per Stimulus: {}'.format(len(odor_responses[0][0,:])))   
    session = {}
    session['sniffs'] = data_session['Session']['Sniffs']
    session['odor_trials'] = odor_trials
    session['odor_list'] = odor_conds
    session['odor_responses'] = odor_responses
    if "PID" in data_session["Session"]:
        PID_responses = data_session['Session']['PID']
        session['PID_responses'] = PID_responses
    session['fps'] = session_fps
    session['nodor'] = nodor
    try:
        session['ncell'] = np.size(odor_responses[0][0,:,0])
    except:
        session['ncell'] = np.size(odor_responses[0][0,0])
    session['cell_mask'] = data_session['Session']['CellMask']
    session['Fluo'] = Fluo

    session['voyeur_data'] = data_session['Session']['VoyeurData']

    if create_pynapple_objects:
        # Create Pynapple objects
        cprint('Creating Pynapple objects...')
        session_pynapple = create_pynapple_session(session, session['voyeur_data'], session_fps)
        session.update(session_pynapple)
    return session

def read_WSstimsession(sessionname):
    """Function to read V73 mat files generated from stim/odor imaging preprocessing.
    Odor_list will be raw compare to read_session function

    Input:
        sessionname: mat file
    Output:
        session dictionary 
    """
    data_session= mread.loadmat(sessionname + '_S_v73.mat',use_attrdict=True)
    odor_conds= data_session['Session']['UniqueConds']
    Fluo= data_session['Session']['F']
    nodor= len(odor_conds)
    odor_responses= np.squeeze(data_session['Session']['OdorResponse'])
    print('Number of Stimulus Types: {}'.format(len(odor_responses)))
    print('Number of Trial per Stimulus: {}'.format(len(odor_responses[0][0,0,:])))
    print('Dimensions of single trial array (# time points by # neuron # Trial): {}'.format(odor_responses[0].shape))
    session = {}
    session['odor_responses'] = odor_responses
    session['nodor'] = nodor
    session['ncell'] = np.size(odor_responses[0][0,:,0])
    session['cell_mask'] = data_session['Session']['CellMask']
    session['Fluo'] = Fluo
    session['odor_list'] = odor_conds
    return session

def create_pynapple_session(session, voyeur_data, fps):
    """Create Pynapple timestamp and time series objects from session data"""
    
    pynapple_objects = {}
    start_time = voyeur_data['paramsgottime'].min()/1000.0
    # 1. Frame triggers (imaging frame times)
    if 'frametrigger' in voyeur_data:
        frame_times = voyeur_data['frametrigger'] / 1000.0 - start_time  # Convert to seconds if needed

        # Create Pynapple Ts object for frame times
        pynapple_objects['frame_timestamps'] = nap.Ts(t=frame_times, time_units='s')
        
        # Create time support for the entire session
        session_start = frame_times[0]
        session_end = frame_times[-1]
        pynapple_objects['session_epochs'] = nap.IntervalSet(
            start=[session_start], 
            end=[session_end], 
            time_units='s'
        )
    
    # 2. Trial onset times (inhalation onset)
    if 'inh_onset' in voyeur_data:
        trial_onsets = voyeur_data['inh_onset'] / 1000.0 - start_time  # Convert to seconds if needed
        trial_onsets = trial_onsets[trial_onsets > 0]  # Remove zeros
        
        pynapple_objects['trial_onsets'] = nap.Ts(t=trial_onsets, time_units='s')
        
        # Create trial epochs (assuming some trial duration)
        trial_duration = 1.0  # seconds - adjust as needed
        trial_ends = trial_onsets + trial_duration
        
        pynapple_objects['trial_epochs'] = nap.IntervalSet(
            start=trial_onsets,
            end=trial_ends,
            time_units='s'
        )
    
    # 3. Create fluorescence time series data
    if 'Fluo' in session and 'frame_timestamps' in pynapple_objects:
        fluo_data = session['Fluo']
        frame_ts = pynapple_objects['frame_timestamps']
        
        # Ensure dimensions match
        if len(frame_ts) == fluo_data.shape[0]:
            # Create TsdFrame (time series data frame) for all cells
            pynapple_objects['fluorescence_tsd'] = nap.TsdFrame(
                t=frame_ts.t,
                d=fluo_data,
                time_units='s',
                columns=[f'cell_{i}' for i in range(fluo_data.shape[1])]
            )
        else:
            Nt = min(len(frame_ts), fluo_data.shape[0])
            cprint(f"Warning: Frame times ({len(frame_ts)}) and fluorescence data ({fluo_data.shape[0]}) dimensions don't match",style='warning')

            frame_ts = nap.Ts(t=frame_ts.t[:Nt], time_units='s')
            if fluo_data.ndim == 1:
                # Single cell, create Tsd
                single_cell_data = fluo_data[:Nt] if fluo_data.ndim == 1 else fluo_data[:Nt, 0]
                pynapple_objects['fluorescence_tsd'] = nap.Tsd(
                    t=frame_ts.t,
                    d=single_cell_data,
                    time_units='s'
                )
            else:
                # Multiple cells, create TsdFrame
                pynapple_objects['fluorescence_tsd'] = nap.TsdFrame(
                    t=frame_ts.t,
                    d=fluo_data[:Nt,:],
                    time_units='s',
                    columns=[f'cell_{i}' for i in range(fluo_data.shape[1])]
                )
    

    # 3. Create odor-specific epochs
    if 'trial_onsets' in pynapple_objects and 'odor_trials' in session:
        odor_epochs = {}
        for odor_idx, odor_cond in enumerate(session['odor_list']):
            # Find trials for this odor
            odor_trial_indices = session['odor_trials'][odor_idx] 
            
            if len(odor_trial_indices) > 0:
                odor_trial_starts = trial_onsets[odor_trial_indices.astype(int)]
                odor_trial_ends = odor_trial_starts + trial_duration
                
                odor_epochs[f'odor_{odor_idx}'] = nap.IntervalSet(
                    start=odor_trial_starts,
                    end=odor_trial_ends,
                    time_units='s'
                )
        
        pynapple_objects['odor_epochs'] = odor_epochs
        # 4. Additional behavioral timestamps
    behavioral_events = {}
    """     # Sniff times (if available)
        if 'firstlick' in voyeur_data:
            lick_times = voyeur_data['firstlick'] / 1000.0
            lick_times = lick_times[lick_times > 0]
            if len(lick_times) > 0:
                behavioral_events['first_lick'] = nap.Ts(t=lick_times, time_units='s') """
        # Laser times (if available) 
    if 'laserontime' in voyeur_data:
        laser_times = voyeur_data['laserontime'] / 1000.0 
        laser_times = laser_times[laser_times != 0]
        laser_times = laser_times - start_time
        if len(laser_times) > 0:
            behavioral_events['laser_ontimes'] = nap.Ts(t=laser_times, time_units='s')
        
    # Add any other behavioral events you need
    if behavioral_events:
        pynapple_objects['behavioral_events'] = behavioral_events
    cprint('Pynapple objects created successfully.',style='success')

    return pynapple_objects