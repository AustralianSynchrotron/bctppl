#!/usr/bin/env python3



##
##
##
##
##    This algorithm was implemented by Ashkan Pakzad <ashkan.pakzad@unimelb.edu.au>
##    and extracted from his repository github.com:quell-devs/TomoApp.git
##
##
##
##



import tqdm
import scipy.ndimage as ndi
import numpy as np
import argparse
import os
import commonsource as cs



myPath = os.path.dirname(os.path.realpath(__file__))


parser = argparse.ArgumentParser(description='Ring artefact removal.')
parser.add_argument('input', type=str, default="",
                    help='Input stack of CT projections to fill.')
parser.add_argument('output', type=str, default="",
                    help='Output HDF5 file.')
parser.add_argument('-c', '--correct', action='store_true', default=False,
                    help='Perform contrast correctrion.')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='Be verbose.')

args = parser.parse_args()



def _bilateral_filter_1d(signal: np.ndarray, wing_size: int, sigma_x: float, sigma_i: float):
    """Apply bilateral filter to 1D signal

    TODO: Optimise this function.

    Args:
        signal: 1D numpy array input signal
        sigma_x: spatial standard deviation
        sigma_i: intensity standard deviation
        wing_size: size of filter window (one side)

    Returns:
        Filtered signal
    """
    n = len(signal)
    result = np.zeros_like(signal)

    # Pad signal for edge handling
    padded = np.pad(signal, wing_size, mode='reflect')

    # For each point in signal
    for i in range(n):
        # Get window centered on current point
        window = padded[i:i+2*wing_size+1]
        center = signal[i]

        # Compute spatial and intensity weights
        x_weights = np.exp(-np.arange(-wing_size, wing_size+1)**2 / (2*sigma_x**2))
        i_weights = np.exp(-(window - center)**2 / (2*sigma_i**2))

        # Combine weights and normalize
        weights = x_weights * i_weights
        weights = weights / np.sum(weights)

        # Weighted average
        result[i] = np.sum(window * weights)

    return result




inData = cs.getInData(args.input, verbose=args.verbose)
(angles, height, width) = inData.shape
# make average reading projections one by one to avoid loading whole volume into memory
s1d_mean_all = np.zeros((height, width))
if args.verbose :
    print('Computing mean of input projections...', flush=True)
for i in tqdm.tqdm(range(1,angles-1), disable=not args.verbose):
    s1d_mean_all += inData[i,:,:]
if args.verbose :
    print('... Done', flush=True)
s1d_mean_all /= (inData.shape[0] - 2)
outWrapper = cs.OutputWrapper(args.output, inData.shape)

if args.verbose :
    print('Removing rings ...', flush=True)
for i in tqdm.tqdm(range(height), disable=not args.verbose):

    sino_org = np.array(inData[:,i,:])
    sino = np.copy(sino_org)

    # average row
    s1d_mean = s1d_mean_all[i]
    # median smooth average row, size 3
    s1d_med = ndi.median_filter(s1d_mean, size=3)

    ### determine parameters of bilateral filter
    # effective sinogram width
    threshold = 0.015 * (np.max(s1d_mean) - np.min(s1d_mean)) + np.min(s1d_mean)
    non_zero = s1d_mean > threshold
    if np.sum(non_zero) < 2:
        #logging.warning(f'No valid foreground detected in row {i}, skipping')
        continue
    width_eff_lims = np.where(non_zero)[0][[0, -1]]
    width_eff = width_eff_lims[1] - width_eff_lims[0]
    bilateral_wing = 0.0055 * width_eff

    # spatial standard deviation
    sigma_x = (2*bilateral_wing+1)/6

    # brightness standard deviation
    heur_q = np.quantile(s1d_mean, 0.9)
    heur_wing = int(np.round(2*bilateral_wing))
    heur_wing = max(heur_wing, 1)
    heur_sigma_x = (2*heur_wing+1)-(1/6)
    sigma_i = 0.95 * np.max(np.abs(_bilateral_filter_1d(s1d_med,heur_wing,heur_sigma_x,heur_q) - s1d_med))
    sigma_i = max(sigma_i, 1e-6)

    # apply bilateral filter
    s1d_filt = _bilateral_filter_1d(s1d_med, int(np.round(bilateral_wing)), sigma_x, sigma_i)

    # compute detector anomalies
    s1d_err = s1d_mean - s1d_filt

    # subtract detector anomalies from each row
    s_err = np.tile(s1d_err, (angles, 1))
    sino = sino - s_err

    # contrast correction
    if args.correct:
        contrast_fac = (np.mean(s1d_err)/width) * sino_org
        sino = sino + contrast_fac

    # store
    outWrapper.put(sino, np.s_[:,i,:], flush=False)

if args.verbose :
    print('... Done', flush=True)




