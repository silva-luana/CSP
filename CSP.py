import pandas as pd
import numpy as np
from scipy.linalg import eig, inv

def csp(sig, mrk, basal):

    # separate dataset into Xhand and Xfoot

    Xhand = pd.DataFrame(columns = ['C3', 'C4', 'Cz'])
    Xfoot = pd.DataFrame(columns = ['C3', 'C4', 'Cz'])

    idxH = 0; idxF = 0
    
    # n_samples => number of samples per epoch you want to analyze
    n_samples = 350+basal # +100 is for the 1 sec basal samples

    for i in xrange(mrk.shape[0]):
	if mrk.loc[i,'class'] == 'hand':
	    for k in range(n_samples):
		Xhand.loc[(idxH*n_samples + k), 'C3'] = sig.loc[(i*n_samples + k), 'C3']
		Xhand.loc[(idxH*n_samples + k), 'C4'] = sig.loc[(i*n_samples + k), 'C4']
		Xhand.loc[(idxH*n_samples + k), 'Cz'] = sig.loc[(i*n_samples + k), 'Cz']
	    idxH += 1 # just counts when Xhand increases its index
	    
	elif mrk.loc[i, 'class'] == 'foot':
	    for j in range(n_samples):    
		Xfoot.loc[(idxF*n_samples + j), 'C3'] = sig.loc[(i*n_samples + j), 'C3']
		Xfoot.loc[(idxF*n_samples + j), 'C4'] = sig.loc[(i*n_samples + j), 'C4']
		Xfoot.loc[(idxF*n_samples + j), 'Cz'] = sig.loc[(i*n_samples + j), 'Cz']
	    idxF += 1

    n_channels = sig.shape[1]
    H = np.zeros((idxH, n_samples, n_channels))
    F = np.zeros((idxF, n_samples, n_channels))
    channels = ['C3', 'C4', 'Cz']

    for task in range(idxH):
	for chan in range(n_channels):
	    if chan == 0: chan_ = 'C3'
	    elif chan == 1: chan_ = 'C4' 
	    elif chan == 2: chan_ = 'Cz'
		
	    for sample in range(n_samples):
		H[task,sample,chan] = Xhand[chan_][sample]

    for task in range(idxF):
	for chan in range(n_channels):
	    if chan == 0: chan_ = 'C3'
	    elif chan == 1: chan_ = 'C4' 
	    elif chan == 2: chan_ = 'Cz'
		
	    for sample in range(n_samples):
		F[task,sample,chan] = Xfoot[chan_][sample]            

    # Calculate the spatial covariance
    Rhand = np.cov(Xhand.transpose())
    Rfoot = np.cov(Xfoot.transpose())

    # Normalize the covariance matrix over all trials of each group
    Rhand_ = Rhand / len(H)
    Rfoot_ = Rfoot / len(F)

    # Composite spatial covariance
    R = Rhand_ + Rfoot_

    # Generalized singular value decomposition
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig.html

    eigvals, eigvecs = eig(np.array(R, dtype = np.float32))

    E = np.diag(np.real(eigvals))
    U = eigvecs # left and right eigvecs are the same because covariance matrix is sqare

    # Withening transform matrix =>> P = E^(-1/2) * U^T
    P = inv(pow(E, .5))*U.transpose()

    # Common eigenvecs and the sum of corresponding eigenvals for 2 matrices will always be 1
    Shand = P * Rhand_ * P.transpose()
    Sfoot = P * Rfoot_ * P.transpose()

    # Projection matrix
    W = U.transpose()*P

    # Using W, the original EEG can be transformed into uncorrelated components
    # Z can be seen as EEG source components including common and specific components of different tasks
    Z = np.dot(W,sig.transpose().values)

    # The original EEG can be reconstructed as
    # The columns of W^(-1) are spatial patterns, which can be considered as EEG source distribution vectors.
    # The first and the last columns of W^(-1) are the most important spatial patterns, that explain the largest
    # variance of one task and the smallest variance of the other.
    # X = np.dot(inv(W),Z)
    
    return Z, W
  