import numpy as np

#Matched filter
def calc_mf(hsi_img, tgtsig):
    if len(hsi_img.shape)!=3:
        raise Exception('image input should have shape (nrow,ncol,nbands)')
    X = np.copy(hsi_img)
    NR,NC,NB = X.shape
    X = X.reshape((NR*NC, NB))
    
    #Calculate image statistics: mean 'mu' and covariance 'S'
    mu    = np.mean(X, axis=0).reshape((1,-1))
    S     = np.cov(X, rowvar=False)
    Sinv  = np.linalg.inv(S)
    
    #Perform mean-correction on the image
    X = X-mu
    
    #Perform mean-correction on the target signature
    if len(tgtsig.shape)>1:
        raise Exception('target signature should be a 1-d array')
    T = np.copy(tgtsig).reshape((1,-1))
    T = T-mu
    #Transpose and represent the target signature with lowercase 't'
    t = T.T
    
    mf_scores = np.empty((NR*NC, ))
    for pix in range(NR*NC):
        if pix==0:
            print('Calculating matched filter (MF) scores...', end='')
        if pix==NR*NC-1:
            print('Done!')
        #Represent mu-corrected pixel of interest as column vector
        x        = X[pix, :].reshape((-1,1))
        
        #Calculate the MF score
        numer = float(np.dot(t.T, np.dot(Sinv, x)))
        denom = float(np.dot(t.T, np.dot(Sinv, t)))
        mf_scores[pix] = numer/denom
        
    #Reshape the MF scores to resemble the shape of the image
    mf_scores = mf_scores.reshape((NR, NC))
    return(mf_scores)

def calc_ace(hsi_img, tgtsig):
    if len(hsi_img.shape)!=3:
        raise Exception('image input should have shape (nrow,ncol,nbands)')
    X = np.copy(hsi_img)
    NR,NC,NB = X.shape
    X = X.reshape((NR*NC, NB))
    
    #Calculate image statistics: mean 'mu' and covariance 'S'
    mu    = np.mean(X, axis=0).reshape((1,-1))
    S     = np.cov(X, rowvar=False)
    Sinv  = np.linalg.inv(S)
    
    #Perform mean-correction on the image
    X = X-mu
    
    #Perform mean-correction on the target signature
    if len(tgtsig.shape)>1:
        raise Exception('target signature should be a 1-d array')
    T = np.copy(tgtsig).reshape((1,-1))
    T = T-mu
    #Transpose and represent the target signature with lowercase 't'
    t = T.T
    
    ace_scores = np.empty((NR*NC, ))
    for pix in range(NR*NC):
        if pix==0:
            print('Calculating ACE scores...', end='')
        if pix==NR*NC-1:
            print('Done!')
        #Represent pixel of interest as column vector
        x        = X[pix, :].reshape((-1,1))
        
        #Calculate the ACE numerator
        numer    = float(np.dot(t.T, np.dot(Sinv, x)))**2
        
        #Calculate ACE denominator
        denom1   = float(np.dot(x.T, np.dot(Sinv, x)))
        denom2   = float(np.dot(t.T, np.dot(Sinv, t)))
        denom    = denom1*denom2
        
        #Calculate arc cosine and assign theta value as this pixel's ACE score
        ace_scores[pix] = numer/denom
    
    #Reshape the ACE scores to resemble the shape of the image
    ace_scores = ace_scores.reshape((NR, NC))
    return(ace_scores)
    
#Calculate SAM as a target detector (strictly speaking, SAM is NOT a target detector)
def calc_sam(hsi_img, tgtsig):
    #Check shape of input image
    if len(hsi_img.shape)!=3:
        raise Exception('image input should have shape (nrow,ncol,nbands)')
    X = np.copy(hsi_img)
    NR,NC,NB = X.shape
    X = X.reshape((NR*NC, NB))
    
    #Check shape of target signature
    if len(tgtsig.shape)>1:
        raise Exception('target signature should be a 1-d array')
    T = np.copy(tgtsig).reshape((1,-1))
    
    #Background correct both the image and the reference target spectrum
    mu = np.mean(X, axis=0).reshape((1,-1))
    X  = X-mu
    T  = T-mu
    
    t  = T.T
    
    #Loop through all pixels, comparing each background-corrected pixel spectrum to the background-corrected target spectrum
    sam_scores = np.empty((NR*NC, ))
    for pix in range(NR*NC):
        if pix==0:
            print('Calculating SAM scores...', end='')
        if pix==NR*NC-1:
            print('Done!')
        x        = X[pix, :].reshape((-1,1))
        xmag     = np.sqrt(np.sum(x**2))
        tmag     = np.sqrt(np.sum(t**2))
        costheta = np.dot(t.T, x)/(xmag*tmag)
        
        #Numeric corrections
        if costheta>1:
            costheta = 1
        if costheta<-1:
            costheta = -1
        
        #Assign theta as SAM score
        sam_scores[pix] = np.arccos(costheta)
    
    #Reshape sam_scores to image dimensions
    sam_scores = sam_scores.reshape((NR, NC))
    return(sam_scores)    