import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt

def load_dataset(hsidataset, targetsig, root):
    #This loads the following items
    #1. The HSI image 'img'
    #2. The target signature 'tgt_sig'
    #3. The target location truth mask 'tgt_mask'
    
    #-----------------------------------
    #LOADING IMAGE AND BAND WAVELENGTHS
    #-----------------------------------
    
    #Path of image
    img_pth = root + '/' + hsidataset + '_reflectance'
    
    #Use the ENVI header file to extract band wavelengths
    img_lambdas = np.array([float(i) for i in envi.open(img_pth+'.hdr').metadata['wavelength']])
    
    #Both ENVI header file and image file are required by envi.open() to load the image as an array
    img     = np.array(envi.open(img_pth+'.hdr', img_pth+'.img').load())
    
    #These images from the SHARE2012 and SHARE2010 are stored in % reflectance x 100. Transform to (0,1) space
    img     = img/10000
    #Force negative reflectance values to be zero
    img[img<0]=0
    
    #---------------------------------
    #LOADING TARGET SPECTRAL SIGNATURE
    #---------------------------------
    
    #Path of target signature file
    tgt_pth = './targetspectra/' + targetsig + 'felt_from_share2012_asdmeasurements.txt'
    tgt_sig = np.loadtxt(tgt_pth)
    
    #Interpolate target signature to the same band wavelength values as the image
    tgt_sig = np.interp(img_lambdas, tgt_sig[:,0], tgt_sig[:,1])
    
    #Put target signature in reflectance space (0,1)
    if np.mean(tgt_sig)>1:
        tgt_sig = tgt_sig/10000
    
    plt.figure()
    plt.plot(img_lambdas, tgt_sig, c='r', label='target signature')
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Reflectance')
    plt.legend()
    #plt.show()
    plt.savefig('./figures/'+targetsig+'felt.png')
    
    #-------------------------------------
    #LOADING THE TARGET LOCATION TRUTHMASK
    #-------------------------------------
    tgt_mask_pth = './truthmasks/' + targetsig + 'felt_' + hsidataset
    tgt_mask     = np.array(envi.open(tgt_mask_pth+'.hdr', tgt_mask_pth+'.img').load())
    
    return([img, tgt_sig, tgt_mask])











    
    

    
    
    
    
    
    # Imageh preprocessing, normalization for the pretrained resnet
    
    #TRAIN directory
    train_path = 'D:/_RESEARCH/SSRNET/data/TRAIN/'+dataset+'/'
    #TEST Directory
    test_path  = 'D:/_RESEARCH/SSRNET/data/TEST/' +dataset+'/imgs_redtargets_on_90percentile_SAMbased_edges/'
    #test_path  = 'D:/_RESEARCH/SSRNET/data/TEST/' +dataset+'/'
    
    if dataset == 'conesus':
        #TRAIN File names
        train_refHSI_fnm = train_path+'conesus_0920_1722_refHSI_VNIR_SWIR_nbands327_TRAIN_1'
        train_LRHSI_fnm  = train_path+'conesus_LRHSI_TRAIN'
        train_HRMSI_fnm  = train_path+'conesus_HRMSI_TRAIN'
        
        #TEST File names
        test_refHSI_fnm  = test_path+'conesus_0920_1714_refHSI_VNIR_SWIR_nbands327_TEST_1_implanted_p0.2'
        test_LRHSI_fnm   = test_path+'conesus_LRHSI_TEST'
        test_HRMSI_fnm   = test_path+'conesus_HRMSI_TEST'
    
    elif dataset == 'rocdowntown':
        #TRAIN File names
        train_refHSI_fnm = train_path+'rocdowntown_0920_1559_refHSI_VNIR_SWIR_nbands327_TRAIN'
        train_LRHSI_fnm  = train_path+'rocdowntown_LRHSI_TRAIN'
        train_HRMSI_fnm  = train_path+'rocdowntown_HRMSI_TRAIN'
        
        #TEST File names
        test_refHSI_fnm  = test_path+'rocdowntown_0920_1604_refHSI_VNIR_SWIR_nbands327_TEST_implanted_p0.2'
        test_LRHSI_fnm   = test_path+'rocdowntown_LRHSI_TEST'
        test_HRMSI_fnm   = test_path+'rocdowntown_HRMSI_TEST'
    
    elif dataset == 'cupriteng':
        #TRAIN File names
        train_refHSI_fnm = train_path+'cupriteng_ang20200712t195248_corr_v2y1_nbands372'
        train_LRHSI_fnm  = train_path+'cupriteng_LRHSI_TRAIN'
        train_HRMSI_fnm  = train_path+'cupriteng_HRMSI_TRAIN'
        
        #TEST File names
        test_refHSI_fnm  = test_path+'cupriteng_ang20200712t200039_corr_v2y1_refHSI_nbands372'
        test_LRHSI_fnm   = test_path+'cupriteng_LRHSI_TEST'
        test_HRMSI_fnm   = test_path+'cupriteng_HRMSI_TEST'
        
    elif dataset == 'avon':
        #TRAIN File names
        train_refHSI_fnm = train_path+'avon_0920_1844_refHSI_VNIRbands_nbands327_TRAIN'
        train_LRHSI_fnm  = train_path+'avon_LRHSI_TRAIN'
        train_HRMSI_fnm  = train_path+'avon_HRMSI_TRAIN'
        
        #TEST File names
        test_refHSI_fnm  = test_path+'avon_0920_1851_refHSI_VNIRbands_nbands327_TEST_implanted_p0.2'
        test_LRHSI_fnm   = test_path+'avon_LRHSI_TEST'
        test_HRMSI_fnm   = test_path+'avon_HRMSI_TEST'
    
    elif dataset == 'ritcampus':
        #TRAIN file names
        train_refHSI_fnm = train_path+'ritcampus_0729_1935_refHSI_VNIR_SWIR_nbands327_TRAIN'
        train_LRHSI_fnm  = train_path+'ritcampus_0729_1935_LRHSI_TRAIN'
        train_HRMSI_fnm  = train_path+'ritcampus_0729_1935_HRMSI_TRAIN'
        
        #TEST File names
        test_refHSI_fnm  = test_path+'ritcampus_0729_1940_refHSI_VNIR_SWIR_nbands327_TEST_implanted_p0.2'
        test_LRHSI_fnm   = test_path+'ritcampus_LRHSI_TEST'
        test_HRMSI_fnm   = test_path+'ritcampus_HRMSI_TEST'
    
    elif dataset == 'paviau':
        #TRAIN file names
        train_refHSI_fnm = train_path+'paviau_refHSI_VNIR_nbands102'
        train_LRHSI_fnm  = train_path+'paviau_LRHSI_TRAIN'
        train_HRMSI_fnm  = train_path+'paviau_HRMSI_TRAIN'
        
        #TEST File names
        test_refHSI_fnm  = test_path+'paviau_refHSI_VNIRbands_nbands102_TEST'
        test_LRHSI_fnm   = test_path+'paviau_LRHSI_TEST'
        test_HRMSI_fnm   = test_path+'paviau_HRMSI_TEST'
    
        
    #TRAIN images
    train_refHSI = np.array(envi.open(train_refHSI_fnm+'.hdr', train_refHSI_fnm+'.img').load())
    train_LRHSI  = np.array(envi.open(train_LRHSI_fnm+'.hdr', train_LRHSI_fnm+'.img').load())
    train_HRMSI  = np.array(envi.open(train_HRMSI_fnm+'.hdr', train_HRMSI_fnm+'.img').load())
    
    #TEST images
    test_refHSI  = np.array(envi.open(test_refHSI_fnm+'.hdr', test_refHSI_fnm+'.img').load())
    test_LRHSI   = np.array(envi.open(test_LRHSI_fnm+'.hdr', test_LRHSI_fnm+'.img').load())
    test_HRMSI   = np.array(envi.open(test_HRMSI_fnm+'.hdr', test_HRMSI_fnm+'.img').load())
    
    # throwing up the edge
    w_edge          = train_refHSI.shape[0]//scale_ratio*scale_ratio-train_refHSI.shape[0]
    h_edge          = train_refHSI.shape[1]//scale_ratio*scale_ratio-train_refHSI.shape[1]
    w_edge          = -1  if w_edge==0  else  w_edge
    h_edge          = -1  if h_edge==0  else  h_edge
    
    #Throw up edge on train img
    train_refHSI   = train_refHSI[:w_edge, :h_edge, :]
    train_HRMSI    = train_HRMSI[:w_edge, :h_edge, :]
    
    #Throw up edge on test img
    test_refHSI    = test_refHSI[:w_edge, :h_edge, :]
    test_HRMSI     =  test_HRMSI[:w_edge, :h_edge, :]
    
    # cropping area for test image
    width, height, n_bands = test_refHSI.shape 
    w_str                  = (width - size) // 2 
    h_str                  = (height - size) // 2 
    w_end                  = w_str + size
    h_end                  = h_str + size
    test_refHSI_copy       = test_refHSI.copy()
    
    #--------------------------------------------------------------------------
    test_ref               = test_refHSI_copy[w_str:w_end, h_str:h_end, :].copy()
    test_lr                = wp.simLRHSI(test_ref, 'gaussian', 4)
    test_hr                = np.copy(test_HRMSI[w_str:w_end, h_str:h_end, :])
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    train_ref               = np.copy(train_refHSI)
    train_lr                = wp.simLRHSI(train_ref, 'gaussian', 4)
    train_hr                = np.copy(train_HRMSI)
    #--------------------------------------------------------------------------
    
    #Now crop the TEST refHSI image and save
    HRHSI_refHSI_test_cropped          = np.copy(test_ref)
    HRHSI_refHSI_test_cropped_fnm      = test_path + dataset + '_refHSI_cropped_ImgSize_' +str(size)
    NR_cropped, NC_cropped, NB_cropped = HRHSI_refHSI_test_cropped.shape
    envi.save_image(HRHSI_refHSI_test_cropped_fnm+'.hdr',HRHSI_refHSI_test_cropped,        force=True)
    print('\nSaved refHSI_cropped.max(): ',              HRHSI_refHSI_test_cropped.max())
    print('Saved refHSI_cropped.min(): ',                HRHSI_refHSI_test_cropped.min())
    
    #Transform to 8-bit
    if train_ref.mean()>=1:
        train_ref = 255*train_ref/10000
        train_lr  = 255*train_lr/10000
    
    if train_ref.mean()<1:
        train_ref = 255*train_ref
        train_lr  = 255*train_lr
        
    if test_ref.mean()>=1:
        test_ref = 255*test_ref/10000
        test_lr  = 255*test_lr/10000
    
    if test_ref.mean()<1:
        test_ref = 255*test_ref
        test_lr  = 255*test_lr
    
    train_hr = 255*train_hr/10000
    test_hr  = 255*test_hr/10000
    
    print('\ntrain_ref.shape: ', train_ref.shape)
    print('train_lr.shape: ',    train_lr.shape)
    print('train_hr.shape: ',    train_hr.shape)
    
    print('\ntrain_ref.mean(): ', train_ref.mean())
    print('train_lr.mean(): ',    train_lr.mean())
    print('train_hr.mean(): ',    train_hr.mean())
    
    print('\ntest_ref.shape: ',  test_ref.shape)
    print('test_lr.shape: ',     test_lr.shape)
    print('test_hr.shape: ',     test_hr.shape)
    
    print('\ntest_ref.mean(): ', test_ref.mean())
    print('test_lr.mean(): ',    test_lr.mean())
    print('test_hr.mean(): ',    test_hr.mean())
    
    train_ref = torch.from_numpy(train_ref).permute(2,0,1).unsqueeze(dim=0)
    train_lr = torch.from_numpy(train_lr).permute(2,0,1).unsqueeze(dim=0) 
    train_hr = torch.from_numpy(train_hr).permute(2,0,1).unsqueeze(dim=0) 
    test_ref = torch.from_numpy(test_ref).permute(2,0,1).unsqueeze(dim=0) 
    test_lr = torch.from_numpy(test_lr).permute(2,0,1).unsqueeze(dim=0) 
    test_hr = torch.from_numpy(test_hr).permute(2,0,1).unsqueeze(dim=0) 

    return [train_ref, train_lr, train_hr], [test_ref, test_lr, test_hr]