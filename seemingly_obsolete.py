def crop_img(img):
    """
    Crop the image to select the region of interest
    """
    x_min = 0
    x_max = 1200
    y_min = 0
    y_max = 1200 
    return img[y_min:y_max,x_min:x_max]

def crop_vid(video):
    """
    Crop the image to select the region of interest
    """
    x_min = 0
    x_max = 1200
    y_min = 0
    y_max = 1200 

    return video[:,y_min:y_max,x_min:x_max]

def preprocess_foam(img):
    """
    Apply image processing functions to return a binary image
    
    """
    #img = crop(img)
    # Apply thresholds
    
    adaptive_thresh = filters.threshold_local(img,91)
    idx = img > adaptive_thresh
    idx2 = img < adaptive_thresh
    img[idx] = 0
    img[idx2] = 201
    
    img = ndimage.binary_dilation(img)
    img = ndimage.binary_dilation(img).astype(img.dtype)
    img = ndimage.binary_dilation(img).astype(img.dtype) 
    img = ndimage.binary_dilation(img).astype(img.dtype)
    img = ndimage.binary_dilation(img).astype(img.dtype)
    

    return util.img_as_int(img)

#plt.imshow(preprocess_foam(img))
#pixel_val = preprocess_foam(img)

#print(pixel_val[20])
#list_zero = []

#for i in pixel_val[20].tolist():

    #if i == 0:
        
        #list_zero.append(i)
#print(len(list_zero))


def find_d_single_from_df(r):
    from probfit import BinnedLH, Chi2Regression, Extended,BinnedChi2,UnbinnedLH,gaussian
    import iminuit as Minuit
    r = np.asarray(r)
    def f1(r,D):
        if D> 0:
            return (r/(2.*D*tracking_time))*np.exp(-((r**2)/(4.*D*tracking_time))) # remember to update time
        else:
            return 0
    compdf1 = probfit.functor.AddPdfNorm(f1)
    
    ulh1 = UnbinnedLH(compdf1, r, extended=False)
    import iminuit
    m1 = iminuit.Minuit(ulh1, D=1., limit_D = (0,5),pedantic= False,print_level = 0)
    m1.migrad(ncall=10000)
    return m1.values['D']
     