def signal_extractor_no_pos(video, final_df, red_blue,roi_size,bg_size,gap_size):  # change so taht red initial is after appearance timing

    global lip_int_size
    global lip_BG_size
    lip_int_size= roi_size
    lip_BG_size = bg_size

    def get_bck_pixels(image,cutoff = 0.985,plot=False):
        #function added by Henrik which gives mask of background pixels=True
        import scipy.stats as stats
        xmin,xmax = np.percentile(image,0.1),np.percentile(image,99.9)

        bins = 200
        bw = (xmax-xmin)/bins
        mean,sig = np.mean(image),np.std(image)
        alpha_guess,beta_guess = (mean/sig)**2,sig**2/mean
        pval_mov = 1-bw*stats.gamma.pdf(image,alpha_guess,scale=beta_guess)
        background_pix =  pval_mov<cutoff
        if plot:
            fig,ax = plt.subplots(1,2,figsize=(11,5))
            c,edges = np.histogram(image.flatten(),range=(xmin,xmax),bins=bins)
            centers = edges[:-1]+bw/2
            x,y,sy = centers[c>0],c[c>0],np.sqrt(c[c>0])

            mean,sig = np.mean(image),np.std(image)
            alpha_guess,beta_guess = (mean/sig)**2,sig**2/mean

            ax[0].plot(x,np.sum(y)*bw*stats.gamma.pdf(x,alpha_guess,scale=beta_guess),color="darkred")
            ax[0].errorbar(x,y,sy,color="dimgrey",fmt=".",capsize=2)

            pval_mov = 1-bw*stats.gamma.pdf(image,alpha_guess,scale=beta_guess)
            background_pix =  pval_mov<cutoff
            im = ax[1].imshow(image,cmap="Greys")
            fig.colorbar(im,ax=ax[1])
            ax[1].imshow(background_pix,alpha=0.2,cmap="Reds")
            ax[0].axvline(x=stats.gamma.ppf(cutoff,alpha_guess,scale=beta_guess),color="k",linestyle="--")
        return background_pix
    # avoid calculating bck_pixels in each line of data frame, construct a 3d array first
#
#

#    for frame in tqdm(video):                       #SSRB added 16-03-2021
#        bck_video.append(get_bck_pixels(frame))
#        #SSRB added 16-03-2021
#
#    bck_video = np.asarray(bck_video)#SSRB added 16-03-2021
#
    def cmask(index, array, BG_size, int_size,gap_size):
        """
        Only used for size and hence bck doesnt matter
        """
        a, b = index
        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
        mask2 = x * x + y * y <= lip_int_size + gap_size   # to make a "gab" between BG and roi

        BG_mask = (x * x + y * y <= lip_BG_size)


        return (np.sum((array[mask]))), np.median(((array[BG_mask])))

    final_df = final_df.sort_values(['frame'], ascending=True)

    def df_extractor2(row):
        b, a = row['x'], row['y'] #b,a
        frame = int(row['frame'])
        array = video[frame]

        #bck_array = bck_video[frame] #SSRB added 16-03-2021

        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
        mask2 = x * x + y * y <= lip_int_size+gap_size  # to make a "gab" between BG and roi
        BG_mask = (x * x + y * y <= lip_BG_size)
        """Henrik added multiplication with get_bck_pixels"""
        BG_mask = np.bitwise_xor(BG_mask, mask2)#*bck_array #SSRB added 16-03-2021


#quantile = 0.5 = median for BG, try to change

        return np.sum((array[mask])), np.quantile(((array[BG_mask])),0.50),np.mean((array[mask])),np.mean((array[BG_mask])) # added np in sum


    size_maker = np.ones(video[0].shape)
    ind = 25, 25  # dont ask - leave it here, it just makes sure the below runs
    mask_size, BG_size = cmask(ind, size_maker, lip_BG_size, lip_int_size, gap_size)
    mask_size = np.sum(mask_size)

    a = final_df.apply(df_extractor2, axis=1)
    # a = df_extractor2(final_df, video)

    intensity = []
    bg = []
    mean = []
    b_mean = []
    for line in a:
        i, b, m, bm = line
        bg.append(b)
        intensity.append(i)
        mean.append(m)
        b_mean.append(bm)
    if red_blue == 'blue' or red_blue == 'Blue':
        final_df['blue_int'] = intensity
        final_df['blue_int_mean'] = mean
        final_df['blue_bg'] = bg
        final_df['blue_bg_mean'] = b_mean
        final_df['blue_int_corrected'] = (final_df['blue_int']) - (final_df['blue_bg']*mask_size)
        final_df['blue_int_gausscorrected'] = (final_df['blue_int']) - (final_df['background']*mask_size)

    elif red_blue == 'red' or red_blue == 'Red':
        final_df['red_int'] = intensity
        final_df['red_int_mean'] = mean

        final_df['red_bg'] = bg
        final_df['red_bg_mean'] = b_mean
        final_df['red_int_corrected'] = (final_df['red_int']) - (final_df['red_bg']*mask_size)
        final_df['red_int_gausscorrected'] = (final_df['red_int']) - (final_df['background']*mask_size)

    else:
        final_df['green_int'] = intensity
        final_df['green_int_mean'] = mean
        final_df['green_bg'] = bg
        final_df['green_bg_mean'] = b_mean
        final_df['green_int_corrected'] = (final_df['green_int']) - (final_df['green_bg']*mask_size)
        final_df['green_int_gausscorrected'] = (final_df['green_int']) - (final_df['background']*mask_size)


    return final_df