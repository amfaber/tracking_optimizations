def signal_extractor_no_pos(video, full_tracked, red_blue,roi_size,bg_size, params, pool = None):  # change so taht red initial is after appearance timing
    lip_int_size= roi_size
    lip_BG_size = bg_size
    def cmask(index, array, BG_size, int_size):
        a, b = index
        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
        mask2 = x * x + y * y <= lip_int_size   # to make a "gab" between BG and roi
    
        BG_mask = (x * x + y * y <= lip_BG_size)
        BG_mask = np.bitwise_xor(BG_mask, mask2)
        return (sum((array[mask]))), np.median(((array[BG_mask])))
    
    full_tracked = full_tracked.sort_values(['frame'], ascending=True)
    

    size_maker = np.ones(video[0].shape)
    ind = 25, 25  # dont ask - leave it here, it just makes sure the below runs
    mask_size, BG_size = cmask(ind, size_maker, lip_BG_size, lip_int_size)
    mask_size = np.sum(mask_size)

    if pool is None:
        a = full_tracked.apply(lambda row: get_intensity_and_background(row['y'], row['x'], video, row['frame']), axis=1)
    else:
        a = pool.map(partial(wrap_int_bg, params = params), full_tracked[['x', 'y', 'frame']].to_numpy())

    intensity = []
    bg = []
    for line in a:
        i, b = line
        bg.append(b)
        intensity.append(i)
    if red_blue == 'blue' or red_blue == 'Blue':
        full_tracked['blue_int'] = intensity
        full_tracked['blue_bg'] = bg
        full_tracked['blue_int_corrected'] = (full_tracked['blue_int']) - (full_tracked['blue_bg']*mask_size)
    elif red_blue == 'red' or red_blue == 'Red':
        full_tracked['red_int'] = intensity
        full_tracked['red_bg'] = bg
        full_tracked['red_int_corrected'] = (full_tracked['red_int']) - (full_tracked['red_bg']*mask_size)
    else:
        full_tracked['green_int'] = intensity
        full_tracked['green_bg'] = bg
        full_tracked['green_int_corrected'] = (full_tracked['green_int']) - (full_tracked['green_bg']*mask_size)

    return full_tracked