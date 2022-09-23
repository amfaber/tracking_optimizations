from time import time
whole = time()
import pandas as pd
import torch_tracking
import torch
import numpy as np
import tifffile
import trackpy as tp

class VideoChunker:
    def __init__(self,
     filepath,
     gb_limit = 1,
     dtype = None,
     n_chunks = None,
     return_frame_correction = True,
     offset = None,
     division = None,
     transform = None,
     auto_apply_transform = True,
     pad_video = False,
     ):
        self.vid_file = tifffile.TiffFile(filepath).series[0]
        self.frames, self.height, self.width = self.vid_file.shape
        self.return_frame_correction = return_frame_correction
        self.offset = offset
        self.division = division
        self.transform = transform
        self.auto_apply_transform = auto_apply_transform
        self.mean_frame = None
        self.pad_video = pad_video

        if dtype is None:
            self.dtype = np.dtype(np.float32) 
        else:
            self.dtype = np.dtype(dtype)
        self.dtype_size = self.dtype.itemsize
        if n_chunks == None:
            self.byte_limit = gb_limit * (1<<30)
            self.n_chunks = np.ceil(self.frames / (self.byte_limit / (self.dtype_size * self.height * self.width))).astype(int)
        else:
            self.n_chunks = n_chunks
        self.frames_per_load = np.ceil(self.frames / self.n_chunks).astype(int)
        self.__iter__()
    
    def apply_transform(self, video):
        if self.offset is not None:
            video = np.subtract(video, self.offset, out = video)
        if self.division is not None:
            video = np.divide(video, self.division, out = video)
        if self.transform is not None:
            video = self.transform(video)
        return video
    
    def get_mean_frame(self):
        part_means = []
        weights = []
        auto_apply_mem = self.auto_apply_transform
        if self.transform is None:
            self.auto_apply_transform = False
        else:
            self.auto_apply_transform = True
        for frame_correction, vid_chunk in self:
            part_means.append(vid_chunk.mean(axis = 0))
            weights.append(vid_chunk.shape[0])
            del vid_chunk
        self.auto_apply_transform = auto_apply_mem
        self.mean_frame = np.average(part_means, axis = 0, weights = weights)
        if self.transform is None:
            self.apply_transform(self.mean_frame)

    # def get_illumination_profile_across_chunks(self):
    #     if self.n_chunks > 1:
    #         if self.mean_frame is None:
    #             self.get_mean_frame()
    #         correction = get_correction_profile(self.mean_frame)
    #         if self.transform is None:
    #             self.division *= correction
    #         else:
    #             self.transform = lambda vid: np.divide(self.transform(vid), correction, out = vid)
    #     else:
    #         self.transform = lambda vid: Correct_illumination_profile(vid, inplace = True)

    def __iter__(self):
        self.chunk_idx = 0
        return self
    
    def __next__(self):
        if self.chunk_idx == self.n_chunks:
            raise StopIteration
        
        chunk = self.vid_file.asarray(
            key = slice(self.chunk_idx * self.frames_per_load, (self.chunk_idx+1)*self.frames_per_load))
        if self.pad_video:
            padded = np.full((chunk.shape[0], chunk.shape[1] + 1, chunk.shape[2] + 1), np.nan, dtype = self.dtype)
            padded[:, :-1, :-1] = chunk
            chunk = padded[:, :-1, :-1]
        else:
            chunk = chunk.astype(self.dtype)
        
        if self.auto_apply_transform:
            chunk = self.apply_transform(chunk)
            if self.n_chunks == 1:
                self.mean_frame = chunk.mean(axis = 0)
            
        frame_correction = self.chunk_idx * self.frames_per_load
        self.chunk_idx += 1
        if self.return_frame_correction:
            return frame_correction, chunk
        else:
            return chunk

class Params:
    defaults = {
        # 'mean_multiplier': 0.8,
        # 'sep': 5,
        # 'object_size': 9,
        # 'lip_int_size': 9,
        # 'lip_BG_size': 60,
        # 'memory': 1,
        # 'search_range': 5,
        # 'duration_filter': 20,
        # 'tracking_time': 0.036,
        # 'pixel_size': 0.18333,
        # 'n_processes': 8,
        # 'save_path': None,
        # 'exp_type_folders': None,
        # 'exp_types': None,
        # 'gap_size': 0,
        }
    values = defaults.copy()

    def __init__(self, parampath = None, **kwargs):
        super().__setattr__("_path", parampath)
        self.__dict__.update(self.defaults)
        # if parampath is not None:
        #     with open(parampath, 'r') as f:
        #         yaml_contents = yaml.safe_load(f)
        #         self.values.update(yaml_contents)
        self.values.update(kwargs)
        self.__dict__.update(self.values)
    
    def __getitem__(self, key):
        return self.values[key]
    
    def __setitem__(self, key, value):
        self.values[key] = value
        self.__dict__.update(self.values)

    def __setattr__(self, key, value):
        self.values[key] = value
        self.__dict__.update(self.values)
    
    # def to_yaml(self, path):
    #     with open(path, 'w') as f:
    #         yaml.dump(self.values, f)

    def __str__(self):
        return str(self.values)
    
    def __repr__(self):
        return str(self.values)
    
    # def update_mm_and_sep(self, path):
    #     with open(path, 'r+') as f:
    #         contents = f.read()
    #         contents = re.sub(r'mean_multiplier: \d+\.?\d*', 'mean_multiplier: ' + str(self.mean_multiplier), contents)
    #         contents = re.sub(r'sep: \d+\.?\d*', 'sep: ' + str(self.sep), contents)
    #         f.seek(0)
    #         f.write(contents)
if __name__ == "__main__":

    params = Params(
        lip_int_size = 8,  #originally 15 for first attemps
        lip_BG_size = 70,   # originally 40 for first attemps
        gap_size = 2, # adding gap
        
        dynamic_sep = 7,   # afstand mellem centrum af to partikler, 7 er meget lidt så skal være højere ved lavere densitet
        dynamic_mean_multiplier = 12,  #hvor mange partikler finder den, around 1-3, lavere giver flere partikler
        dynamic_object_size = 11, #diameter used in tp.locate, odd integer
        dynamic_search_range = 4,
        dynamic_memory = 0,
        
        static_sep = 15,
        static_mean_multiplier = 4,
        static_object_size = 5,
    )
    vid_path = r"C:\Users\andre\Documents\tracking_optimizations\emily_tracking\sample_vids\s_20.tif"
    # vid_path = r"/Users/amfaber/Documents/tracking_script/emily_tracking/sample_vids/s_20.tif"
    chunker = VideoChunker(vid_path,
            gb_limit = 0.7,
            dtype = np.float32,
            pad_video = True,
            )


    device = "cuda"
    full = time()
    
    # test_tp = True
    test_tp = False
    full_df = []
    for frame_corr, chunk in chunker:
        # if test_tp:
        #     tp_df = tp.locate(chunk[0], params.static_object_size, separation = params.static_sep, engine = "python")
        #     test_tp = False
        each = time()
        tpadded_vid = torch.tensor(chunk.base, device = device
        )
        tvid = tpadded_vid[:, :-1, :-1]
        test = torch_tracking.locate(tvid, params.static_object_size, separation = params.static_sep, padded_vid = tpadded_vid, params = params,
         minmass = 300)
        
        test["frame"] += frame_corr
        full_df.append(test)
        del tvid
        del chunk
        # torch.cuda.empty_cache()
        print(f"This iteration took {time() - each}")
    full_df = pd.concat(full_df)
    print(f"All iterations took {time() - full}")
    # tp.quiet()
    tp.link(full_df, params.dynamic_search_range)
    # import pickle
    # with open("results.pckl", "wb") as file:
    #     pickle.dump((full_df, tp_df), file)
    print(f"Whole script took {time() - whole}")