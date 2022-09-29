import trackpy as tp
import tifffile
from time import time

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
    start = time()
    vid = tifffile.imread(r"C:\Users\andre\Documents\tracking_optimizations\emily_tracking\sample_vids\s_20.tif")
    tp.quiet()
    df = tp.batch(vid, diameter = params.static_object_size, separation = params.static_sep)
    print(f"locating {time()-start}")
    now = time()
    tp.link(df, params.dynamic_search_range)
    print(f"linking {time()-now}")