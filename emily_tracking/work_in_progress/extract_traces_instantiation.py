import new_extract_traces as extr
from pathlib import Path
from hatzakis_lab_tracking import Params
import os

params = Params(
    lip_int_size = 8,  #originally 15 for first attemps
    lip_BG_size = 70,   # originally 40 for first attemps
    gap_size = 2, # adding gap
    
    dynamic_sep = 7,   # afstand mellem centrum af to partikler, 7 er meget lidt så skal være højere ved lavere densitet
    dynamic_mean_multiplier = 12,  #hvor mange partikler finder den, around 1-3, lavere giver flere partikler
    dynamic_object_size = 11, #diameter used in tp.locate, odd integer
    dynamic_search_range = 4,
    dynamic_memory = 0,
    
    static_sep = 8,
    static_mean_multiplier = 4,
    static_object_size = 11,

    n_processes = os.cpu_count(),
    fit_processes = 4,
)


folder = Path("../sample_vids")
no_particle_path = str(folder / "Experiment_Process_001_20220823.tif")

# streptavidin, 20 frames videos
video_static = [str(folder / "c_20.tif")]


# protease signal, 2000 frames videos
video_dynamic = [str(folder / "s_20.tif")]

save_path = str(folder / "chunking_new")
if __name__ == "__main__":
    extr.extract_traces_average(video_static, video_dynamic, no_particle_path, save_path, params = params, only_calibrate = [True]*2)