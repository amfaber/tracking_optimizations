from soren_tracking import *

mean_multiplier = 0.8

sep = 5                 #pixel seperation, ahould be close t search_range
object_size = 9         # 5 originally 
lip_int_size = 9        #originally 14 for first attemps
lip_BG_size = 60        # originally 50 for first attemps

# Tracking parameters
memory = 1              #frame
search_range = 5        # pixels

duration_filter = 0    # frames, originally 5

tracking_time = 0.036   #s --> maybe should change according to exposue 

pixel_size = 0.18333    #µm 60x is 18333 nm

#actual data paths and save_path
exp_type_folders    = ['sample_vids/']
exp_types           = ['HEK_RC_NovoEarly_488_30min']       # experiment type for given folder before
                                             # For column in csv file to classify data
save_path =  'sample_output/'        # where to save .csv


# -------------------------------- Running ------------------------------------ #

list_of_vids = []
type_list = []
replica_list = []


for main_folder in range(len(exp_type_folders)):
    file_paths,relica_number,exp_type = create_list_of_vids(exp_type_folders[main_folder],exp_types[main_folder])
    list_of_vids.extend(file_paths)
    type_list.extend(exp_type)
    replica_list.extend(relica_number)  

print(list_of_vids)


for i in range(len(list_of_vids)):
    save_path1 = save_path
    path = list_of_vids[i]
    lipase = type_list[i]
    replica = replica_list[i]
    runner_tracker(path,lipase,save_path1,replica)


create_big_df(save_path) # saving data
