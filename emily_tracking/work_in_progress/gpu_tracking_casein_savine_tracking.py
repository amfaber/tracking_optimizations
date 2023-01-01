import gpu_tracking

def track(casein_file, savinase_file):
	
	mean_casein = gpu_tracking.mean_from_disk(casein_file)
	
	casein_df = gpu_tracking.batch(
		mean_casein[None, ...],
		diameter = 9,
		snr = 1.5,
		minmass_snr = 0.3,
	)

	savinase_at_casein_spots = gpu_tracking.batch_file(
		savinase_file,
		diameter = 9,
		points_to_characterize = casein_df.drop(columns = "frame")
	)

	savinase_df = gpu_tracking.batch_file(
		savinase_file,
		9,
		snr = 1.5,
		minmass_snr = 0.3,
	)
	
if __name__ == "__main__":
	savinase = r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\testing\easy_test_data.tif"
	casein = r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\testing\easy_test_data.tif"
	track(casein, savinase)