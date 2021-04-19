import pickle
import numpy as np

def GetStats(array):
	mean = np.mean(array, 0)
	stddev = np.std(array, 0)
	percentile = np.transpose(np.percentile(array, np.linspace(0, 100, 101), 0))
	return mean, stddev, percentile

src_seqs, tgt_seqs = pickle.load(open('aa/test.pkl', "rb"))
n_frames, n_velocities = 120, 119
angle_array = np.concatenate([seq for seq in src_seqs])
#  (change of angle)/frame.
velocity_array = angle_array[1:] - angle_array[:-1]
# remove every 120th entry to get rid bad velocity calculations between sequences.
velocity_array = np.delete(velocity_array, np.arange(n_frames - 1, velocity_array.shape[0], n_frames), 0)
# (change of angle)/frame^2.
acceleration_array = velocity_array[1:] - velocity_array[:-1]
# remove every 119th entry to get rid bad acceleration calculations between sequences.
acceleration_array = np.delete(acceleration_array, np.arange(n_velocities - 1, acceleration_array.shape[0], n_velocities), 0)

# mean shape = (72 x 1), stddev shape = (72 x 1), percentile shape (72 x 101) --> Every single percentile 0 through 100.
angle_mean, angle_stddev, angle_percentile = GetStats(angle_array)
velocity_mean, velocity_stddev, velocity_percentile = GetStats(velocity_array)
acceleration_mean, acceleration_stddev, acceleration_percentile = GetStats(acceleration_array)

pkl_file = open("stats.pkl", "wb")
pickle.dump(angle_mean, pkl_file)
pickle.dump(angle_stddev, pkl_file)
pickle.dump(angle_percentile, pkl_file)
pickle.dump(velocity_mean, pkl_file)
pickle.dump(velocity_stddev, pkl_file)
pickle.dump(velocity_percentile, pkl_file)
pickle.dump(acceleration_mean, pkl_file)
pickle.dump(acceleration_stddev, pkl_file)
pickle.dump(acceleration_percentile, pkl_file)
pkl_file.close()
