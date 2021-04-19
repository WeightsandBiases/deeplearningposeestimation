import numpy as np
import pickle

class Joint:
	def __init__(self, name):
		self.name = name
		self.raw_data = dict({'x': dict({'angles': [], 'velocities': [], 'accelerations': []}),
		                      'y': dict({'angles': [], 'velocities': [], 'accelerations': []}),
		                      'z': dict({'angles': [], 'velocities': [], 'accelerations': []})})
		self.stats = {}
		self.axes = ['x', 'y', 'z']
		self.kinematics = ['angles', 'velocities', 'accelerations']

	'''Gets rid of the raw data. Used to reduce object size prior tp writing to pickle.'''
	def ClearRaw(self):
		for axis in self.axes:
		self.raw_data[axis] = []

	'''Initializes the stats dictionary. Every axis gets 3 dicts of stats. Angle, angle velocity, angle acceleration.'''
	def InitStats(self):
		for axis in self.axes:
			self.stats[axis] = dict({'angles': dict({'mean': 0, 'stddev': 0, 'percentile': {}}),
									'velocities': dict({'mean': 0, 'stddev': 0, 'percentile': {}}),
									'accelerations': dict({'mean': 0, 'stddev': 0, 'percentile': {}})})

	'''Takes in a nx3 numpy array and assigns it to the raw_data dict.'''
	def SetStats(self, seq):
		for i, axis in enumerate(self.axes):
			self.raw_data[axis].append(seq[:,i])

	def CalculateMeanAndStd(self):
		for axis in self.axes:
			for kinematic in self.kinematics:
				n, tot = 0, 0
				for seq in self.raw_data[axis][kinematic]:
					n += seq.shape[0]
					tot += np.sum(seq)
				self.stats[axis]['angles']['mean'] = tot/n


def GetStats(array):
	mean = np.mean(array, 0)
	stddev = np.std(array, 0)
	percentile = np.transpose(np.percentile(array, np.linspace(0, 100, 101), 0))
	return mean, stddev, percentile

# Load data and calculate all stats.
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

angle_mean, angle_stddev, angle_percentile = GetStats(angle_array)
velocity_mean, velocity_stddev, velocity_percentile = GetStats(velocity_array)
acceleration_mean, acceleration_stddev, acceleration_percentile = GetStats(acceleration_array)




joint_names = []
joints = {}
# Create all the joint objects and populates the joints dictionary.


src_seqs, tgt_seqs = pickle.load(open('aa/test.pkl', "rb"))

angle_array = np.concatenate([seq for seq in src_seqs])
angle_mean = np.mean(angle_array, 0)
angle_stddev = np.std(angle_array, 0)
angle_percentile = np.transpose(np.percentile(angle_array, np.linspace(0, 100, 101), 0))

# angle/frame
velocity_array = angle_array[1:] - angle_array[:-1]

# remove every 120th entry to get rid bad velocity calculations between sequences
n_frames = 120
velocity_array = np.delete(velocity_array, np.arange(n_frames - 1, velocity_array.size, n_frames))









a = dict({1: "ok", 2: "bye"})

b = a

a[1] = "sgg"

print(b)