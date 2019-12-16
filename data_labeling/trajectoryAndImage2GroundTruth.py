# disabling contracts for speed
import contracts
import yaml
import geometry as geo
import numpy as np
from duckietown_world.world_duckietown.tile import get_lane_poses
import duckietown_world as dw
import geometry as g
import pandas as pd


contracts.disable_all()

# Load Duckietown map
m = dw.load_map('robotarium2')

# Folders that should be processed
folderNames = ['rec1_bright_lf2_clw','rec2_bright_dark_clw', 'rec3_bright_ccw']

# Calculates relative pose between two poses
def relative_pose(q0, q1):
	return g.SE2.multiply(g.SE2.inverse(q0), q1)

# Interpolates position based on timestamp between two poses
def interpolate(q0, q1, alpha):
	q1_from_q0 = relative_pose(q0, q1)
	vel = g.SE2.algebra_from_group(q1_from_q0)
	rel = g.SE2.group_from_algebra(vel * alpha)
	q = g.SE2.multiply(q0, rel)
	return q


# Calculates distance to middle lane and heading
def calculatePose(qPose):
	success = False
	transforms = dw.SE2Transform.from_SE2(qPose)

	# Filters these tiles out. We don't want them in training set
	notWantedTiles = ['3way_left', '4way', 'asphalt']

	pose_object = transforms
	lanePoses = list(get_lane_poses(m, pose_object.as_SE2()))

	try:
		tile = list(lanePoses[0].tile.children.keys())[0]

		if tile in notWantedTiles:

			return success, 0, 0, str(tile)

	except Exception as e:
		print(e)

	if len(lanePoses) == 0:
		return success, 0, 0, 'asphalt'

	else:

		distance_from_center = lanePoses[0].lane_pose.distance_from_center / 2
		rel_heading = lanePoses[0].lane_pose.relative_heading
		tile = list(lanePoses[0].tile.children.keys())[0]
		success = True
		return success, distance_from_center, rel_heading, str(tile)


# Open trajectory yaml files in each folder
for folderNamesSingle in folderNames:
	finalArrayWithoutPose = []
	finalArrayPoses = []
	realTimestamps = []
	seqs2 = []
	final_trajectory = []
	timeStart = []

	for trajectoryFileNumber in range(0,500):
		trajectoryFile = folderNamesSingle + '_' + str(trajectoryFileNumber) + '.yaml'

		print('Munging ' + str(trajectoryFile))

		# Path where all yaml files are
		pathToYaml = 'trajectoryFiles/' + str(folderNamesSingle) + '/' + str(trajectoryFile)

		try:
			with open(pathToYaml, 'r') as stream:
				try:
					data = yaml.safe_load(stream)

				except yaml.YAMLError as exc:
					print(exc)

		except Exception as e:
			if trajectoryFileNumber != 0:
				break

			else:
				continue

		# Read in yaml information and store in arrays
		timestart = data['begin_time_stamp']
		data_points = len(data['trajectory_data'])


		x = np.zeros((data_points,))
		y= np.zeros((data_points,))

		R = np.zeros((3,3, data_points))
		phi = np.zeros((3, data_points))

		dx = 999.999*np.ones((data_points, ))

		dy = 999.999*np.ones((data_points, ))

		dr = 999.999*np.ones((data_points, ))
		dphi = 999.999*np.ones((data_points, ))


		for idx, [time, traj] in enumerate(data['trajectory_data'].items()):

			x[idx] = np.array(traj[0])

			y[idx] = np.array(traj[1])

			R[:,:,idx] = np.reshape(np.array(traj[3:]), (3,3))

			phi[:,idx] = np.array([np.arctan2(-R[1,2,idx],R[2,2,idx]),
								   np.arctan2(R[0,2,idx],np.sqrt(R[0,0,idx]**2 + R[0,1,idx]**2)),
								   np.arctan2(-R[0,1,idx], R[0,0,idx])])

			realTimestamps.append(float(time) + float(timestart))
			timeStart.append(timestart)
			z = phi[2,idx]
			points = np.array([x[idx], y[idx]])
			final_trajectory.append([points, z])

	for entry in range(0, len(final_trajectory)):

		x =  (final_trajectory[entry][0][0] )
		y = final_trajectory[entry][0][1]
		alpha = final_trajectory[entry][1]
		q5 = geo.SE2_from_translation_angle([x,y],alpha)
		seqs2.append(q5)

	# Path to file with watchtower image timesteps
	path = 'trajectoryFiles/' + str(folderNamesSingle) + '/image_timesteps.csv'
	imagesDFcolumns = ['ImageName', 'Unused','Seconds', 'Nanoseconds']
	imagesDF = pd.read_csv(path)
	timeStampImagesSecondsArray = imagesDF.iloc[:,2]
	timeStampImagesNanoSecArray = imagesDF.iloc[:,3]
	imageNumber = imagesDF.iloc[:,0]
	entryNumberTrajectory = 0


	# For each watchtower image timesteps, calculate the pose with the trajectory data
	for entry in range(0, len(timeStampImagesSecondsArray)-1):
		if entryNumberTrajectory + 2 >= len(realTimestamps):
			break
		timeStampImagesSeconds = imagesDF.iloc[entry, 2]
		timeStampImagesNanoSec = imagesDF.iloc[entry,3] * (10**(-9))
		imageNumber = imagesDF.iloc[entry,0]

		# Check if exact time is available in both dataframes
		timeStampImages = float(timeStampImagesSeconds) + float(timeStampImagesNanoSec)
		timeStampTrajectory = float(realTimestamps[entryNumberTrajectory])
		timeStampTrajectoryAfter = float(realTimestamps[entryNumberTrajectory + 1])

		if timeStampImages == timeStampTrajectory:
			q2 = seqs2[entryNumberTrajectory]
			finalArrayPoses.append(q2)
			success, centerDistance, relativeHeading, tile = calculatePose(q2)

			if success is False:
				continue

			finalArrayWithoutPose.append([imageNumber, timeStampImages, centerDistance, relativeHeading, tile])

		elif timeStampImages < timeStampTrajectory:
			if entryNumberTrajectory == 0:
				continue

			while timeStampImages < timeStampTrajectory:

				if timeStampTrajectory > timeStampImages > (float(realTimestamps[entryNumberTrajectory - 1])):
					timeStampTrajectory = float(realTimestamps[entryNumberTrajectory-1])
					timeStampTrajectoryAfter = float(realTimestamps[entryNumberTrajectory])
					break

				elif timeStampTrajectory > timeStampImages > (float(realTimestamps[entryNumberTrajectory - 2])):
					timeStampTrajectory = float(realTimestamps[entryNumberTrajectory - 2])
					timeStampTrajectoryAfter = float(realTimestamps[entryNumberTrajectory])
					break

				else:
					entryNumberTrajectory -= 1
					timeStampTrajectory = float(realTimestamps[entryNumberTrajectory])
					timeStampTrajectoryAfter = float(realTimestamps[entryNumberTrajectory + 1])

			if timeStampTrajectory < timeStampImages < timeStampTrajectoryAfter:
				timeStampTrajectory = float(realTimestamps[entryNumberTrajectory])
				timeStampTrajectoryAfter = float(realTimestamps[entryNumberTrajectory + 1])
				timeStamp1 = timeStampTrajectory
				timeStampWanted = timeStampImages
				timeStamp2 = timeStampTrajectoryAfter
				param = (timeStampWanted- timeStamp1) / (timeStamp2 - timeStamp1)
				q2 = seqs2[entryNumberTrajectory+1]
				q1 = seqs2[entryNumberTrajectory]
				qInter = interpolate(q1, q2, param)
				success, centerDistance, relativeHeading, tile = calculatePose(qInter)

				if success is False:
					continue

				finalArrayPoses.append(qInter)
				finalArrayWithoutPose.append([imageNumber, timeStampImages, centerDistance, relativeHeading, tile])

				continue

		elif timeStampImages > timeStampTrajectory:

			if timeStampImages < timeStampTrajectoryAfter:
				timeStampTrajectory = float(realTimestamps[entryNumberTrajectory])
				timeStampTrajectoryAfter = float(realTimestamps[entryNumberTrajectory + 1])
				timeStamp1 = timeStampTrajectory
				timeStampWanted = timeStampImages
				timeStamp2 = timeStampTrajectoryAfter
				param = (timeStampWanted- timeStamp1)/ (timeStamp2 - timeStamp1)
				q2 = seqs2[entryNumberTrajectory+1]
				q1 = seqs2[entryNumberTrajectory]
				qInter = interpolate(q1, q2, param)
				success, centerDistance, relativeHeading, tile = calculatePose(qInter)

				if success is False:
					continue

				finalArrayPoses.append(qInter)
				finalArrayWithoutPose.append([imageNumber, timeStampImages, centerDistance, relativeHeading, tile])

			else:

				while timeStampImages > timeStampTrajectoryAfter:

					if entryNumberTrajectory + 2 >= len(realTimestamps):
						break

					if timeStampImages > timeStampTrajectory and timeStampImages > timeStampTrajectoryAfter:
						timeStampTrajectoryAfter = float(realTimestamps[entryNumberTrajectory + 2])

						if timeStampTrajectory < timeStampImages < timeStampTrajectoryAfter:
							break

						else:
							entryNumberTrajectory += 1

							if timeStampTrajectoryAfter < timeStampImages:
								timeStampTrajectory =  float(realTimestamps[entryNumberTrajectory])

							timeStampTrajectoryAfter = float(realTimestamps[entryNumberTrajectory + 1])

				# if entryNumberTrajectory > 500:
				# 	timeStampTrajectory = float(realTimestamps[entryNumberTrajectory])

				timeStampTrajectory = float(realTimestamps[entryNumberTrajectory])
				timeStamp1 = timeStampTrajectory
				timeStampWanted = timeStampImages
				timeStamp2 = timeStampTrajectoryAfter
				param = (timeStampWanted- timeStamp1) / (timeStamp2 - timeStamp1)
				q2 = seqs2[entryNumberTrajectory+1]
				q1 = seqs2[entryNumberTrajectory]
				qInter = interpolate(q1, q2, param)
				success, centerDistance, relativeHeading, tile = calculatePose(qInter)

				if success is False:
					continue

				finalArrayPoses.append(qInter)
				finalArrayWithoutPose.append([imageNumber, timeStampImages, centerDistance, relativeHeading, tile])
		else:
			print('unhandled')

	finalArrayWithoutPose = np.array(finalArrayWithoutPose)
	ArrayCol = ['ImageNumber',' timeStamp','centerDistance', 'relativeHeading', 'Tile']
	finalArrayWithPose = pd.DataFrame(finalArrayWithoutPose, columns=ArrayCol)
	fileName = 'trajectoryFiles/oliFiles/' + str(folderNamesSingle) + '/output_oli.csv'
	finalArrayWithPose.to_csv(fileName, index=False)
	print('saved as: ' + str(fileName))
