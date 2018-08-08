from pypot.vrep import from_vrep, close_all_connections
import time
import numpy as np
# For real robot===================
from pypot.robot import from_config
import pypot.dynamixel

import itertools
import random
import json
#==================================
vrep = False # if in the vrep env

if vrep:
    close_all_connections()
    poppy = from_vrep('poppy.json', scene = 'experiment.ttt') # /home/eddiesyn/V-REP/
else:
    ports = pypot.dynamixel.get_available_ports()
    #print("availble is :",ports)
    # with open('poppy.json', 'r') as f:
    # 	config = json.load(f)
    	# .set_angle_limit
    dxl_io_up = pypot.dynamixel.DxlIO(ports[0])
    dxl_io_down = pypot.dynamixel.DxlIO(ports[1])
    lower_io = dxl_io_down
    upper_io = dxl_io_up
    # upper_io.set_angle_limits({41: })
    poppy_up = dxl_io_up.scan(range(31,55))
    poppy_down = dxl_io_down.scan(range(11,26))

    with open('config_g.json', 'r') as f:
        config = json.load(f)
    
    upper_io.set_angle_limit({41: config["motors"]["l_shoulder_y"]["angle_limit"]})
    lower_speed = dict(zip(poppy_down, itertools.repeat(10)))
    upper_speed = dict(zip(poppy_up, itertools.repeat(10)))
    lower_io.set_moving_speed(lower_speed)
    upper_io.set_moving_speed(upper_speed)
    # upper_io.set_pid_gain({41: (7, 4.0, 0.6), 42: (5.0, 2.0, 0.4), 44:(5.0, 2.0, 0.4)}) # 12, 00. 10 {41: (7, 4.0, 0.4), 42: (5.0, 2.0, 0.4)}
    upper_io.set_pid_gain({41: (6, 2.0, 0.2), 42: (5.0, 2.0, 0.2)})
    # speed = 15

i = 0
# Relative functions for real robot===========================
#############################################
############ Initialization #################
#############################################
def reset_wholebody(speed):
    #ports = pypot.dynamixel.get_available_pots()

    if not ports:
        raise IOError('no port found!')

    print('ports found', ports)

    print('connecting on the first available port:', ports[0], ports[1])
    lower_io = dxl_io_down
    upper_io = dxl_io_up

    lower_ids = lower_io.scan(range(11, 26))
    upper_ids = upper_io.scan(range(31, 55))
    if len(lower_ids + upper_ids) != 25:
        raise Exception("some motors can't be scanned")

    lower_speed = dict(zip(lower_ids, itertools.repeat(10)))
    upper_speed = dict(zip(upper_ids, itertools.repeat(10)))
    lower_io.set_moving_speed(lower_speed)
    upper_io.set_moving_speed(upper_speed)

    lower_pose = dict(zip(lower_ids, itertools.repeat(0)))
    upper_pose = dict(zip(upper_ids, itertools.repeat(0)))
    upper_pose[41] = 15
    upper_pose[42] = 15
    upper_pose[44] = 0
    upper_pose[51] = -90
    upper_pose[52] = -80
    upper_pose[54] = 0
    lower_pose[21] = -18

    # upper_io.set_pid_gain({41: (7, 4.0, 0.4), 42: (5.0, 2.0, 0.4)})

    upper_io.set_goal_position(upper_pose)
    lower_io.set_goal_position(lower_pose)
    upper_pose[41] = 90
    upper_pose[42] = 75
    upper_io.set_goal_position(upper_pose)
    time.sleep(20)

    
def reset_RealArm(speed): # speed is supposed to be less than 35
    real_move(41, speed, arm_theta1)
    real_move(42, speed, arm_theta2)
    real_move(44, speed, arm_theta3)
# Move control
def real_move(joint_id, speed, goal_pos): #After test, speed value of 35 is suitable
    
    set_speed = dxl_io_up.set_moving_speed({joint_id: speed})
    goal_pose = dxl_io_up.set_goal_position({joint_id: goal_pos})
    #print('target pose', goal_pose, 'manule speed', speed)
# Get current motor information
def get_info(joint_id):
    pose = dxl_io_up.get_present_position((joint_id, ))
    lim = dxl_io_up.get_angle_limit((joint_id, ))
    ori_speed = dxl_io_up.get_moving_speed((joint_id, ))
    #print('original pose', pose, 'joint bound', lim, 'original speed', ori_speed)    
    return(pose,lim,ori_speed)
#===============================================================
# reset_wholebody(10)

#initial motor angle
if vrep:
	arm_theta1 = 0
	arm_theta2 = 0 
	arm_theta3 = -10
	for m in poppy.motors:
		if m.id == 41:
			motor_41 = m
		if m.id == 42:
			motor_42 = m
		if m.id == 44:
			motor_44 = m
else:
	#reset_wholebody(15)
	arm_theta1 = 90
	arm_theta2 = 75 
	arm_theta3 = 0


def move(motor_, present_position_, goal_position_):  #did not use param present_position_??????????????????????
	if vrep:
		if (motor_ == motor_42) and (motor_41.present_position > -10) and (goal_position_ < 0):
			time.sleep(0.001)
		elif (motor_ == motor_41) and (motor_42.present_position < 0) and (goal_position_ > -10):
			time.sleep(0.001)
		else:
			pass
   #          pose41,lim41,_ = get_info(41)
			motor_.goal_position = goal_position_
			time.sleep(0.5)
	# move motor to goal_position in real robot
	# should consider the constraints between motor41 and motor42 and time for reach goal_pos
	else:
		pass
  #       pose41,lim41,_ = get_info(41)
  #       pose42,lim42,_ = get_info(42)
  #       #if pose41< 0 and goal_position_ > -10:
		# if (motor_ == 42) and (pose41 > -10) and (goal_position_ < 0): #real robot limit may change ??????????????
  #           real_move(motor_, 25, goal_position_)
  #       elif (motor_ == 41) and (pose42 <0) and (goal_position_ > -10):
  #           real_move(motor_, 25, goal_position_)


def motor_move(io, ids, pos):
	io.set_goal_position(dict(zip(ids, pos)))
	present_pos = np.array(list(io.get_present_position(tuple(ids))))
	t = max(np.abs(pos - present_pos) / 10)
	time.sleep(t+0.5)
	# t0 = time.time()
	# while sum(list(io.is_moving(tuple(ids)))) != 0:
	# 	# time.sleep(0.5)
	# 	# print("motor is moving")
	# 	if time.time() - t0 > 5:
	# 		raise Exception("some motors are stuck")
def motor_step(io, ids, pos):
	motor_move(io, ids, pos)
	pos_jetzt = np.array(list(io.get_present_position(tuple(ids))))
	# if np.sum(np.abs(pos_jetzt - pos) > [0.5, 0.5, 1]) > 0:
	# 	motor_move(io, ids, pos)
	# 	pos_jetzt = np.array(list(io.get_present_position(tuple(ids))))
	# if np.sum(np.abs(pos_jetzt - pos) > np.array([0.5, 0.5, 1])) > 0:
	# 	motor_move(io, ids[np.abs(pos_jetzt - pos) > np.array([0.5, 0.5, 1])], 
	# 		np.random.uniform(-1,1,len(ids[np.abs(pos_jetzt - pos) > [0.5, 0.5, 1]])))
	# 	pos_jetzt = np.array(list(io.get_present_position(tuple(ids))))

	return pos_jetzt


def initial_pose():
	move(motor_41, motor_41.present_position, arm_theta1)
	move(motor_42, motor_41.present_position, arm_theta2)
	move(motor_44, motor_44.present_position, arm_theta3)

class Env(object):
 	# action will be angle move between [-1,1]
	state_dim = 9	# theta1 & theta2 & theta3, distance to goal,get_point
	action_dim = 3
	get_point = False
	grab_counter = 0
	vrep_matrix = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
	i = 0
	# point_bound = np.array([[-75, 55], [-19, 70], [-100, -10]])  
	if vrep:
		poppy.reset_simulation()		
		arm1l = 18.5
		arm2l = 21.5  #21.5 + 12
		theta_bound = np.array([[-75, 55], [-19, 70], [-100, -10]])
		action_bound = [-5, 5]
		point_bound = np.array([[-23, -33], [-10,6]]) # YZ
		point_l = 2
		arm_theta1 = 0
		arm_theta2 = 0 
		arm_theta3 = -10
		initial_pose()
		motor_41.compliant = False
		motor_42.compliant = False
		motor_44.compliant = False
		motor_41.torque_limit = 15
		motor_42.torque_limit = 15
		motor_44.torque_limit = 15
		motor_41.moving_speed = 10
		motor_42.moving_speed = 10
		motor_44.moving_speed = 10

	else:
		# table place:(20,0, -32.5), table size:(33.5,31.5,34)
		arm1l = 15
		arm2l = 11.5 + 29.5#11.5 + 40
		theta_bound = np.array([[20, 140], [15, 105], [-100, -10]]) #np.array([[20, 140], [15, 105], [-100, -10]]) ()-120, 155) (-105,110), (-148, 1)
		action_bound = [-10, 10]
		point_bound = np.array([[30, 38], [0, 35]]) #XY
		point_l = 5.
		# reset_RealArm(10)
		motor_id = np.array([41, 42, 44])
		#table_plane = np.array([[51.5, 32],[51.5, -1.5],[20,32],[20, -1.5]]) 
		table_bound = np.array([[20, 51.5], [-1.5, 32]]) # x_bound, y_bound
		epislon = 3
		#reset_RealArm(speed)
		constraints = False


	def __init__(self, point_info = np.array([32.5, 0, -23 ]) ):
		self.arm_info = np.zeros(3)
		self.EE = np.zeros(3)
		self.arm_info[0] = arm_theta1
		self.arm_info[1] = arm_theta2
		self.arm_info[2] = arm_theta3
		if vrep:
			point_info = self.vrep_matrix.dot(point_info)
		print('point_info:', point_info)
		self.point_info = point_info
		self.point_info_init = self.point_info.copy()
		self.EE = self.get_EE(self.arm_info)




	def step(self, action):
		done = False
		action_ = action  #* 180 / np.pi 	# action is np.array(2,)
		if not vrep:
			action_[np.abs(action_) < 1] = 1 # in the real robot should consider the case where action is smaller than 1deg??????? 
		goal_position_1 = np.clip((self.arm_info + action_)[0], self.theta_bound[0, 0] , self.theta_bound[0, 1] )
		goal_position_2 = np.clip((self.arm_info + action_)[1], self.theta_bound[1, 0],  self.theta_bound[1, 1] )
		goal_position_3 = np.clip((self.arm_info + action_)[2], self.theta_bound[2, 0],  self.theta_bound[2, 1] )
		if vrep:
			move(motor_41, motor_41.present_position, goal_position_1)
			move(motor_42, motor_42.present_position, goal_position_2)
			move(motor_44, motor_44.present_position, goal_position_3)

			self.arm_info[0] = motor_41.present_position 
			self.arm_info[1] = motor_42.present_position 
			self.arm_info[2] = motor_44.present_position
		else:
			######### collosion avoidance #########				
			# adaptive time sleep
			pred_arm = np.array([goal_position_1, goal_position_2, goal_position_3])
			pred_EE = self.get_EE(pred_arm)
			print('pred_EE:', pred_EE, ' distance: %.2f'% np.linalg.norm(pred_EE-self.point_info))
			collision = (pred_EE[2] < (-32.5+4) ) & ( (self.table_bound[0,0]-self.epislon) <pred_EE[0]< (self.table_bound[0,1]+self.epislon) ) & ( (self.table_bound[1,0]-self.epislon) < pred_EE[1] < (self.table_bound[1,1]+self.epislon) )
			if collision:
				self.constraints = True
				print('Collisionnnnnnnnnnn:', collision,'type(collision): ', type(collision))
				s = self.get_state()
				r = -.5
				done = False
				print('  self.point_info[0]: %.2f' % self.point_info[0], '  self.point_info[1]: %.2f' % self.point_info[1])	
				print('emmmmmmmm I will collide with tableeeeeeeeeeeeeeee')
				return s, r, done
			if goal_position_2  > 80:
					s = self.get_state()
					r = -1.
					done = False
					print('emmmmmmmm I will collide with myself')
					return s, r, done 

			# elif (pred_EE[2] < (-32.5+5) ) and ( (self.table_bound[0,0]-self.epislon) <pred_EE[0]< (self.table_bound[0,1]+self.epislon) ) and ( (self.table_bound[1,0]-self.epislon) < pred_EE[1] < (self.table_bound[1,1]+self.epislon) ):
			elif collision:
				s = self.get_state()
				r = -.5
				done = False
				print('emmmmmmmm I will collide with tableeeeeeeeeeeeeeee')
				return s, r, done, collosion 

			self.arm_info = motor_step(upper_io, self.motor_id, np.array([goal_position_1, goal_position_2, goal_position_3]))

		self.i += 1
		print('i:', self.i, ' go_pos1: %.2f' % goal_position_1, '   go_pos2: %.2f' % goal_position_2 , '   go_pos3: %.2f' % goal_position_3)
		print('  self.point_info[0]: %.2f' % self.point_info[0], '  self.point_info[1]: %.2f' % self.point_info[1])		
		#print('arm_info: ', self.arm_info[0], self.arm_info[1])
		self.EE = self.get_EE(self.arm_info)
		s = self.get_state()
		# print('self.EE: ', self.EE)
		r = self._r_func(s[6])
		done = self.get_point
		return s, r, done

	def _r_func(self, distance):
		# print('distance: ',distance)
		t = 50
		abs_distance = distance
		r = -abs_distance/200
		# print('point_l : ', self.point_l, 'get_point: ',  self.get_point, ' abs_dis: %.2f'% abs_distance)
		if abs_distance < self.point_l and (not self.get_point):
			print('******************r+1**************************,| grab_counter: ', self.grab_counter)
			r += 1.
			self.grab_counter += 1

			if self.grab_counter > t:
				r += 10.	
				self.get_point = True
				print('******************r+10**************************')
		elif abs_distance > self.point_l:
			self.grab_counter = 0
			self.get_point = False
		return r

	def reset(self):
		self.get_point = False
		self.constraints = False
		self.i = 0
		if vrep:
			poppy.reset_simulation()
			self.point_info[1] = np.clip(self.point_bound[0, 0] + 10*np.random.random(), self.point_bound[0,0], self.point_bound[0,1])
			self.point_info[2] = np.clip(self.point_bound[1, 0] + 16*np.random.random(), self.point_bound[1,0], self.point_bound[1,1])
			self.point_info[0] = 23
		# initial random position
			self.arm_info[0] = np.clip(self.theta_bound[0, 0] + 130*np.random.random(), -30, 30)
			self.arm_info[1] = np.clip(self.theta_bound[1, 0] + 89*np.random.random(), 0, 10)
			self.arm_info[2] = np.clip(self.theta_bound[2, 0] + 90*np.random.random(), -40, -10)
			move(motor_41, motor_41.present_position, self.arm_info[0])
			move(motor_42, motor_42.present_position, self.arm_info[1])
			move(motor_44, motor_44.present_position, self.arm_info[2])
			self.EE = self.get_EE(self.arm_info)
			print('initial random point: ', self.point_info)
			print('initial random state: ', self.arm_info)
			self.arm_info[0] = motor_41.present_position  # initial state should be observation ???????
			self.arm_info[1] = motor_42.present_position		
			self.arm_info[2] = motor_44.present_position
		else: 
			# self.point_info[0] = np.clip(self.point_bound[0, 0] + 8*np.random.random(), self.point_bound[0,0], self.point_bound[0,1])
			# self.point_info[1] = np.clip(self.point_bound[1, 0] + 35*np.random.random(), self.point_bound[1,0], self.point_bound[1,1])

			#[array([ 37.11031882,   5.46191293, -25.        ]), array([ 30.25567756,   8.39831027, -25.        ])]
			self.point_info[0] = 33.3#.copy()
			self.point_info[1] = 27.4#.copy()			
			self.point_info[2] = -25 #[ 35.80688059,   6.03394421, -25.        ]
					# initial fixed position
			self.arm_info[0] = arm_theta1#np.clip(self.theta_bound[0, 0] + 130*np.random.random(), -30, 30)
			self.arm_info[1] = arm_theta2#np.clip(self.theta_bound[1, 0] + 89*np.random.random(), 0, 10)
			self.arm_info[2] = arm_theta3#np.clip(self.theta_bound[2, 0] + 90*np.random.random(), -40, -10)
			self.EE = self.get_EE(self.arm_info)
			print('initial random point: ', self.point_info)
			print('initial random state: ', self.arm_info)
			self.arm_info = motor_step(upper_io, self.motor_id, np.array([arm_theta1, 15, arm_theta3]))
			self.arm_info = motor_step(upper_io, self.motor_id, np.array([arm_theta1, arm_theta2, arm_theta3]))
			print('initial real state: ', self.arm_info)

        # if vrep:
		# 	self.point_info = self.vrep_matrix.dot(self.point_info)
		self.EE = self.get_EE(self.arm_info)

		print(" \n -----------------reset--------------- \n")
		return self.get_state()




	def get_state(self):
		state_ = np.zeros(9)
		state_[:3] = self.arm_info
		# print('self.EE.shape: ', len(self.EE))
		state_[3] = self.EE[0] - self.arm_info[0] 
		state_[4] = self.EE[1] - self.arm_info[1]
		state_[5] = self.EE[2] - self.arm_info[2]
		state_[6] = np.linalg.norm(self.point_info - self.EE)
		state_[7] = 1 if self.grab_counter > 0 else 0
		state_[8] = 1 if self.constraints == True else 0
		return state_ 


	def rotation_matrix(self, theta, axis):
		R = np.zeros((4,4))
		theta_ = - theta*np.pi/180
		R[3, 3] = 1
		if axis == 0:   # axis x
			R[0, 0] = 1
			R[1, 1] = np.cos(theta_)
			R[1, 2] = -np.sin(theta_)
			R[2, 1] = np.sin(theta_)
			R[2, 2] = np.cos(theta_)
		elif axis == 1:   # axis y
			R[0, 0] = np.cos(theta_)
			R[0, 2] = -np.sin(theta_)
			R[1, 1] = 1
			R[2, 0] = np.sin(theta_)
			R[2, 2] = np.cos(theta_)
		elif axis == 2:
			R[0, 0] = np.cos(theta_)
			R[0, 1] = -np.sin(theta_)
			R[1, 0] = np.sin(theta_)
			R[1, 1] = np.sin(theta_)
			R[2, 2] = 1

		return R

	def translation_matrix(self, length, axis):
		T = np.zeros((4,4))
		T[:3,:3] = np.eye(3)
		T[3, 3] = 1
		if axis == 0:
			T[0, 3] = length
		elif axis == 1:
			T[1, 3] = length
		elif axis == 2:
			T[2, 3] = length

		return T

	def get_EE(self, arm_info):
		# EE = np.zeros(2)
		# arm_info_ = [arm_info[0] * np.pi /180, arm_info[1]* np.pi /180]
		# #print('arm_info_radiant: ', arm_info_)
		# EE[0] = -self.arm2l*np.sin(np.sum(arm_info_)) - self.arm1l*np.sin(arm_info_[0])
		# EE[1] = self.arm2l*np.cos(np.sum(arm_info_)) + self.arm1l*np.cos(arm_info_[0])
		#print('self.arm_info[0]: ', self.arm_info[0])
		if vrep:
			R1 = self.rotation_matrix(self.arm_info[0], 1) # theta1
			R2 = self.rotation_matrix(self.arm_info[1], 0) # theta2 			
		else:
			R1 = self.rotation_matrix(self.arm_info[0]-90, 1) # theta1
			R2 = self.rotation_matrix(self.arm_info[1]-90, 0) # theta2 

		R3 = self.rotation_matrix(self.arm_info[2], 1) # theta3
		T1 = self.translation_matrix(-self.arm1l, 2) 
		T2 = self.translation_matrix(-self.arm2l, 2)


		EE_full = R1.dot(R2.dot(T1.dot(R3.dot(T2))))
		EE = EE_full[:3, -1]
		EE += np.array([0, 0, -5])

		# if (30<self.point_info[0]< 40) & (20< self.point_info[1]<30):
		# 	EE += np.array([1, 1, -4.8])
		# else:
		# 	EE += np.array([0, 0, -5])

		if vrep:
			EE = self.vrep_matrix.dot(EE)

		return EE
