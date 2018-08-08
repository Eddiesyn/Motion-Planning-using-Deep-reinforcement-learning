from pypot.robot import from_config
import pypot.dynamixel
import itertools
import random
import time

vrep = False # if in the vrep env

if vrep:
    close_all_connections()
    poppy = from_vrep('poppy.json', scene = 'experiment.ttt') # /home/eddiesyn/V-REP/
else:
    ports = pypot.dynamixel.get_available_ports()
    #print("availble is :",ports)
    dxl_io_up = pypot.dynamixel.DxlIO(ports[0])
    dxl_io_down = pypot.dynamixel.DxlIO(ports[1])
    lower_io = dxl_io_down
    upper_io = dxl_io_up
    poppy_up = dxl_io_up.scan(range(31,55))
    poppy_down = dxl_io_down.scan(range(11,26))
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

reset_wholebody(10)