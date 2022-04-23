import math as m
import cv2
import airsim
import numpy as np
import time

import helper

# UAV setup initialization
height = 10 # hold height 
velocity = 10
thr = 0.3
step = 0.7 # constant - step
start = [0,0,0] # start pos
goal = [100,0,0] # des pos 
uav_size = [45,45] # size, h x w (cm)

# image grid parameters
cols = 3 # columns split by frame
rows = 3 # rows split by frame
val = 0 # index of grid with minimum cost 
#w1 = w2 = w3 = w4 = w5 = w6 = w7 = w8 = w9 = 0 # weights

# define parameters and thresholds
fovh = m.radians(90) # simGetCurrentFieldOfView(camera_name, vehicle_name='', external=False) 
yaw = 0
limit_yaw = 5
dist_thr = 0.2 # minimum safety threshold distance (cm)
prev_frame_time = 0
i = 1
filename = "points.asc"

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

# set the horizontal focal length for simulation
#hfov = client.simSetCameraFov('stereo_cam', fovh, external=False)
hfov = fovh

# start timer for fps estimation
new_frame_time = time.time()

print (" ########################### Check UAV state ###########################")
# current pose
pos = client.simGetVehiclePose()

# set start pose to (0,0,0)
if pos==start:
    helper.moveUAV(client, start, yaw)
print (" ######################## UAV in start position ########################")

print (" ################################# ARM #################################")
client.armDisarm(True) # False to disarm

print (" ############################### TAKEOFF ###############################")
client.takeoffAsync(timeout_sec=20) # Takeoff to 3m from current pose

print (" ###################### Avoidance Algorithm Start ######################")

def main():

    global prev_frame_time, i
    ### input depth map from camera using a projection ray that hits each pixel
    rawImage = client.simGetImage("0", airsim.ImageType.DepthPerspective)
    img = cv2.imdecode(np.frombuffer(rawImage, np.uint8) , cv2.IMREAD_UNCHANGED)

    h, w, g = img.shape
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # bounding box based on the uav size and fov
    #h, w = helper.bounding_box(h, w, uav_size, hfov, dist_thr)

    # size of each grid [3 x 3]
    h_i = h / 3
    w_i = w / 3

    img1 = img[0:int(h_i), 0:int(w_i)]
    img2 = img[0:int(h_i), int(w_i):2*int(w_i)]
    img3 = img[0:int(h_i), 2*int(w_i):3*int(w_i)]
    img4 = img[int(h_i):2*int(h_i), 0:int(w_i)]
    img5 = img[int(h_i):2*int(h_i), int(w_i):2*int(w_i)]
    img6 = img[int(h_i):2*int(h_i), 2*int(w_i):3*int(w_i)]
    img7 = img[2*int(h_i):3*int(h_i), 0:int(w_i)]
    img8 = img[2*int(h_i):3*int(h_i), int(w_i):2*int(w_i)]
    img9 = img[2*int(h_i):3*int(h_i), 2*int(w_i):3*int(w_i)]

    ### find favourable grid using weights
    depth_cost = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    depth_cost[0] = np.sum(img1)
    depth_cost[1] = np.sum(img2)
    depth_cost[2] = np.sum(img3)
    depth_cost[3] = np.sum(img4)
    depth_cost[4] = np.sum(img5)
    depth_cost[5] = np.sum(img6)
    depth_cost[6] = np.sum(img7)
    depth_cost[7] = np.sum(img8)
    depth_cost[8] = np.sum(img9)

    depth_cost_var = [0,0,0]
    
    best_grid = max(depth_cost)
    depth_cost_var[0] = (depth_cost[0] + depth_cost[3] + depth_cost[6])/(3*best_grid)
    depth_cost_var[1] = (depth_cost[1] + depth_cost[4] + depth_cost[7])/(3*best_grid)
    depth_cost_var[2] = (depth_cost[2] + depth_cost[5] + depth_cost[8])/(3*best_grid)

    best_grid_var = max(depth_cost_var)

    ###############################################

    # find navigation waypoint in world frame
    wp = start

    # waypoints for different actions
    l_wp = [ wp[0], wp[1]-step ]
    s_wp = [ wp[0]+step, wp[1] ]
    r_wp = [ wp[0], wp[1]+step ]

    pose = client.simGetVehiclePose()

    # distance cost
    dist_cost_var = [0, 0, 0]
    dist_cost_var[0] = helper.dist_cost(pose.position.x_val, pose.position.y_val, l_wp[0], l_wp[1]) + helper.dist_cost(l_wp[0], l_wp[1], goal[0], goal[1])
    dist_cost_var[1] = helper.dist_cost(pose.position.x_val, pose.position.y_val, s_wp[0], s_wp[1]) + helper.dist_cost(s_wp[0], s_wp[1], goal[0], goal[1])
    dist_cost_var[2] = helper.dist_cost(pose.position.x_val, pose.position.y_val, r_wp[0], r_wp[1]) + helper.dist_cost(r_wp[0], r_wp[1], goal[0], goal[1])

    # normalizing the distance cost between (0, 1)
    max_dist_cost = max(dist_cost_var)
    dist_cost_var[0] = dist_cost_var[0] / max_dist_cost
    dist_cost_var[1] = dist_cost_var[1] / max_dist_cost
    dist_cost_var[2] = dist_cost_var[2] / max_dist_cost
    print(dist_cost_var)

    best_grid_var = max(dist_cost_var)
    
    #print (" #################### Cost Estimated ####################")

    # best_grid for 3x3 and best_grid_var for 3x1
    
    #print(best_grid_var, " in ", depth_cost_var)
    #print(best_grid_var, " in ", dist_cost_var)
    #print (" ################ Favourable Grids Found ################")

    #print (" ######################## Navigation Initiated #######################")

    #print (" ################### NavWaypoint Found ##################")

    # finds the target distance and angle for UAV navigation
    
    curr_state = pose.position.x_val, pose.position.y_val, pose.position.z_val
    t_dist, t_ang = helper.findUAVcontrol(goal, curr_state)

    # estimates the yaw and position angles from the helper scripts
    yaw = 0

    cost_var = [0, 0, 0]
    cost_var[0] = (dist_cost_var[0] + depth_cost_var[0]*5)/6
    cost_var[1] = (dist_cost_var[1] + depth_cost_var[1]*5)/6
    cost_var[2] = (dist_cost_var[2] + depth_cost_var[2]*5)/6
    
    best_grid_tot = max(cost_var)

    if cost_var[1] > thr:
        print("Straight")
        wp[1] = wp[1]
        wp[0] = wp[0] + step        
    else:
        if best_grid_tot == cost_var[1]:
            print("Straight")
            wp[1] = wp[1]
            wp[0] = wp[0] + step
        elif best_grid_tot == cost_var[0]:
            print("Left")
            wp[1] = wp[1] - step
            wp[0] = wp[0]
        elif best_grid_tot == cost_var[2]:
            print("Right")
            wp[1] = wp[1] + step
            wp[0] = wp[0] 
    
    #client.moveToPositionAsync(wp[0], wp[1], height, velocity, timeout_sec=3e+38, yaw_mode=False).join()
    
    distance = helper.dist_cost(pose.position.x_val, pose.position.y_val, start[0], start[1])
    # move the UAV to the estimated control positions
    helper.moveUAV(client, wp, yaw)

    print (" ################### Waypoint ", i, " Reached ##################")
    #print(distance)
    i= i+1
    #moveByRollPitchYawThrottleAsync(roll, pitch, yaw, throttle, duration, vehicle_name='')
    ##moveToPositionAsync(x, y, z, velocity, timeout_sec=3e+38, yaw_mode=True)
    return (t_dist, pos)

if __name__ == "__main__":

    while True:
        t_dist, pose = main()
        if (t_dist < 0.3):

            if pose.position.z_val == 0:
                # Disarm the UAV
                client.armDisarm(False)
            else:
                # Land and disarm
                client.landAsync(timeout_sec=10)
                client.armDisarm(False)

            print (" ########################### End of Mission ##########################")
            airsim.wait_key('Press any key to continue')
            # break
