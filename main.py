import math as m
import cv2
import airsim
import numpy as np
import time

import helper

# UAV setup initialization
height = 10 # hold height 
velocity = 20 # speed of the UAV
start = [0,0,0] # start pos
goal = [5,15,0] # des pos 
uav_size = [45,45] # size, h x w (cm)
step = 0.5 # constant - step

# image grid parameters
cols = 3 # columns split by frame
rows = 3 # rows split by frame
val = 0 # index of grid with minimum cost 
#w1 = w2 = w3 = w4 = w5 = w6 = w7 = w8 = w9 = 0 # weights

# define parameters and thresholds
fovh = m.radians(90) # simGetCurrentFieldOfView(camera_name, vehicle_name='', external=False) 
yaw = 0
limit_yaw = 5
step = 0.1 # waypoint progression step
dist_thr = 0.2 # minimum safety threshold distance (cm)
prev_frame_time = 0 # to find fps
new_frame_time = 0 # to find fps
i = 1 # Waypoint ID
filename = "points.asc" # save point cloud
font = cv2.FONT_HERSHEY_SIMPLEX

# pixel potential for image processing and depth control
ran_depth = [0.2, 15] #
potential_thr = 500000 # threshold based on image size
fac = 10 # factor for intensity change
max_int = 255.0 # depends on dtype of image data

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
client.takeoffAsync(timeout_sec=10) # Takeoff to 3m from current pose

print (" ###################### Avoidance Algorithm Start ######################")

def main():

    global prev_frame_time, i 
    ### input depth map from camera using a projection ray that hits each pixel
    rawImage = client.simGetImage("0", airsim.ImageType.DepthPerspective)
    img = cv2.imdecode(np.frombuffer(rawImage, np.uint8) , cv2.IMREAD_UNCHANGED)
    frame = img
    # decrease the intensity of the obtained frames
    img_upd = (max_int)*(img/max_int)**fac
    img_upd = img_upd.astype(np.uint8)

    # img = img_upd

    # for visualization
    img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET) 

    h, w, g = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    #################### Cost Estimated ####################

    # best_grid for 3x3 and best_grid_var for 3x1
    best_grid = max(depth_cost)
    print(best_grid)
    print(depth_cost)
    ################ Favourable Grids Found ################

    ######################## Navigation Initiated #######################

    # find navigation waypoint in world frame
    wp = start

    # finds the target distance and angle for UAV navigation
    t_dist, t_ang = helper.findUAVcontrol(goal, wp)

    # estimates the yaw and position angles from the helper scripts
    yaw = 0
    if best_grid == depth_cost[4]:
        print("Left")
        wp[1] = wp[1] - step
        wp[0] = wp[0]
    elif best_grid == depth_cost[5]:
        print("Straight")
        wp[1] = wp[1]
        wp[0] = wp[0] + step
    elif best_grid == depth_cost[6]:
        print("Right")
        wp[1] = wp[1] + step
        wp[0] = wp[0] 
    elif best_grid == depth_cost[2]:
        print("Up")
        wp[1] = wp[1]
        wp[0] = wp[0]
        wp[2] = wp[2] + step   
    elif best_grid == depth_cost[2]:
        print("Down")
        wp[1] = wp[1]
        wp[0] = wp[0]
        wp[2] = wp[2] - step   
    if best_grid == depth_cost[4]:
        print("Top Left")
        wp[1] = wp[1] - step
        wp[0] = wp[0]
        wp[2] = wp[2] + step
    elif best_grid == depth_cost[6]:
        print("Top Right")
        wp[1] = wp[1] + step
        wp[0] = wp[0] 
        wp[2] = wp[2] + step
    if best_grid == depth_cost[4]:
        print("Bottom Left")
        wp[1] = wp[1] - step
        wp[0] = wp[0]
        wp[2] = wp[2] - step
    elif best_grid == depth_cost[6]:
        print("Bottom Right")
        wp[1] = wp[1] + step
        wp[0] = wp[0] 
        wp[2] = wp[2] - step
    
    # client.moveToPositionAsync(wp[0], wp[1], height, velocity, timeout_sec=3e+38, yaw_mode=False).join()
    
    # moveByRollPitchYawThrottleAsync(roll, pitch, yaw, throttle, duration, vehicle_name='')
    # #moveToPositionAsync(x, y, z, velocity, timeout_sec=3e+38, yaw_mode=True)

    # move the UAV to the estimated control positions
    helper.moveUAV(client, wp, yaw)

    print(" ################### Waypoint ", i," Reached ##################")
    i = i + 1

    ######################## Visualization #######################

    # finding the fps 
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    # converting the fps into integer
    fps = int(fps)
    # converting the fps to string so that we can display it on frame
    fps = str(fps)
    cv2.putText(img_color, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # divide the map into 9 grids in 3x3 format
    pxstep = int(img_color.shape[1]/3)
    pystep = int(img_color.shape[0]/3)
    gx = pxstep
    gy = pystep
    while gx < img.shape[1]:
        cv2.line(img_color, (gx, 0), (gx, img.shape[0]), color=(0, 0, 0), thickness=2)
        gx += pxstep
    while gy < img.shape[0]:
        cv2.line(img_color, (0, gy), (img.shape[1], gy), color=(0, 0, 0), thickness=2)
        gy += pystep

    # shade the navigational grid red based on estimated cost
    #mark_x, mark_y, mark_x_end, mark_y_end = helper.mark()
    #visualize = cv2.rectangle(img_color, (mark_x, mark_y), (mark_x_end, mark_y_end), (100, 100, 100), thickness=-1)

    conc_img = cv2.vconcat([frame, img, img_upd, img_color])
    # display the image with the cost map
    cv2.imshow("Cost Map", conc_img)
    cv2.waitKey(1)

    return (t_dist, client)

if __name__ == "__main__":

    while True:
        t_dist, client = main()
        pos = client.simGetVehiclePose()

        if (t_dist < 0.5):
            
            print("\n")
            print (" ########################### End of Mission ##########################")
            if pos[2] == 0:
                # Disarm the UAV
                client.armDisarm(False)
            else:
                # Land and disarm
                client.landAsync(timeout_sec=10)
                client.armDisarm(False)
            
            print("\n")

            airsim.wait_key('Press any key to continue...')
            break
