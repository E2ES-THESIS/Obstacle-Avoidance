from re import L
import numpy as np
import math as m
import airsim
import cv2 
from PIL import Image

# pos to navigate for the UAV
def moveUAV(client, pos, yaw):
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos[0], pos[1], -2), airsim.to_quaternion(0, 0, yaw)), True) 
    # -5 for africa, -10 for landscape
    
# euclidean distance between two points
def dist_cost(x1, y1, x2, y2):
    cost = m.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))
    return cost

# find vertical FOV from horizontal initialized FOV
def hfov2vfov(hfov, h, w):
    aspect = h/w
    vfov = 2*m.atan(m.tan(hfov/2) * aspect)
    return vfov

# find the bounding box based on the FOV
def bounding_box(h, w, uav_size, hfov, dist_thr):
    vfov = hfov2vfov(hfov,h,w)
    h_i = m.ceil(uav_size[0] * h / (m.tan(hfov/2)*dist_thr*2))
    w_i = m.ceil(uav_size[1] * w / (m.tan(vfov/2)*dist_thr*2))
    return h_i, w_i

# splitting into 3x3 boundary box
def find_box(h_i, w_i, pc, i):

    x_p = pc[i][0]
    y_p = pc[i][1]
    
    # find box number of (x_p,y_p)
    # row 2 - checked first, because it is most preferable to reduce the cost of navigation
    if y_p >= h_i and y_p <= 2*h_i:
        if x_p >= 0 and x_p <= 1*w_i:
            return 'b4'
        if x_p >= w_i and x_p <= 2*w_i:
            return 'b5'
        if x_p >= 2*w_i and x_p <= 3*w_i:
            return 'b6'

    # row 1
    elif y_p >= 0 and y_p <= h_i:
        if x_p >= 0 and x_p <= w_i:
            return 'b1'
        if x_p >= w_i and x_p <= 2*w_i:
            return 'b2'
        if x_p >= 2*w_i and x_p <= 3*w_i:
            return 'b3'

    # row 3
    elif y_p >= 2*h_i and y_p <= 3*h_i:
        if x_p >= 0 and x_p <= 1*w_i:
            return 'b7'
        if x_p >= w_i and x_p <= 2*w_i:
            return 'b8'
        if x_p >= 2*w_i and x_p <= 3*w_i:
            return 'b9'  
    print("helloe")


K = 10
# relationship between the gain and the distance from the UAV state
# gain is basically the tuning parameter
def gain(dist):
    '''
    if dist == 0:
        g = 100
    if dist>0 and dist<=15:
        g = 80
    if dist>15 and dist<=30:
        g = 65
    if dist>30 and dist<=60:
        g = 50
    if dist>60 and dist<100:
        g = 25
    if dist>100:
        g = 5
    return g
    '''
    g = K * dist
    return(g)

# the lowest gain is set to be the weight of the boundary box
def box_weight(var, z, wght):

    g = gain(z)
    
    if var == 'b1': i = 0 
    elif var == 'b2': i = 1
    elif var == 'b3': i = 2
    elif var == 'b4': i = 3
    elif var == 'b5': i = 4
    elif var == 'b6': i = 5
    elif var == 'b7': i = 6
    elif var == 'b8': i = 7
    elif var == 'b9': i = 8

    if wght[i] == 0:
        wght[i] = g
    elif wght[i] > g:
        wght[i] = g
    else: 
      wght[i] = wght[i]
    
    return wght

# cost is the sum of the weight of boundary box, the effort to traverse to a particular boundary box and with the goal position
# the lowest cost is then selected to be the heuristic navigation direction
def cost(X_curr, Y_curr, X_des, Y_des, weight, dist_thr, X_bound, Y_bound):
    
    # cost for current pos and boundary box
    cost1 = dist_cost(X_curr, Y_curr, X_bound, Y_bound)

    # cost from boundary box pos to destination pose
    cost2 = dist_cost(X_bound, Y_bound, X_des, Y_des)

    # cost from the current pos to the goal
    tot_dis_cost = cost1 + cost2

    # cost based on weights
    if weight > dist_thr*K:
        wght_cost = weight*50
    else:
        wght_cost = weight

    total_cost = tot_dis_cost + wght_cost

    return total_cost

# finds the dist and angle for UAV control
def findUAVcontrol(goal, pos):
    
    # euclidean distance btw  both the points
    vec = np.array(goal) - np.array(pos)
    dist = m.sqrt(vec[0]**2 + vec[1]**2)

    # angle btw both the points
    angle = m.atan2(vec[1],vec[0])
    if angle > m.pi:
        angle -= 2*m.pi
    elif angle < -m.pi:
        angle += 2*m.pi
    return dist, angle

# find the waypoint to progress to the goal position
def find_waypoint(depth_cost_var, best_grid_var, step, potential_thr, wp):
    if depth_cost_var > potential_thr:
        if best_grid_var == depth_cost_var[0]:
            print("Left")
            wp[1] = wp[1] - step
            wp[0] = wp[0]
            # yaw = yaw + min(t_ang - yaw, m.radians(limit_yaw))
        elif best_grid_var == depth_cost_var[1]:
            print("Straight")
            wp[1] = wp[1]
            wp[0] = wp[0] + step
            # yaw = 0
        elif best_grid_var == depth_cost_var[2]:
            print("Right")
            wp[1] = wp[1] + step
            wp[0] = wp[0] 
            # yaw = yaw - min(t_ang - yaw, m.radians(limit_yaw))

    return wp

def savePointCloud(image, fileName,w,h):
   f = open(fileName, "w")
   for x in range(image.shape[0]):
     for y in range(image.shape[1]):
        pt = image[x,y]
        if (m.isinf(pt[0]) or m.isnan(pt[0])):
          None # skip it
        else: 
          pt = normalize(pt,w,h)
          f.write("%f %f %f\n" % (pt[0], pt[1], pt[2]-1))
   f.close()

def normalize(pc,w,h):
    w_pc_max = max(pc[0])
    w_pc_min = min(pc[0])
    w_pc = w_pc_max + w_pc_min
    h_pc_max = max(pc[1])
    h_pc_min = min(pc[1]) 
    h_pc  = h_pc_max + h_pc_min

    ratio_x = w/w_pc
    ratio_y = h/h_pc

    pc[0] = pc[0] * m.ceil(ratio_x)
    pc[1] = pc[1] * m.ceil(ratio_y)

    pc[0] = 0 if pc[0] < 0 else w - 1 if pc[0] > w - 1 else int(pc[0])
    pc[1] = 0 if pc[1] < 0 else h - 1 if pc[1] > h - 1 else int(pc[1])

    return pc

def find_depth(client):
    requests = []
    requests.append(airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, pixels_as_float=True, compress=False))
    responses = client.simGetImages(requests)

    depth = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width, responses[0].height)
    depth = np.expand_dims(depth, axis=2)
    depth = depth.squeeze()
    return depth, responses[0].width, responses[0].height

# x,y,z in world coordinates
def convert_depth_3D_vec(x_depth, y_depth, depth, fov):
    h, w = depth.shape
    center_x = w // 2
    center_y = h // 2
    focal_len = w / (2 * np.tan(fov / 2))
    x = depth[y_depth, x_depth]
    y = (x_depth - center_x) * x / focal_len
    z = -1 * (y_depth - center_y) * x / focal_len
    return x,y,z

def logger(client):
    rotor_state = client.getRotorStates()
    print("Rotor State: ", rotor_state)