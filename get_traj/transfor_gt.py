from cupshelpers import Printer
import numpy as np
import math
from matplotlib import pyplot as plt 
from mpl_toolkits import mplot3d

from scipy.spatial.transform import Rotation as R 
# from tf.transformations import quaternion_multiply, quaternion_inverse  


os0_tf = np.array([[0.707107,0.707107,0.0,0.0],[ -0.707107,0.707107,0.0,0.0],[0.0,0.0,1.0,0.0],[ 0.0,0.0,0.0,1.0]])  

os1_tf = np.array([[0.728751,-0.684722,0.00880163,-0.0118653],[  0.68459,0.728792,0.0140687,-0.568633],
                [ -0.0160477, -0.00422711,0.999862 ,-0.00518761],[ 0.0,0.0,0.0,1.0]])   

velo_tf = np.array([[ 0.999046,-0.0385039,0.0206179,-0.109294],[ 0.0383028,0.999216,0.010062,-0.295213],
                    [ -0.0209891 ,-0.00926264,0.999737,0.090953],[ 0.0,0.0,0.0,1.0]])  

hori_tf = np.array([[0.999472,0.0231217 ,-0.0228489,-0.114303],[-0.0226391,0.99952,0.0211604,-0.312768],
                    [ 0.0233272,-0.020632,0.999515,-0.102705],[ 0.0,0.0,0.0,1.0]])  
  
avia_tf = np.array([[ 0.999495,0.000987174,-0.0317686,-0.106596],[ -0.000559863,0.999909,0.0134568,-0.234419],
                    [0.031779,-0.0134322,  0.999405, -0.131481],[ 0.0,0.0,0.0,1.0]])       

 
def read_data(folder_path, loam_fname, optk_fname):
     
    loam_pose = np.genfromtxt(folder_path+loam_fname, delimiter=' ')
    optk_pose = np.genfromtxt(folder_path+optk_fname, delimiter=' ')
    print(loam_pose[0,0]) 
    print("==>>> Reading LOAM  {} | shape {}".format( loam_fname, loam_pose.shape))
    # print("==>>> Reading Optk  {} | shape {}".format( optk_fname, optk_pose.shape))
    return loam_pose, optk_pose

_EPS = np.finfo(float).eps * 4.0


def read_data_optk(folder_path, optk_fname):
     
    optk_pose = np.genfromtxt(folder_path+optk_fname, delimiter=' ')

    return  optk_pose

_EPS = np.finfo(float).eps * 4.0

def quaternion_to_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1])
        ), dtype=np.float64)

 

def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = np.empty((4, ), dtype=np.float64) 
    # M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    M = np.array([  [matrix[0,0], matrix[0,1], matrix[0,2], 0], 
                    [matrix[1,0], matrix[1,1], matrix[1,2], 0], 
                    [matrix[2,0], matrix[2,1], matrix[2,2], 0], 
                    [0.0,         0.0,         0.0,        1.0]])
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def quaternion_conjugate(quaternion):
    """Return conjugate of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_conjugate(q0)
    >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
    True

    """
    return np.array((-quaternion[0], -quaternion[1],
                        -quaternion[2], quaternion[3]), dtype=np.float64)

def quaternion_inverse(quaternion):
    """Return inverse of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_inverse(q0)
    >>> np.allclose(quaternion_multiply(q0, q1), [0, 0, 0, 1])
    True

    """
    return quaternion_conjugate(quaternion) / np.dot(quaternion, quaternion)

def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.

    >>> q = quaternion_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> numpy.allclose(q, [-44, -14, 48, 28])
    True

    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array((
         x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
         x1*y0 - y1*x0 + z1*w0 + w1*z0,
        -x1*x0 - y1*y0 - z1*z0 + w1*w0), dtype=np.float64)


def pose_transform(loam_data, tf_matrix):
    tf_rot   = np.array([ [tf_matrix[0,0], tf_matrix[0,1], tf_matrix[0,2]], 
                        [tf_matrix[1,0], tf_matrix[1,1], tf_matrix[1,2]], 
                        [tf_matrix[2,0], tf_matrix[2,1], tf_matrix[2,2]] ])
    tf_trans =  np.array( [tf_matrix[0,3], tf_matrix[1,3], tf_matrix[2, 3]])
    # print("ROT: : ", tf_rot)
    # print("TRANS: ", tf_trans)

    pos_x = loam_data[:,1]
    pos_y = loam_data[:,2]
    pos_z = loam_data[:,3]

    pos_qx = loam_data[:,4]
    pos_qy = loam_data[:,5]
    pos_qz = loam_data[:,6]
    pos_qw = loam_data[:,7]

    loam_tf =   np.empty(shape=(0,8))
    for i in range(len(pos_x)):
        #  Transform local coordinate to global coordinate
        pos_global =   tf_rot @ np.transpose( np.array([pos_x[i], pos_y[i], pos_z[i]])) + tf_trans 
 
        curr_quat = np.array([ pos_qx[i], pos_qy[i], pos_qz[i], pos_qw[i]])   
        pos_tf = pos_global - quaternion_to_matrix(curr_quat) @ np.transpose(tf_trans) 

        quat_tf =  quaternion_from_matrix ( tf_rot @ quaternion_to_matrix(curr_quat) )

        pos_tf_info = np.array([loam_data[i,0], pos_tf[0], pos_tf[1], pos_tf[2], quat_tf[0], quat_tf[1], quat_tf[2], quat_tf[3] ])
        loam_tf  = np.vstack((loam_tf, pos_tf_info ))  
    return loam_tf

def optk_transform(optk_data, ):

    tf_trans = np.array( [-0.172, -0.04, 0.185]) 
    tf_rot  =  R.from_euler('zyx', [[-1.1, 0, 0]], degrees=True)

    pos_x = optk_data[:,1]
    pos_y = optk_data[:,2]
    pos_z = optk_data[:,3]

    pos_qx = optk_data[:,4]
    pos_qy = optk_data[:,5]
    pos_qz = optk_data[:,6]
    pos_qw = optk_data[:,7]

    start_pose = np.array([optk_data[0,1], optk_data[0,2], optk_data[0,3]])
    start_quat = np.array([optk_data[0,4], optk_data[0,5], optk_data[0,6], optk_data[0,7]])

    optk_tf =   np.empty(shape=(0,8))
    for i in range(len(pos_x)): 

        if i%1 is   0: 
            #  Transform local coordinate to global coordinate 
            quat_optk_base = quaternion_multiply( quaternion_inverse(start_quat),tf_rot.as_quat()[0,:] )
            pos_global =   quaternion_to_matrix( quat_optk_base ) @ np.transpose( np.array([pos_x[i], pos_y[i], pos_z[i]]) )- start_pose + tf_trans  

            # Get the Delta Q relative to the start orientation
            curr_quat = np.array([ pos_qx[i], pos_qy[i], pos_qz[i], pos_qw[i]])   
            pos_tf = pos_global - quaternion_to_matrix( quaternion_multiply( quaternion_inverse(quat_optk_base), curr_quat) )@ np.transpose(tf_trans) 

            # print(tf_rot.as_matrix())
            quat_tf =  quaternion_multiply( quaternion_inverse(quat_optk_base), curr_quat)
            # quat_tf = quaternion_multiply( quat_optk_base
            pos_tf_info = np.array([optk_data[i,0], pos_tf[0], pos_tf[1], pos_tf[2], quat_tf[0], quat_tf[1], quat_tf[2], quat_tf[3] ])
 
            optk_tf  = np.vstack((optk_tf, pos_tf_info ))  
    return optk_tf

def save_tf_result(data, fname):
    f = open(fname, "a")
    for i in range(len(data[:,0])):
        f.write("{} {} {} {} {} {} {} {}\n".format(data[i,0], data[i,1], data[i,2], data[i,3],
                                                    data[i,4], data[i,5], data[i,6], data[i,7]))
    # print("-> Saving Loam Pose TUM : ", data.header.stamp.to_sec(),  data.pose.pose.position.x   , data.pose.pose.position.y,  data.pose.pose.position.z ) 
    f.close()


if __name__ == "__main__": 
    folder_path = "/media/hasar/KINGSTON/TIERS LiDAR Dataset/scripts/GT/"
    os0_path =     "   "
    optk_path =   "gt_hard.csv" 
    
    
    os0_data, optk =  read_data(folder_path, os0_path, optk_path)
    os0_pose_tf = pose_transform(os0_data, os0_tf)


    # Optk transform 
    optk_data = optk_transform(optk)

    # #  Saving data
    save_tf_result(optk_data, "./GT/transfored/result_optk.txt")
    save_tf_result(os0_pose_tf, "./GT/transfored/result_os0.txt")
