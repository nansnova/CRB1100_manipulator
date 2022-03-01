"""
This code aims to portrait the general algorithm to describe and simulate the direct kinematics
of a the ABB CRB 1100 arm.

Authors:
- Leonardo Gracida Muñoz A01379812
- Nancy Lesly García Jiménez A01378043
- Jose Angel Del Angel Dominguez A01749386
"""
import numpy as np
import spatialmath as sm
import matplotlib.pyplot as plt
import time

def npVector2List(vector):
    """ converts a numpy 2d array with the values of a vector to a python list """
    return [ vector[0,0], vector[1,0], vector[2,0] ]

def angleIsReached(th, th_desired):
    """ this function is necesary to simulate positive
    and negative angles in every joint of the robot.
    The function determines if the angle th_desired is reached
    after a small increment of th towards th_desired"""
    if (th_desired > 0):
        if(th >= th_desired):
            return True
        else:
            return False
    elif (th_desired < 0):
        if(th <= th_desired):
            return True
        else:
            return False
    elif (th_desired == 0):
        if(th == 0):
            return True
        else:
            return False
        
def stepTowardsThDesired(th, th_desired, step = 5*(3.1416/180) ):
    """ this function is necesary to simulate positive
    and negative angles in every joint of the robot.
    The function ensures that an increment of th is done pointing
    in the direction of th_desired """
    if (th_desired > 0):
        return step
    elif (th_desired < 0):
        return -step
    
def updateJoints(q1, q2, q3, q4, q5, oe, w1, w2, w3, w4, w5, th1, th2, th3, th4, th5):
    """ This function returns the new cordinates of the robot joints and end
    effector based in the transformation of robot's reference configuration after
    each joint is moved theta1 (th1),theta2 (th2),theta3 (th3) and theta4 (th4)
    degrees
    Returns [q3, q5, oe] """
    #th1 rotation from base along z axis
    #th2 rotation from 1st joint along y axis
    #th3 rotation from 2nd joint along y axis (elbow)
    #th4 rotation from 2nd joint along x axis
    #th5 rotation from joint in final efector along y axis

    xi1 = sm.Twist3(-np.cross( npVector2List(w1) , npVector2List(q1) ), npVector2List(w1) )
    xi2 = sm.Twist3(-np.cross( npVector2List(w2) , npVector2List(q2) ), npVector2List(w2) )
    xi3 = sm.Twist3(-np.cross( npVector2List(w3) , npVector2List(q3) ), npVector2List(w3) )
    xi4 = sm.Twist3(-np.cross( npVector2List(w4) , npVector2List(q4) ), npVector2List(w4) )
    xi5 = sm.Twist3(-np.cross( npVector2List(w5) , npVector2List(q5) ), npVector2List(w5) )
    
    gs1 = xi1.exp(th1)
    g12 = xi2.exp(th2)
    g23 = xi3.exp(th3)
    g34 = xi4.exp(th4)
    g45 = xi5.exp(th5)
    
    gs2 = sm.SE3(gs1 @ g12)
    gs4 = sm.SE3(gs1 @ g12 @ g23 @  g34)
    gs5 = sm.SE3(gs1 @ g12 @ g23 @  g34 @ g45)
    
    new_q3 = gs2 * npVector2List(q3)
    new_q5 = gs4 * npVector2List(q5)
    new_oe = gs5 * npVector2List(oe)
    
    return [new_q3, new_q5, new_oe]
    
    
""" ==================== JOINTS POSITIONS AND ROTATION AXES IN REFERENCE CONFIGURATION =========== """

robot_base = np.array([[0],
                       [0],
                       [-327]])
q1 = np.array([[0],
               [0],
               [0]])
q2 = np.array([[0],
               [0],
               [0]])
q3 = np.array([[0],
               [0],
               [235]])
q4 = np.array([[0],
               [0],
               [235]])
q5 = np.array([[250],
               [0],
               [235]])
oe = np.array([[314],
               [0],
               [235]])

w1 = np.array([[0],
               [0],
               [1]])
w2 = np.array([[0],
               [1],
               [0]])
w3 = np.array([[0],
               [1],
               [0]])
w4 = np.array([[1],
               [0],
               [0]])
w5 = np.array([[0],
               [1],
               [0]])

""" ==================== END OF JOINTS POSITIONS AND ROTATION AXES IN REFERENCE CONFIGURATION =========== """

theta1 = 0
theta2 = 0
theta3 = 0
theta4 = 0
theta5 = 0
""" ============ CHANGE THE FOLLOWING VALUES TO DTERMINE THE ROBOT MOVEMENT ========== """
theta1_final = 45*(np.pi/180)   
theta2_final = 29.46215223*(np.pi/180)
theta3_final = 10.29739296*(np.pi/180)
theta4_final = 0*(np.pi/180)
theta5_final = 4.56502313*(np.pi/180)
""" ============ END OF CHANGE THE FOLLOWING VALUES TO DTERMINE THE ROBOT MOVEMENT ========== """

""" ==================== GENERAL PLOT INITIALIZATION =========== """

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-500, 500])
ax.set_ylim([-500, 500])
ax.set_zlim([-327, 500])

""" ==================== GENERAL PLOT INITIALIZATION  ENDS =========== """

""" ======= PLOTING ROBOT'S REFERENCE CONFIGURATION ====== """

l0_cords = np.concatenate((robot_base, q2), axis = 1)
l1_cords = np.concatenate((q2, q3), axis = 1)
l2_cords = np.concatenate((q3, q5), axis = 1)
l3_cords = np.concatenate((q5, oe), axis = 1)
l0, = ax.plot(l0_cords[0,:] , l0_cords[1, :], l0_cords[2, :], label='L0')
l1, = ax.plot(l1_cords[0,:] , l1_cords[1, :], l1_cords[2, :], label='L1')
l2, = ax.plot(l2_cords[0,:] , l2_cords[1, :], l2_cords[2, :], label='L2')
l3, = ax.plot(l3_cords[0,:] , l3_cords[1, :], l3_cords[2, :], label='L3')
ax.legend()

""" ======= END OF PLOTING ROBOT'S REFERENCE CONFIGURATION ====== """

step = 1*(np.pi/180)
seconds_between_frames = 0.025

while not( angleIsReached(theta1, theta1_final) ):    
    theta1 = theta1 + stepTowardsThDesired(theta1, theta1_final, step)
    
    newJoints = updateJoints(q1, q2, q3, q4, q5, oe, w1.copy(), w2.copy(), w3.copy(), w4.copy(), w5.copy(), theta1, theta2, theta3, theta4, theta5)
    q3_n = newJoints[0]
    q5_n = newJoints[1]
    oe_n = newJoints[2]
    
    l1_cords = np.concatenate((q2, q3_n), axis = 1)
    l2_cords = np.concatenate((q3_n, q5_n), axis = 1)
    l3_cords = np.concatenate((q5_n, oe_n), axis = 1)
    
    """ === UPDATING PYPLOT DATA TO CREATE ANIMATION === """
    l1.set_data_3d(l1_cords[0,:], l1_cords[1,:], l1_cords[2,:])
    l2.set_data_3d(l2_cords[0,:], l2_cords[1,:], l2_cords[2,:])
    l3.set_data_3d(l3_cords[0,:], l3_cords[1,:], l3_cords[2,:])
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    """ === END OF UPDATING PYPLOT DATA === """
    time.sleep(seconds_between_frames)
    
    
    
while not( angleIsReached(theta2, theta2_final) ):
    theta2 = theta2 + stepTowardsThDesired(theta2, theta2_final, step)
    
    newJoints = updateJoints(q1, q2, q3, q4, q5, oe, w1.copy(), w2.copy(), w3.copy(), w4.copy(), w5.copy(), theta1, theta2, theta3, theta4, theta5)
    q3_n = newJoints[0]
    q5_n = newJoints[1]
    oe_n = newJoints[2]
    
    l1_cords = np.concatenate((q2, q3_n), axis = 1)
    l2_cords = np.concatenate((q3_n, q5_n), axis = 1)
    l3_cords = np.concatenate((q5_n, oe_n), axis = 1)
    
    """ === UPDATING PYPLOT DATA TO CREATE ANIMATION === """
    l1.set_data_3d(l1_cords[0,:], l1_cords[1,:], l1_cords[2,:])
    l2.set_data_3d(l2_cords[0,:], l2_cords[1,:], l2_cords[2,:])
    l3.set_data_3d(l3_cords[0,:], l3_cords[1,:], l3_cords[2,:])
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    """ === END OF UPDATING PYPLOT DATA === """
    time.sleep(seconds_between_frames)

    
while not( angleIsReached(theta3, theta3_final) ):
    theta3 = theta3 + stepTowardsThDesired(theta3, theta3_final, step)
    
    newJoints = updateJoints(q1, q2, q3, q4, q5, oe, w1.copy(), w2.copy(), w3.copy(), w4.copy(), w5.copy(), theta1, theta2, theta3, theta4, theta5)
    q3_n = newJoints[0]
    q5_n = newJoints[1]
    oe_n = newJoints[2]
    
    l1_cords = np.concatenate((q2, q3_n), axis = 1)
    l2_cords = np.concatenate((q3_n, q5_n), axis = 1)
    l3_cords = np.concatenate((q5_n, oe_n), axis = 1)
    
    """ === UPDATING PYPLOT DATA TO CREATE ANIMATION === """
    l1.set_data_3d(l1_cords[0,:], l1_cords[1,:], l1_cords[2,:])
    l2.set_data_3d(l2_cords[0,:], l2_cords[1,:], l2_cords[2,:])
    l3.set_data_3d(l3_cords[0,:], l3_cords[1,:], l3_cords[2,:])
    
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    """ === END OF UPDATING PYPLOT DATA === """
    time.sleep(seconds_between_frames)
    
while not( angleIsReached(theta4, theta4_final) ):
    theta4 = theta4_final
    
    newJoints = updateJoints(q1, q2, q3, q4, q5, oe, w1.copy(), w2.copy(), w3.copy(), w4.copy(), w5.copy(), theta1, theta2, theta3, theta4, theta5)
    q3_n = newJoints[0]
    q5_n = newJoints[1]
    oe_n = newJoints[2]
    
    l1_cords = np.concatenate((q2, q3_n), axis = 1)
    l2_cords = np.concatenate((q3_n, q5_n), axis = 1)
    l3_cords = np.concatenate((q5_n, oe_n), axis = 1)
    
    """ === UPDATING PYPLOT DATA TO CREATE ANIMATION === """
    l1.set_data_3d(l1_cords[0,:], l1_cords[1,:], l1_cords[2,:])
    l2.set_data_3d(l2_cords[0,:], l2_cords[1,:], l2_cords[2,:])
    l3.set_data_3d(l3_cords[0,:], l3_cords[1,:], l3_cords[2,:])
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    """ === END OF UPDATING PYPLOT DATA === """
    time.sleep(seconds_between_frames)
    
while not( angleIsReached(theta5, theta5_final) ):
    theta5 = theta5 + stepTowardsThDesired(theta5, theta5_final, step)
    
    newJoints = updateJoints(q1, q2, q3, q4, q5, oe, w1.copy(), w2.copy(), w3.copy(), w4.copy(), w5.copy(), theta1, theta2, theta3, theta4, theta5)
    q3_n = newJoints[0]
    q5_n = newJoints[1]
    oe_n = newJoints[2]
    
    l1_cords = np.concatenate((q2, q3_n), axis = 1)
    l2_cords = np.concatenate((q3_n, q5_n), axis = 1)
    l3_cords = np.concatenate((q5_n, oe_n), axis = 1)
    
    """ === UPDATING PYPLOT DATA TO CREATE ANIMATION === """
    l1.set_data_3d(l1_cords[0,:], l1_cords[1,:], l1_cords[2,:])
    l2.set_data_3d(l2_cords[0,:], l2_cords[1,:], l2_cords[2,:])
    l3.set_data_3d(l3_cords[0,:], l3_cords[1,:], l3_cords[2,:])
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    """ === END OF UPDATING PYPLOT DATA === """
    time.sleep(seconds_between_frames)


newJoints = updateJoints(q1, q2, q3, q4, q5, oe, w1.copy(), w2.copy(), w3.copy(), w4.copy(), w5.copy(), theta1_final, theta2_final, theta3_final, theta4_final, theta5_final)
q3_n = newJoints[0]
q5_n = newJoints[1]
oe_n = newJoints[2]
    
l1_cords = np.concatenate((q2, q3_n), axis = 1)
l2_cords = np.concatenate((q3_n, q5_n), axis = 1)
l3_cords = np.concatenate((q5_n, oe_n), axis = 1)
    
""" === UPDATING PYPLOT DATA TO CREATE ANIMATION === """
l1.set_data_3d(l1_cords[0,:], l1_cords[1,:], l1_cords[2,:])
l2.set_data_3d(l2_cords[0,:], l2_cords[1,:], l2_cords[2,:])
l3.set_data_3d(l3_cords[0,:], l3_cords[1,:], l3_cords[2,:])
    
fig.canvas.draw()
fig.canvas.flush_events()
    

print("final end efector position is")
print(np.array2string(oe_n))
