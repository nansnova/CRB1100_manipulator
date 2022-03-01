import numpy as np
import spatialmath as sm
import copy
np.seterr(all='raise') #this line makes any numpy warning into an exception
# so that we can catch the FloatingPointError exception

acos = np.arccos
d3Solutions = []
solutions = []

def cos_sol(l1,l2,l3,d2,d3):
    """This funtion evaluates the function E(n) given by the geometric analysys
    of the ABB CRB 1100 ARM. In case that the arguments given to an arccos argument
    are out of the arcos function baoundaries [-1, 1], this function returns the number
    3000, meaning a huge error in the inverse kinematics calculation given current l1,
    l2, l3, d2, d3
    
    returns: Inverse kinematics Error [Number]
    """
    try:
        temp1 = ((l2**2)-(l1**2)-(d3**2))/((-2)*d3*l1)
        temp2 = ((l3**2)-(d3**2)-(d2**2))/((-2)*d3*d2)
        temp3 = ((d3**2)-(l1**2)-(l2**2))/((-2)*l1*l2)
        temp4 = ((l1**2)-(d3**2)-(l2**2))/((-2)*d3*l2)
        temp5 = ((d2**2)-(d3**2)-(l3**2))/((-2)*d3*l3)
        temp6 = ((d3**2)-(d2**2)-(l3**2))/((-2)*d2*l3)
        res = ( acos( temp1 ) + acos( temp2 ) + acos( temp3 ) + acos( temp4 ) + acos( temp5 ) + acos( temp6 ) )
        return res -2*np.pi
    except FloatingPointError:
        return 3000
    
def npVector2List(vector):
    """ converts a numpy 2d array with the values of a vector to a python list """
    return [ vector[0,0], vector[1,0], vector[2,0] ]

def err_in_fwd_K_ef(oe_desired, q1, q2, q3, q4, q5, inital_oe, w1, w2, w3, w4, w5, th1, th2, th3, th4, th5):
    """ This function returns the new cordinates of the robot end
    effector based in the transformation of robot's reference configuration after
    each joint is moved theta1 (th1),theta2 (th2),theta3 (th3) and theta4 (th4)
    degrees
    Returns: ef : end efector new position """
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
    
    gs5 = sm.SE3(gs1 @ g12 @ g23 @  g34 @ g45)
    
    new_oe = gs5 * npVector2List(oe)
    
    error = new_oe - oe_desired
    
    return np.linalg.norm(error)

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

l1 = 235
l2 = 250
l3 = 64

pd = np.array([[250],
                [250],
               [0]])

theta1 = np.arctan2(pd[1,0],pd[0,0])
d1 = np.sqrt( pd[0,0]**2 + pd[1,0]**2 )

phi = np.arctan2(pd[2,0], d1)
d2 = np.sqrt(d1**2 + pd[2,0]**2)
d3 = 5
iterating_rate = 0.1

while ( True ):
    res = cos_sol(l1,l2,l3,d2,d3)
    if (d3 >= 570):
        break
    elif( abs(res) != 0 ):
        d3 += iterating_rate
    else:
        d3Solutions.append(copy.deepcopy(d3))
        d3 += iterating_rate
    
for i in range(len(d3Solutions)):
    a2 = acos( ((l2**2)-(l1**2)-(d3Solutions[i]**2))/((-2)*d3Solutions[i]*l1) ) + acos( ((l3**2)-(d3Solutions[i]**2)-(d2**2))/((-2)*d3Solutions[i]*d2) )
    a3 = acos( ((d3Solutions[i]**2)-(l1**2)-(l2**2))/((-2)*l1*l2) )
    a5 = acos( ((l1**2)-(d3Solutions[i]**2)-(l2**2))/((-2)*d3Solutions[i]*l2) ) + acos( ((d2**2)-(d3Solutions[i]**2)-(l3**2))/((-2)*d3Solutions[i]*l3) )
        
    theta2 = (np.pi/2) - phi - a2
    theta3 = (np.pi/2) - a3
    theta4 = 0
    theta5 = np.pi - a5
    solutions.append([theta1, theta2, theta3, theta4, theta5])
    
solutions = np.array(solutions)

print("solutions in rads are:")
print("[theta1 \t, theta2 \t, theta3 \t, theta4 \t, theta5 \t]")
print(np.array2string(solutions))
print("solutions in deg are:")
print("[theta1 \t, theta2 \t, theta3 \t, theta4 \t, theta5 \t]")
print(np.array2string(solutions*(180/np.pi)))


""" We declare 4 kinds of solutions
    1. Elbow down wrist down - [th3 < 0, th5 < 0]
    2. Elbow down wrist up -   [th3 < 0, th5 >= 0]
    3. Elbow up wrist down -   [th3 >= 0, th5 < 0]
    4. Elbow up wrist up -     [th2 >= 0, th5 >= 0]
"""
class1_sols = []
class2_sols = []
class3_sols = []
class4_sols = []

for i in range(len(solutions)):
    if(solutions[i, 2] < 0 and solutions[i,4] < 0):
        class1_sols.append( solutions[i,:] )
    elif(solutions[i, 2] < 0 and solutions[i,4] >= 0):
        class2_sols.append( solutions[i,:] )
    elif(solutions[i, 2] >= 0 and solutions[i,4] < 0):
        class3_sols.append( solutions[i,:] )
    elif(solutions[i, 2] >= 0 and solutions[i,4] >= 0):
        class4_sols.append( solutions[i,:] )


def map_helper(theta):
    global pd
    global q1
    global q2
    global q3
    global q4
    global q5
    global oe
    global w1
    global w2
    global w3
    global w4
    global w5
    
    return err_in_fwd_K_ef(pd, q1, q2, q3, q4, q5, oe, w1, w2, w3, w4, w5, theta[0], theta [1], theta[2], theta[3], theta[4])


print("""
Consider the following 4 kinds of solutions
    1. Elbow down wrist down - [th3 < 0, th5 < 0]
    2. Elbow down wrist up -   [th3 < 0, th5 >= 0]
    3. Elbow up wrist down -   [th3 >= 0, th5 < 0]
    4. Elbow up wrist up -     [th2 >= 0, th5 >= 0]
""")

print("\nThe best class 1 solution [degrees] is:")
if len(class1_sols) > 0:
    class1_err = list(map(map_helper, class1_sols))
    best_c1_index = class1_err.index(min(class1_err))
    best_c1 = class1_sols[best_c1_index]
    print(best_c1*(180/np.pi))
else:
    best_c1 = "there are no posible class 1 solutions for this point"
    print(best_c1)
    
print("\nThe best class 2 solution [degrees] is:")
if len(class2_sols) > 0:
    class2_err = list(map(map_helper, class2_sols))
    best_c2_index = class2_err.index(min(class2_err))
    best_c2 = class2_sols[best_c2_index]
    print(best_c2*(180/np.pi))
else:
    best_c2 = "there are no posible class 2 solutions for this point"
    print(best_c2)
    
print("\nThe best class 3 solution [degrees] is:")
if len(class3_sols) > 0:
    class3_err = list(map(map_helper, class3_sols))
    best_c3_index = class3_err.index(min(class3_err))
    best_c3 = class3_sols[best_c3_index]
    print(best_c3*(180/np.pi))
else:
    best_c3 = "there are no posible class 3 solutions for this point"
    print(best_c3)

print("\nThe best class 4 solution [degrees] is:")
if len(class4_sols) > 0:
    class4_err = list(map(map_helper, class4_sols))
    best_c4_index = class4_err.index(min(class4_err))
    best_c4 = class4_sols[best_c4_index]
    print(best_c4*(180/np.pi))
else:
    best_c4 = "there are no posible class 4 solutions for this point"
    print(best_c4)
    

