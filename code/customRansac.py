import cv2
import numpy as np
import getopt
import sys
import random
import time

from random import Random

#from sklearn.neighbours import KDTree
from scipy.spatial import KDTree

from sympy import *
from mpmath import mp


# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.

    val = (float(q[1] - p[1]) * (r[0] - q[0])) - \
        (float(q[0] - p[0]) * (r[1] - q[1]))
    if (val > 0):

        # Clockwise orientation
        return 1
    elif (val < 0):

        # Counterclockwise orientation
        return 2
    else:

        # Collinear orientation
        return 0


# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1, q1, p2, q2):
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True

    # Special Cases

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True

    # If none of the cases
    return False


#####################################################


#
# Computers a homography from 4-correspondences
#
def calculateHomography(correspondences):
    # loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    # svd composition
    u, s, v = np.linalg.svd(matrixA)

    # reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    # normalize and now we have h
    h = (1/h.item(8)) * h
    return h


#
# Calculate the geometric distance between estimated points and original points
#
def geometricDistance(correspondence, h):

    p1 = np.transpose(
        np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    if (estimatep2.item(2) != 0):
        estimatep2 = (1/estimatep2.item(2))*estimatep2
    else:
        return 1000
    p2 = np.transpose(
        np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


#
# Runs through ransac algorithm, creating homographies from random correspondences
#
def customFindHomography(obj, scene, thresh):

    random.seed(1234)

    correspondenceList = []

    for z in range(len(scene[:, 0])):
        (x1, y1) = obj[z, 0], obj[z, 1]
        (x2, y2) = scene[z, 0], scene[z, 1]
        correspondenceList.append([x1, y1, x2, y2])

    corr = np.matrix(correspondenceList)

    maxInliers = []
    finalH = None
    finalMask = np.zeros(shape=(len(obj[:, 0])))
    for i in range(300):

        mask = np.zeros(shape=(len(obj[:, 0])))

        # find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        # call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 4:
                inliers.append(corr[i])
                mask[i] = 1

        if len(inliers) > len(maxInliers):

            maxInliers = inliers
            finalH = h
            finalMask = mask

        #print ("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):

            break

    return finalH, finalMask


#
# Finds plane through 3 points
#
def planeThroughPoints(p1, p2, p3):
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals -d
    d = - np.dot(cp, p3)
    plane_normal = cp
    plane = [a, b, c, d]

    return plane_normal, plane


#
# Finds plane-point distance
#
def pointPlaneDistance(plane, point):
    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = plane[3]
    x0 = point[0]
    y0 = point[1]
    z0 = point[2]
    return np.abs(a*x0 + b*y0 + c*z0 + d) / np.sqrt(a**2 + b**2 + c**2)


#
# Find Homography checking if the 4 scene points are co-planar first
#
def customFindHomographyPlane3D(obj, scene, point_cloud, thresh):

    random.seed(1234)

    plane_error = 0.01

    correspondenceList = []

    for z in range(len(scene[:, 0])):
        (x1, y1) = obj[z, 0], obj[z, 1]
        (x2, y2) = scene[z, 0], scene[z, 1]
        correspondenceList.append([x1, y1, x2, y2])

    corr = np.matrix(correspondenceList)

    maxInliers = []
    finalH = None
    finalMask = np.zeros(shape=(len(obj[:, 0])))
    for i in range(300):

        mask = np.zeros(shape=(len(obj[:, 0])))

        # find 4 random co-planar points to calculate a homography
        point_on_plane = False
        while(not point_on_plane):
            corr1 = corr[random.randrange(0, len(corr))]
            corr2 = corr[random.randrange(0, len(corr))]
            randomFour = np.vstack((corr1, corr2))
            corr3 = corr[random.randrange(0, len(corr))]
            randomFour = np.vstack((randomFour, corr3))
            corr4 = corr[random.randrange(0, len(corr))]
            randomFour = np.vstack((randomFour, corr4))

            # Qui prende 3 punti, in ogni corr_i ci sono 4 elementi (perchè vengono da corr, dove ogni elemento è formato da 4 numeri: punto 1 e punto 2, vedi sopra)
            # con 3 punti può stimare un piano
            plane_normal, plane = planeThroughPoints(point_cloud[int(corr1[0, 3]), int(corr1[0, 2])],
                                                     point_cloud[int(corr2[0, 3]), int(
                                                         corr2[0, 2])],
                                                     point_cloud[int(corr3[0, 3]), int(corr3[0, 2])])

            point4 = point_cloud[int(corr4[0, 3]), int(corr4[0, 2])]
            distance = pointPlaneDistance(plane, point4)
            if(distance < plane_error):
                point_on_plane = True
                # print("################################Plane found#####################################")
            # else:
                #print("------------------------------------------------------Plane not found----------------------------------------------------------")

        # call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 4:
                inliers.append(corr[i])
                mask[i] = 1

        if len(inliers) > len(maxInliers):

            maxInliers = inliers
            finalH = h
            finalMask = mask

        #print ("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):
            break

    return finalH, finalMask


#
# returns the distance between the centralPoint and the 3d position of the correspondence's scene feature
#
def distance3D(corr, point_sampled, point_cloud):
    p0 = point_cloud[int(corr[0, 3]), int(corr[0, 2])]
    d = np.linalg.norm(point_sampled-p0)
    return d


#
# returning max distance and index of the max-distance element
#
def findMin(corr, point_sampled, point_cloud):

    min = 10000
    minCorr = corr[0]
    for c in corr:
        d = distance3D(c, point_sampled, point_cloud)
        if (d < min):
            min = d
            minCorr = c
    return minCorr


#
# Correspondances are sampled based on the 3D position of the scene features. First one is totally random, while the other 3 are sampled
# through normal distributions on x,y,z. A random 3D point is sampled and then the 3 closest features on scene are found.
#
def normalSampling3D(corr, point_cloud, std_dev):

    # sampling with a uniform distribution the first match
    corr1 = corr[random.randrange(0, len(corr))]
    # taking the 3D position of the scene feature
    firstPoint = point_cloud[int(corr1[0, 3]), int(corr1[0, 2])]
    minCorr = []
    for i in range(3):
        # sampling with normal distributions a point in space
        x_sampled = np.random.normal(firstPoint[0], std_dev)
        y_sampled = np.random.normal(firstPoint[1], std_dev)
        z_sampled = np.random.normal(firstPoint[2], std_dev)
        point_sampled = [x_sampled, y_sampled, z_sampled]

        # find the corr with min distance from point_sampled
        minCorr.append(findMin(corr, point_sampled, point_cloud))

    # Time Complexity: O((n-k)*k)

    # corr1 + minDistanceThree are the 4 correspondances to return
    corr2 = minCorr[0]
    randomFour = np.vstack((corr1, corr2))
    corr3 = minCorr[1]
    randomFour = np.vstack((randomFour, corr3))
    corr4 = minCorr[2]
    randomFour = np.vstack((randomFour, corr4))

    return randomFour


def customFindHomographyNormalSampling3D(obj, scene, point_cloud, thresh, std_dev):

    random.seed(1234)

    correspondenceList = []

    for z in range(len(scene[:, 0])):
        (x1, y1) = obj[z, 0], obj[z, 1]
        (x2, y2) = scene[z, 0], scene[z, 1]
        correspondenceList.append([x1, y1, x2, y2])

    corr = np.matrix(correspondenceList)

    maxInliers = []
    finalH = None
    finalMask = np.zeros(shape=(len(obj[:, 0])))
    for i in range(300):

        mask = np.zeros(shape=(len(obj[:, 0])))

        randomFour = normalSampling3D(corr, point_cloud, std_dev)

        # call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 4:
                inliers.append(corr[i])
                mask[i] = 1

        if len(inliers) > len(maxInliers):

            maxInliers = inliers
            finalH = h
            finalMask = mask

        #print ("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):

            break

    return finalH, finalMask


#########################################################
# INIZIO AGGIUNTA

def findNearPoint(corr1, firstPoint, corr, point_cloud, randomFour, dimension):

    # Creo un'istanza del seed solo per questa funzione così ogni volta posso ordinare gli indici in modo diverso (così tutto è più randomico)
    myRandom_2 = Random(time.time())
    valid = False
    not_found = False
    count = 0

    while((not valid) and (not not_found)):

        corr2 = corr[myRandom_2.randrange(0, len(corr))]

        secondPoint = point_cloud[int(corr2[0, 3]), int(corr2[0, 2])]

        #print("FIRST: " + str(firstPoint) + " ||| SECOND: " + str(secondPoint))

        not_zero = secondPoint[0] != 0.0 and secondPoint[1] != 0.0 and secondPoint[2] != 0.0

        if(not_zero and (abs(firstPoint[0] - secondPoint[0]) <= dimension[0]) and (abs(firstPoint[1] - secondPoint[1]) <= dimension[1]) and (abs(firstPoint[2] - secondPoint[2]) <= dimension[2]) and np.any(corr1 != corr2) and (corr2 not in randomFour)):

            valid = True

        if(count == len(corr)):
            # print(count)
            not_found = True

        count = count + 1

    #print("FINE findNearPoint")

    if(not_found):
        #print("if not found")
        return randomFour

    if(len(randomFour) != 0):
        #print("if randomFour == 0")
        randomFour = np.vstack((randomFour, corr2))

    else:
        # print("else")
        randomFour = np.vstack((corr1, corr2))

    return randomFour


def dimensionSampling(corr, point_cloud, dimension):

    valid = False
    randomFour = None

    myRandom_1 = Random(time.time())

    count = 0

    while(not valid):

        not_zero = False

        while(not not_zero):

            # Estrai il primo punto
            corr1 = corr[myRandom_1.randrange(0, len(corr))]

            # Prendi posizione nello spazio del primo punto
            firstPoint = point_cloud[int(corr1[0, 3]), int(corr1[0, 2])]

            if(firstPoint[0] != 0.0 and firstPoint[1] != 0.0 and firstPoint[2] != 0.0):
                not_zero = True

        randomFour = []

        #print("dimensionSampling - DONE --- " + " VALID: " + str(valid) + " --- NOT_ZERO: " + str(not_zero))

        if count > len(corr):
            dimension[0] = dimension[0] + 0.05
            dimension[1] = dimension[1] + 0.05
            dimension[2] = dimension[2] + 0.05
            print("--- Incremented limits ---")

        for i in range(3):
            start_dim = len(randomFour)

            #print("CALL TO FIND NEAR")

            randomFour = findNearPoint(
                corr1, firstPoint, corr, point_cloud, randomFour, dimension)

            # Controllo di aver aggiunto un corr a randomFour, nel caso in cui non l'abbia trovato vuol dire che non ci sono punti abbastanza vicini a corr1
            # e quindi devo estrarre un nuovo corr1
            if(start_dim == len(randomFour)):
                valid = False
                break
            else:
                valid = True

        count = count + 1

    return randomFour

def customFindHomographyDimension2(obj,scene,point_cloud, thresh, dimension):
    dimension = np.array(dimension)
    diagonal = sqrt(np.dot(dimension,dimension.T))
    myRandom_1 = Random(time.time())
    myRandom_1.seed(1234)
    
    correspondenceList = []

    for z in range(len(scene[:,0])):
            (x1, y1) = obj[z,0] , obj[z,1]
            (x2, y2) = scene[z,0] , scene[z,1]
            correspondenceList.append([x1, y1, x2, y2])

    corr = np.matrix(correspondenceList)

    maxInliers = []
    finalH = None
    finalMask = np.zeros(shape = (len(obj[:,0])) )
    for i in range(150):

        mask = np.zeros(shape = (len(obj[:,0])) )

        #find 4 random points to calculate a homography
        corr1 = corr[myRandom_1.randrange(0, len(corr))]
        firstPoint = point_cloud[int(corr1[0, 3]), int(corr1[0, 2])] #pos in the space of corr1
        randomFour = np.vstack(corr1)
        
        while len(corr) and len(randomFour)<4:
            c = corr[myRandom_1.randrange(0, len(corr))]
            secondPoint = point_cloud[int(c[0, 3]), int(c[0, 2])]
            diff = firstPoint-secondPoint
            dist = sqrt(np.dot(diff,diff.T))*0.7
            if dist<= diagonal:
                randomFour = np.vstack((randomFour,c))

        #call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []
        

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 4:
                inliers.append(corr[i])
                mask[i] = 1
                

        if len(inliers) > len(maxInliers):
            
            maxInliers = inliers
            finalH = h
            finalMask = mask

        #print ("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))
        if len(maxInliers) > (len(corr)*thresh):
            break

    return finalH, finalMask;
#
def customFindHomographyRectangles(obj,scene, thresh,correspondenceList2):
    correspondenceList = []

    for z in range(len(scene[:,0])):
            (x1, y1) = obj[z,0] , obj[z,1]
            (x2, y2) = scene[z,0] , scene[z,1]
            correspondenceList.append([x1, y1, x2, y2])

    corr = np.matrix(correspondenceList)
    corr2 = np.matrix(correspondenceList2)

    maxInliers = []
    finalH = None
    finalMask = np.zeros(shape = (len(obj[:,0])) )
    
    for i in range(50):
        mask = np.zeros(shape = (len(obj[:,0])) )
        randomlist=random.sample(range(0, len(corr2)), thresh)
        randomFour = np.vstack((corr2[randomlist[0]],(corr2[randomlist[0]])))
         # find 4 random points to calculate a homography
        for i in range(2,len(randomlist)):
            randomFour = np.vstack((randomFour,(corr2[randomlist[i]])))
        #call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []
        

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 4:
                inliers.append(corr[i])
                mask[i] = 1
                

        if len(inliers) > len(maxInliers):
            
            maxInliers = inliers
            finalH = h
            finalMask = mask

        #print ("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))
        if len(maxInliers) > (len(corr)*thresh):
            break

    return finalH, finalMask;
#

def customFindHomographyDimension(obj, scene, point_cloud, thresh, dimension):

    # random.seed(1234)

    correspondenceList = []

    for z in range(len(scene[:, 0])):
        (x1, y1) = obj[z, 0], obj[z, 1]
        (x2, y2) = scene[z, 0], scene[z, 1]
        correspondenceList.append([x1, y1, x2, y2])

    corr = np.matrix(correspondenceList)

    maxInliers = []
    finalH = None
    finalMask = np.zeros(shape=(len(obj[:, 0])))
    for i in range(300):

        mask = np.zeros(shape=(len(obj[:, 0])))

        randomFour = dimensionSampling(corr, point_cloud, dimension)

        # call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 4:
                inliers.append(corr[i])
                mask[i] = 1

        if len(inliers) > len(maxInliers):

            maxInliers = inliers
            finalH = h
            finalMask = mask

        #print ("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):

            break

    return finalH, finalMask


# FINE AGGIUNTA
#########################################################


def buildKDTree(obj, scene, point_cloud):
    correspondenceList = []

    for z in range(len(scene[:, 0])):
        (x1, y1) = obj[z, 0], obj[z, 1]
        (x2, y2) = scene[z, 0], scene[z, 1]
        correspondenceList.append([x1, y1, x2, y2])

    corr = np.matrix(correspondenceList)

    # build tree
    X = []
    for i in range(len(corr)):
        X.append(point_cloud[int(corr[i, 3]), int(corr[i, 2])])
    X = np.matrix(X)
    tree = KDTree(X, leafsize=2)

    return tree, corr

#
# randomly sample the other 3 points in a certain 3d-tree region containing the closest points
#


def sampling3DTree(corr, point_cloud, tree):
    # sampling with a uniform distribution the first match
    corr1 = corr[random.randrange(0, len(corr))]
    # taking the 3D position of the scene feature
    firstPoint = point_cloud[int(corr1[0, 3]), int(corr1[0, 2])]

    # find the k corrs with min distance from point_sampled
    dist, ind = tree.query(np.asarray(firstPoint).reshape((1, -1)), k=20)

    # random points
    ind = random.sample(list(ind[0]), 3)

    corr2 = corr[ind[0]]
    randomFour = np.vstack((corr1, corr2))
    corr3 = corr[ind[1]]
    randomFour = np.vstack((randomFour, corr3))
    corr4 = corr[ind[2]]
    randomFour = np.vstack((randomFour, corr4))

    return randomFour

# https://math.stackexchange.com/questions/3509039/calculate-homography-with-and-without-svd


def homographyEstimateSVD(points):
    mat = []
    for p in points:

        x_1 = [p[0, 0], p[0, 1]]
        y_1 = [p[0, 2], p[0, 3]]
        row1 = [x_1[0]*-1, y_1[0]*-1, -1, 0, 0, 0,
                x_1[0]*x_1[1], y_1[0]*x_1[1], x_1[1]]
        row2 = [0, 0, 0, x_1[0]*-1, y_1[0]*-1, -1,
                x_1[0]*y_1[1], y_1[0]*y_1[1], y_1[1]]
        mat.append(row1)
        mat.append(row2)

    P = np.matrix(mat)
    [U, S, Vt] = np.linalg.svd(P)
    homography = Vt[-1].reshape(3, 3)

    return homography

#
# Find Homography, sampling is not uniform but based on 3D distance of scene features. Sampling of the first match is random, then
# sampling of the other 3 points is based on a kd tree built using the 3 spatial dimensions
#


def customFindHomography3DTree(obj, scene, point_cloud, thresh):

    random.seed(1234)

    #print("Building Tree")
    # build KDTree between points in the 3d scene
    tree, corr = buildKDTree(obj, scene, point_cloud)

    maxInliers = []
    finalH = None
    finalMask = np.zeros(shape=(len(obj[:, 0])))
    for i in range(300):

        mask = np.zeros(shape=(len(obj[:, 0])))

        randomFour = sampling3DTree(corr, point_cloud, tree)

        # call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 4:
                inliers.append(corr[i])
                mask[i] = 1

        if len(inliers) > len(maxInliers):

            maxInliers = inliers
            finalH = h
            finalMask = mask

        #print ("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):

            break

    # VALIDATION
    #estimatedHomography = homographyEstimateSVD(maxInliers)
    #finalInliers = []
    #finalMask = np.zeros(shape = (len(obj[:,0])) )
#
    # for i in range(len(corr)):
    #    d = geometricDistance(corr[i], h)
    #    if d < 3:
    #        finalInliers.append(corr[i])
    #        finalMask[i] = 1

    return finalH, finalMask
