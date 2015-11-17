import sys
import math
import argparse
import csv
import hmm

# Symbol Definitions
relative_angle = {'facing-toward'     : 00, 'facing-other' : 01                   }
delta_location = {'closer'            : 10, 'farther'      : 11, 'stationary' : 12}
delta_speed    = {'accelerating'      : 20, 'decelerating' : 21, 'constant'   : 22}
delta_angle    = {'turning-toward'    : 30, 'turning-away' : 31, 'constant'   : 32}

# CSV Feature Indices
ownBoat_x         = 0
ownBoat_y         = 1
ownBoat_vx        = 2
ownBoat_vy        = 3
ownBoat_ax        = 4
ownBoat_ay        = 5
ownBoat_theta     = 6
ownBoat_dTheta    = 7

otherBoat_x       = 8
otherBoat_y       = 9
otherBoat_vx      = 10
otherBoat_vy      = 11
otherBoat_ax      = 12
otherBoat_ay      = 13
otherBoat_theta   = 14
otherBoat_dTheta  = 15

# Thresholds
relativeAngleThresh = (math.pi / 12)
distanceThresh = 1.0
accelThresh = 0.1
deltaAngleThresh = 0.01

# Kill program with error message
def killProgramWithMessage(errorMSG):
    print errorMSG
    sys.exit(2)

# Read in inputFeatures
def readInputFile(inputFileName):
    inputFeatures = []

    with open(inputFileName, 'r') as dataFile:
       r = csv.reader(dataFile, delimiter=',')  
       for row in r:
           inputFeatures.append(row)

    inputFeatures = [map(float,x) for x in inputFeatures]
    return computeSymbols(inputFeatures)

# Compute Symbols from Features
def computeSymbols(inputFeatures):
    
    symbols = []

    for feature in inputFeatures:

        symbol = []

        # determine relative angle
        relativeAngle, prev_relativeAngle = getRelativeAngle(feature)

        if relativeAngle < relativeAngleThresh:
            symbol.append(relative_angle['facing-toward'])
        else:
            symbol.append(relative_angle['facing-other'])

        # determine approach
        dist, newDist = getApproach(feature)

        if abs(dist - newDist) < distanceThresh:
            symbol.append(delta_location['stationary'])
        elif newDist < dist:
            symbol.append(delta_location['closer'])
        else:
            symbol.append(delta_location['farther'])

        # determine acceleration
        accel = getAcceleration(feature)

        if abs(accel) < accelThresh:
            symbol.append(delta_speed['constant'])
        elif accel > 0:
            symbol.append(delta_speed['accelerating'])
        else:
            symbol.append(delta_speed['decelerating'])

        # determine delta angle
        if abs(relativeAngle - prev_relativeAngle) < deltaAngleThresh:
            symbol.append(delta_angle['constant'])
        elif relativeAngle > prev_relativeAngle:
            symbol.append(delta_angle['turning-away'])
        else:
            symbol.append(delta_angle['turning-toward'])
        
        # add observed symbols to symbol list
        symbols.append(symbol)

    return symbols

def getRelativeAngle(feature):
    
    # get 'heading' angle
    heading = math.atan2(feature[otherBoat_vy], feature[otherBoat_vx])
    prev_heading = math.atan2(feature[otherBoat_vy] - feature[otherBoat_ay], feature[otherBoat_vx] - feature[otherBoat_ax])
    #print heading, '  ', feature[otherBoat_theta]
    
    # get from other ship to own ship
    own_x, own_y = feature[ownBoat_x], feature[ownBoat_y]
    other_x, other_y = feature[otherBoat_x], feature[otherBoat_y]
    delta_x = own_x - other_x
    delta_y = own_y - other_y
    angle_other_own = math.atan2(delta_y, delta_x)
    #print angle_other_own

    # get relative difference
    return abs(heading - angle_other_own), abs(prev_heading - angle_other_own)

def getApproach(feature):

    # Determine distance in both x and y
    dist_x = feature[otherBoat_x] - feature[ownBoat_x]
    dist_y = feature[otherBoat_y] - feature[ownBoat_y]
    dist = math.sqrt((dist_x * dist_x) + (dist_y * dist_y))

    # Determine distance at t + 1
    new_dist_x = dist_x + feature[otherBoat_vx]
    new_dist_y = dist_y + feature[otherBoat_vy]
    new_dist = math.sqrt((new_dist_x * new_dist_x) + (new_dist_y * new_dist_y))

    return dist, new_dist

def getAcceleration(feature):

    global counter

    x_accel = None
    y_accel = None

    if abs(feature[otherBoat_vx]) < abs(feature[otherBoat_vx] + feature[otherBoat_ax]):
        x_accel = abs(feature[otherBoat_ax])
    else:
        x_accel = -abs(feature[otherBoat_ax])

    if abs(feature[otherBoat_vy]) < abs(feature[otherBoat_vy] + feature[otherBoat_ay]):
        y_accel = abs(feature[otherBoat_ay])
    else:
        y_accel = -abs(feature[otherBoat_ay])

    return math.sqrt(x_accel * x_accel + y_accel * y_accel)

# Train HMM
def train(inputFileName, outputFileName, numStates):

    # Initialize variables
    collapsedObservations = dict()
    allPossibleSymbols = []

    # List of all possible discrete observations
    counter = 0
    for k1, v1 in relative_angle.iteritems():
        for k2, v2 in delta_location.iteritems():
            for k3, v3 in delta_speed.iteritems():
                for k4, v4 in delta_angle.iteritems():
                    counter += 1
                    collapsedObservations[str(v1)+str(v2)+str(v3)+str(v4)] = counter

                    # debug: print out mapping from observation tuple to unique integer
                    print (k1 + ' ' + k2 + ' ' + k3 + ' ' + k4), '==', counter
    
    for k, v in collapsedObservations.iteritems():
        allPossibleSymbols.append(v)

    # More iterations here
    observed_fixed = []
    observedSymbols = readInputFile(inputFileName)

    # Convert list of symbols to unique interger identifier
    for s in observedSymbols:
        observed_fixed.append(collapsedObservations[str(s[0])+str(s[1])+str(s[2])+str(s[3])])
    

    # Try the method in classify.py
    dataLen = 5
    split = []

    #'''
    # bin in groups of dataLen
    for i in range(0, len(observed_fixed), dataLen):

        if i + dataLen > len(observed_fixed):
            split.append(observed_fixed[i:-1])
        else:
            split.append(observed_fixed[i:i+dataLen])
        
        #debug: print training data split into dataLen pieces
        print split[-1]
    #'''  

    '''
    for i in range(dataLen, len(observed_fixed)):
        split.append(observed_fixed[(i - dataLen): i])

        # debug: print training data split into dataLen pieces
        print split[-1]
    '''

    print allPossibleSymbols 

    HMM = hmm.HMM(numStates, V=allPossibleSymbols)
    HMM = hmm.baum_welch(HMM, split)
    print HMM

    log_Prob_Obs, Alpha, c = hmm.forward(HMM, [28, 28, 28, 28, 28])
    print math.exp(log_Prob_Obs)

# Test HMM
def test(inputFileName, inHmmName):
    pass

# Main
if __name__ == '__main__':

    # Parse input arguments
    parser = argparse.ArgumentParser(description='Test or train HMM')
    parser.add_argument('input_file', help='The .csv input file for training/testing data')
    parser.add_argument('mode', help='Either "train" or "test"')
    parser.add_argument('hmm', help='Name of HMM to be trained or tested')
    parser.add_argument('--numStates', help='The desired number of hidden states')
    args = parser.parse_args()

    if args.mode == 'train':
        numStates = 2 

        if args.numStates and args.numStates in rang(2,6):
            numStates = int(numStates)

        train(args.input_file, args.hmm, numStates) 
    elif args.mode == 'test':
        test(args.input_file, args.hmm) 
    else:
        killProgramWithMessage('Argument Error: Invalid Mode')
