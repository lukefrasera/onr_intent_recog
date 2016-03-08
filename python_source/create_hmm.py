import sys
import os
import math
import argparse
import csv
import pickle
import numpy as np

# HMM Library
from hmmlearn import hmm

# Symbol Definitions
relative_angle         = {'facing-toward'     : 00, 'facing-other' : 01                   }
delta_location         = {'closer'            : 10, 'farther'      : 11, 'stationary' : 12}
delta_speed            = {'accelerating'      : 20, 'decelerating' : 21, 'constant'   : 22}
delta_angle            = {'turning-toward'    : 30, 'turning-away' : 31, 'constant'   : 32}
delta_relative_heading = {'increasing'        : 40, 'decreasing'   : 41, 'constant'   : 42} 
#cpa_delta_distance     = {'increasing'        : 50, 'decreasing'   : 51, 'constant'   : 52}
cpa_time               = {'positive'          : 60, 'negative'     : 61}
cpa_distance_thresh    = {'above'             : 70, 'below'        : 71}

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

timeStamp         = 16
contact_id        = 17

cpa_dDist         = 18
cpa_vTime         = 19

cpa_dist          = 20

# Thresholds
relativeAngleThresh = (math.pi / 12)
distanceThresh = 1.0
accelThresh = 0.1
deltaAngleThresh = 0.01
relHeadThresh = 0.001
cpaDeltaDistThresh = 0.01
cpaDistanceThresh = 200.0

# Other controls
dataLen = 20

# Testing HMM names
ramHMM    = 'hmm_ram'
blockHMM  = 'hmm_block'
herdHMM   = 'hmm_herd'
benignHMM = 'hmm_benign'

all_hmms    = [ramHMM, blockHMM, herdHMM, benignHMM]
intents     = ['1', '5', '7', '3'] # ram, block, herd, benign
all_intents = ['0', '8'] + intents # unknown + missing + intents

# Kill program with error message
def killProgramWithMessage(errorMSG):
    print errorMSG
    sys.exit(2)

# Get list of files under directory
def getFilesFromDir(topLevelDir):

    fileList = []

    # In case the input dir as a tailing '/' on it
    if topLevelDir[-1] != '/':
        topLevelDir = topLevelDir + '/'

    # Get a list of all the subdirectories
    subDirs = [name for name in os.listdir(topLevelDir) if os.path.isdir(os.path.join(topLevelDir, name))]

    # Get all training files in those subdirectories
    for subDir in subDirs:
        expectedFile = topLevelDir + subDir +'/hmm_formatted.csv'

        if os.path.isfile(expectedFile):
            fileList.append(expectedFile)

    # Return list of all training files
    return fileList 

# Read in inputFeatures
def readInputFile(inputFileName):
    inputFeatures = []

    with open(inputFileName, 'r') as dataFile:
       r = csv.reader(dataFile, delimiter=',')  
       for row in r:
           
           # Check for ignored row 
            if row[0][0] == '#':
                continue

            inputFeatures.append(row)

    inputFeatures = [map(float,x) for x in inputFeatures]
    return computeSymbols(inputFeatures), inputFeatures

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
        
        # determine delta relative heading 
        relHead = getRelativeHeading(feature)

        if abs(relHead) < relHeadThresh:
            symbol.append(delta_relative_heading['constant'])
        elif relHead > 0:
            symbol.append(delta_relative_heading['increasing'])
        else:
            symbol.append(delta_relative_heading['decreasing'])
            
        # determine CPA distance
        #cpaDeltaDist = float(getCpaDistanceDelta(feature))

        #if abs(cpaDeltaDist) < cpaDeltaDistThresh :
        #    symbol.append(cpa_delta_distance['constant'])
        #elif cpaDeltaDist < 0.0:
        #    symbol.append(cpa_delta_distance['decreasing'])
        #else:
        #    symbol.append(cpa_delta_distance['increasing'])

        # determine CPA time
        cpaTime = float(getCpaTimeValue(feature))

        if cpaTime <= 0.0:
            symbol.append(cpa_time['negative'])
            #print cpaTime, 'negative'
        else:
            symbol.append(cpa_time['positive'])
            #print cpaTime, 'positive'

        # check against CPA distance threshold
        if float(feature[cpa_dist]) > cpaDistanceThresh:
            symbol.append(cpa_distance_thresh['above'])
        else:
            symbol.append(cpa_distance_thresh['below'])

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

def getRelativeHeading(feature):

    # a = other boat's previous location
    a = [feature[otherBoat_x] - feature[otherBoat_vx]  , feature[otherBoat_y] - feature[otherBoat_vy]]
    
    # b = own boat's previous location
    b = [feature[ownBoat_x]   - feature[ownBoat_vx]    , feature[ownBoat_y] - feature[ownBoat_vy]    ]

    # c = own boat's current location
    c = [feature[ownBoat_x]                            , feature[ownBoat_y]                          ]

    # d = other boat's current location
    d = [feature[otherBoat_x]                          , feature[otherBoat_y]                        ]

    # e = our next location
    e = [feature[ownBoat_x]   + feature[ownBoat_vx]    , feature[ownBoat_y] + feature[ownBoat_vy]    ]

    # Create some unit vectors
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    cd = [d[0] - c[0], d[1] - c[1]]
    ce = [e[0] - c[0], e[1] - c[1]] 

    for vector in [ba, bc, cd, ce]:
        magnitude = math.sqrt(vector[0]*vector[0] + vector[1]*vector[1])

        if magnitude > 0.0:
            vector[0] /= magnitude
            vector[1] /= magnitude

    previous_angle  = math.atan2(ba[1], ba[0]) - math.atan2(bc[1], bc[0])
    current_angle   = math.atan2(cd[1], cd[0]) - math.atan2(ce[1], ce[0])

    return current_angle - previous_angle

def getCpaDistanceDelta(feature):
    return float(feature[cpa_dDist])

def getCpaTimeValue(feature):
    return float(feature[cpa_vTime])

def collapseObservations():
    collapsedObservations = dict()
    counter = 0

    # TODO: This has gotten out of hand
    # make this a function
    for k1, v1 in relative_angle.iteritems():
        for k2, v2 in delta_location.iteritems():
            for k3, v3 in delta_speed.iteritems():
                for k4, v4 in delta_angle.iteritems():
                    for k5, v5 in delta_relative_heading.iteritems():
                        #for k6, v6 in cpa_delta_distance.iteritems():
                        for k7, v7 in cpa_time.iteritems():
                            for k8, v8 in cpa_distance_thresh.iteritems():
                                counter += 1
                                collapsedObservations[str(v1)+str(v2)+str(v3)+str(v4)+str(v5)+str(v7)+str(v8)] = counter

    return collapsedObservations

# Train HMM
def train(inputFileDir, outputFileName, numStates):

    # Initialize variables
    collapsedObservations = dict()
    allPossibleSymbols = []

    # List of all possible discrete observations
    collapsedObservations = collapseObservations()

    for k, v in collapsedObservations.iteritems():
        allPossibleSymbols.append(v)

    # Initialize HMM
    #TODO try different parameters
    HMM = hmm.GaussianHMM(n_components=numStates, n_iter=1000)

    # Get file names from training directory
    filenames = getFilesFromDir(inputFileDir)

    # iterate over all of our training files
    for f in filenames:

        # Debug print statment
        print "Training with file: ", f

        #if '69' in f:
        #    raw_input()

        # More iterations here
        observed_fixed = []
        observedSymbols, features = readInputFile(f)

        # Convert list of symbols to unique interger identifier
        for s in observedSymbols:
            observed_fixed.append(collapsedObservations[str(s[0])+str(s[1])+str(s[2])+str(s[3])+str(s[4])+str(s[5])+str(s[6])])

        # Bin data
        split = binData(observed_fixed)

        # Debug print for all observed sequences
        #for s in split:
        #    print s
        

        # Update HMM
        HMM.fit(split)

    # Test for good HMM
    # This will fail if there's a problem
    a = np.array([28] * dataLen).reshape(1,-1)
    p = HMM.score(a)

    # Save HMM to disk
    saveHmm(outputFileName, HMM) 

def binData(observed_symbols):

    # Initialize Variables
    split = []

    # Bin data in "Rolling Window" method
    for i in range(dataLen, len(observed_symbols)):
        split.append(observed_symbols[(i - dataLen): i])

    return split

def saveHmm(outputFileName, hmm) :
    with open(outputFileName, 'w+') as f:
        pickle.dump(hmm,f)

def loadHmms():

    hmms = []

    # Load HMMs from file
    for filename in all_hmms:
        try:
            with open(filename) as f:
                hmms.append(pickle.load(f))
        except EnvironmentError:
            killWithMessage('There is a problem with the input files')

    return hmms

# Test HMM
def test(inputFileDir):

    # Load HMM data from file
    HMMs = loadHmms()

    # Get file names from training directory
    filenames = getFilesFromDir(inputFileDir)

    # List of all possible discrete observations
    collapsedObservations = collapseObservations()

    # iterate over all of our training files
    for f in filenames:

        # Debug print statment
        print "Testing with file: ", f

        # Initialize Variables
        probs = []
        
        # More iterations here
        observed_fixed = []
        observedSymbols, features = readInputFile(f)

        # Convert list of symbols to unique interger identifier
        for s in observedSymbols:
            observed_fixed.append(collapsedObservations[str(s[0])+str(s[1])+str(s[2])+str(s[3])+str(s[4])+str(s[5])+str(s[6])])

        # Bin data
        split = binData(observed_fixed)

        # Generate test data
        for line in split:

            scores = []
            prob = []
            totalSum = 0

            # Test against all available hmms
            for HMM in HMMs:
                scores.append(HMM.score(np.array(line).reshape(1,-1)))

            # Figure out the scaling factor
            for score in scores:
                totalSum += math.exp(score)

            # Determine probability for each hmm class
            for score in scores:
                prob.append(math.exp(score) / totalSum)

            probs.append(prob)

        # Save to file:
        createOutputReport(f, probs, features)

def createOutputReport(filename, probs, symbols):

    # Initialize Variables
    header = "timestamp, source_id, contact_id, contact_type, intent_type, group_id, intent_class, intent_prob, group_confidence\n"
    index = dataLen
    originalIndex = 0
    intentNum = 0
    skipRow = False
    
    # Get save location
    parentDir =  os.path.dirname(filename)

    # Also open original file
    with open(filename, 'r') as originalFile:

        # Open as a .csv
        tempCSV = csv.reader(originalFile, delimiter=',')  
        originalCSV = []
        for row in tempCSV:
            originalCSV.append(row)

        # Begin writting report
        with open(parentDir + '/probability_report.csv', 'w+') as f:
            
            # Output header
            f.write(header)  

            # ID rows where there isn't enough data
            while originalCSV[originalIndex][0][0] == '#':

                # Output placeholder record
                for intent_index in all_intents:

                    intent_probability = '0'
                    if intent_index == '8':
                        intent_probability = '1' # we consider this special case 'unknown'

                    f.write(originalCSV[originalIndex][1] + ",1,99,99,0,99," + intent_index + ',' + intent_probability + ",0\n")

                originalIndex += 1

            # Then on to rows that didn't have enough history
            for i in range(0, dataLen):

                # Output placeholder record
                for intent_index in all_intents:

                    intent_probability = '0'
                    if intent_index == '8':
                        intent_probability = '1' # we consider this special case 'unknown'
                    
                    # This need to be updated to include an unknown data tag
                    # TODO: Fix this negative
                    if originalCSV[originalIndex][0][0] == '#':
                        f.write(str(originalCSV[originalIndex][1]) + ",1,99,99,0,99," + intent_index + ',' + intent_probability + ",0\n")
                    else:
                        f.write(str(originalCSV[originalIndex][16]) + ",1,99,99,0,99," + intent_index + ',' + intent_probability + ",0\n")

                originalIndex += 1

            #TODO(alex): There needs to be logic here for dealing with missing/unknown intents
            #            Unknown data being close to an even probability across all categories.

            for prob in probs:

                # Take care of the unknown and missing here 
                f.write(str(symbols[index][timeStamp]) + ",1,99,99,0,99,0,0,0\n")
                f.write(str(symbols[index][timeStamp]) + ",1,99,99,0,99,8,0,0\n")

                # This need to be updated to include an unknown data tag

                for p in prob:
                    # Timestamp
                    f.write(str(symbols[index][timeStamp]) + ',')
                    
                    # Some constants
                    f.write("1,")

                    # Contact ID + more constants
                    f.write(str(symbols[index][contact_id]))
                    f.write(",99,0,99,")
                    
                    # Intent type
                    f.write(intents[intentNum])

                    # Constants
                    f.write(",") 

                    # Probability
                    f.write(str(p) + ',')

                    # More constants
                    f.write("0\n")

                    # Increast intent number
                    intentNum += 1
                    intentNum %= len(intents)

                # Increase index
                originalIndex += 1
                index += 1

# Main
if __name__ == '__main__':

    # Parse input arguments
    parser = argparse.ArgumentParser(description='Test or train HMM')
    parser.add_argument('input_file_dir', help='The .csv input file directory for training/testing data')
    parser.add_argument('mode', help='Either "train" or "test"')
    parser.add_argument('--outName', help='Name of HMM to be trained')
    parser.add_argument('--numStates', help='The desired number of hidden states')
    args = parser.parse_args()

    # Test with input files, and produce output hmm
    if args.mode == 'train':
        numStates = 5 
        outName = 'hmm_defaultName'

        if args.numStates and args.numStates in rang(2,6):
            numStates = int(args.numStates)

        if args.outName:
            outName = args.outName

        train(args.input_file_dir, outName, numStates) 

    # Train with input files, and product output probabilities
    elif args.mode == 'test':
        test(args.input_file_dir) 

    # Something has gone wrong
    else:
        killProgramWithMessage('Argument Error: Invalid Mode')


