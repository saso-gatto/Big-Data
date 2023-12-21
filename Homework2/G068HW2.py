import InputCode 
import sys
import random as rand
import time

# This function returns the list of indices of the points belonging to a cluster (ball Bz) with center x
# and radius given. 
def define_bz(x,radius,Z,dist_matrix,index_map):
    bz_indices = []
    for i in range(len(Z)): # loops through the uncovered points
        distance = dist_matrix[index_map.get(x)][index_map.get(Z[i])]
        if (distance<=radius): #if point belongs to the cluster
            bz_indices.append(i) # we take the index
    return bz_indices

# This function returns the weight of a ball of points of center x and radius given.
def compute_weight (x,radius,Z,dist_matrix,weights,index_map):
    ball_weight=0
    for y in Z: #loops through uncovered points
        distance = dist_matrix[index_map.get(x)][index_map.get(y)]
        if (distance<=radius): #if point belongs to the cluster
            ball_weight+= weights[index_map.get(y)] #adds weight of the point 
    return ball_weight

# Function to initialize our main data structures: 
# - dist_matrix: contains the distances of all inputPoints
# - index_map: dictionary to easily access index of a point (key= point: value = index)
# - computes min_distance: the minimum distance among the first k+z+1 points, which will be our first guess for radius.
def initialize_data_structures(inputPoints,k,z):
    dist_matrix =[[0 for x in range(len(inputPoints))] for y in range(len(inputPoints))]  # initialize distance matrix
    min_distance=1e6 # we first set this variable to a high value
    index_map = {} #dictionary

    # In the following loops we initialize the dictionary index_map, the distance matrix and find the minimum distance 
    # of the first k+z+1 points
    for i in range(len(inputPoints)):
        index_map[inputPoints[i]] = i
        for j in range(i,len(inputPoints)): # matrix is symmetric
            distance = InputCode.euclidean(inputPoints[i],inputPoints[j])
            dist_matrix[i][j] =distance
            dist_matrix[j][i] =distance
            if(i< k+z+1 and j < k+z+1 and i!=j):
                if(distance < min_distance):
                    min_distance = distance
    return dist_matrix,index_map,min_distance

def SeqWeightedOutliers(inputPoints,weights,k,z, alpha =0):

    dist_matrix,index_map,min_distance = initialize_data_structures(inputPoints,k,z)
    
    initial_guess= min_distance/2 # radius
    r = initial_guess
    num_iterations = 1
    while (True):
        Z = inputPoints.copy() #contains uncovered points
        solution = [] # contains cluster centers
        Wz = sum(weights)
        while (len(solution)<k and Wz>0):
            val_max = 0
            for x in Z:
                ball_weight=0
                radius = (1+2*alpha)*r
                ball_weight = compute_weight(x,radius,Z,dist_matrix,weights,index_map)
                if (ball_weight > val_max):
                    val_max=ball_weight
                    new_center = x
            solution.append(new_center)

            radius = (3+4*alpha)*r
            # We delete from Z the points belonging to the cluster with center new_center and radius (3+4*alpha)*r
            # and remove their weights from Wz. The function define_bz returns the indices of points in the ball,
            # and we go through them in reversed order so to safely remove them from Z (from right to left) 
            for index_z in reversed(define_bz(new_center,radius,Z,dist_matrix,index_map)):
                Wz-=weights[index_map.get(Z[index_z])]
                del Z[index_z]


        if(Wz <=z):
            print(f"Input size n = {len(inputPoints)}")
            print(f"Number of centers k = {k}")
            print(f"Number of outliers z = {z}")
            print(f"Initial guess = {initial_guess}")
            print(f"Final guess = {r}")
            print(f"Number of guesses = {num_iterations}")
            return solution
        else:
            num_iterations +=1
            r=2*r #updates radius


def ComputeObjective(inputPoints,solution,z):
    distances = []
    # For each input point, we compute its distance from its cluster center (by taking the minimum distance among
    # all cluster centers).  
    for x in inputPoints: 
        min_d=1e6
        for center in solution: 
            dist = InputCode.euclidean(x,center)
            if (dist<min_d):
                min_d = dist
        distances.append(min_d) #contains the distances of all points from their relative cluster center
    distances.sort(reverse=True) # we sort the list in descending order
    del distances[:z] # we remove the z largest distances

    return (distances[0]) # we return the largest distance 



def main():
        
    # INPUT READING
    # Path to dataset
    data_path = sys.argv[1]
    # Number of centers
    k = int(sys.argv[2])
    # Nunmber of allowed outliers
    z = int(sys.argv[3])

    # Reading inputs from data_path
    inputPoints = InputCode.readVectorsSeq(data_path)
    weights = [1]*len(inputPoints) # unit weights
    start= time.time()
    solution = SeqWeightedOutliers(inputPoints,weights,k,z,0)
    end=time.time()
    running_time = (end-start)*1000 # ms conversion
    objective = ComputeObjective(inputPoints,solution,z)

    print(f"Objective function = {objective}")
    print(f"Time of SeqWeightedOutliers = {running_time}")

    


if __name__ == "__main__":
	main()