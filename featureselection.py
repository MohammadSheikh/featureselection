import math
import pickle as cPickle
import time

# Bunch of the code in this main function was taken from the professor's sample slides since this is just a trivial driver main function.
def driver():
    filename = input("Type in the file name to test: ")
    algorithm_choice = int(input("Welcome to Mahamadsaad's Feature Selection Algorithm. Type '1' to use Forward Selection, or '2' to for Backward Elimination: "))

    # Retrieving only the first line of the file
    with open("C:/Users/saadd/Desktop/CS 170/Project 2/Test_data/" + filename) as f:
        first_row = next(f)
        # Getting the number of features by grabbing the length of the first line by initially splitting the string into a list.
        number_of_features = len(first_row.split()) - 1

    # File closed above so let's open it again.
    file = open("C:/Users/saadd/Desktop/CS 170/Project 2/Test_data/" + filename, 'r')

    # Getting the number of instances by doing a summation of all the lines in the file. (credit to stackoverflow for help)
    number_of_instances = sum(1 for line in file)

    # Preventing an out of bounds error when accessing a list out of range
    file.seek(0)

    # Quick traversing by separating each line of the file into the list itself. (credit to stackoverflow for help)
    instances = [[] for index in range(number_of_instances)]
    for index in range(number_of_instances):
        instances[index] = [float(decim) for decim in file.readline().split()]
    
    print ("This dataset has " + str(number_of_features) + " features, with " + str(number_of_instances) + " instances.")
    
    # Quick traversal - grabbing all the features (i) and making it into a list of features.
    set_feat = []
    for feature in range(1, number_of_features + 1):
        set_feat.append(feature)

    accuracy = leave_one_out_cross_validation(instances, set_feat, number_of_instances)
    print ("Running nearest neighbor with all " + str(number_of_features) + " features, using leaving-one-out evaluation, I get an accuracy of " + str(accuracy) + "%.")

    # Forward Selection
    if algorithm_choice == 1:
        start = time.time()
        forward_selection(instances, number_of_features, number_of_instances)
        end = time.time()
    
    # Backward Elimination
    elif algorithm_choice == 2:
        start = time.time()
        backward_elimination(instances, number_of_features, number_of_instances)
        end = time.time()
    
    print("Time elapsed:", round(end-start,1), "seconds" )

def nearest_neighboring(file, set_of_features, number_of_instances, label_object):
    local_distance = 0
    nearest_neighbor_distance = float('inf')
    nearest_neighbor_location = 0

    for i in range(number_of_instances):
        if i == label_object:
            continue

        local_distance = 0

        # Loop through the amount of features, 
        for j in range(len(set_of_features)):

            # Summation of distance from the object of classification index to the end
            local_distance += pow((file[i][set_of_features[j]] - file[label_object][set_of_features[j]]), 2)

        local_distance = math.sqrt(local_distance)

        #Comparison of neighboring distance and label object to classify. (copied from prof's slides)   
        if local_distance < nearest_neighbor_distance:
            nearest_neighbor_distance = local_distance
            nearest_neighbor_location = i

    return nearest_neighbor_location
def leave_one_out_cross_validation(file, set_of_features, number_of_instances):
    number_correctly_classfied = 0

    # Iterating through amount of instances
    for i in range (number_of_instances):
        label_object = i
        nearest_neighbor_location = nearest_neighboring(file, set_of_features, number_of_instances, label_object)

        # If the label object to classify is equal to the nearest neighbor "label", then add one to the correctly classified number (copied from prof's slides)
        if file[label_object][0] == file[nearest_neighbor_location][0]:
            number_correctly_classfied += 1

    accuracy = number_correctly_classfied / number_of_instances
    return round((accuracy * 100), 1)



def forward_selection(file, number_of_features, number_of_instances):
    best_accurate = 0.0
    set_of_features = []
    after_features_set = []

    for i in range(number_of_features):
        add_features = -1
        updating_accuracy = 0.0
        
        for j in range(1, number_of_features + 1):

            # if isempty(intersect(current_set_of_features,k)) Only consider adding, if not already added. (idea from prof)
            if j not in set_of_features:

                # Much quicker option of copying lists efficiently
                copy_set_of_features = cPickle.loads(cPickle.dumps(set_of_features))
                copy_set_of_features.append(j)
                accuracy = leave_one_out_cross_validation(file, copy_set_of_features, number_of_instances)
                print ("\tUsing feature(s) ", copy_set_of_features, " accuracy is ", accuracy, "%")

               # (copied from prof's slides) 
               # Making sure that the iterated accuracy gets updated and printed below as well.    
                if updating_accuracy < accuracy:
                    updating_accuracy = accuracy
                    add_features = j

        # (copied from prof's slides)  
        set_of_features.append(add_features) 

        if add_features >= 0:
            # Storing the "best" accuracy each time a better one comes up
            if best_accurate < updating_accuracy:
                after_features_set.append(add_features)
                best_accurate = updating_accuracy
                print ("Feature set ", set_of_features, " was best, accuracy is ", updating_accuracy, "%")
            else:
                print ("Warning, Accuracy has decreased! Continuing search in case of local maxima")
                print ("Feature set ", set_of_features, " was best, accuracy is ", updating_accuracy, "%")
    
    print ("Finished search. The best feature subset is", after_features_set, " which has an accuracy of: ", best_accurate, "%")

def backward_elimination(file, number_of_features, number_of_instances):
    best_accurate = 0.0

    # Because of backward elimination, we need to pre-populate our features lists.
    set_of_features = [feature+1 for feature in range(number_of_features)]
    after_features_set = [feature+1 for feature in range(number_of_features)]

    for i in range(number_of_features):
        remove_features = -1
        updating_accuracy = 0.0

        for j in range(1, number_of_features + 1):

            # if isNotempty(intersect(current_set_of_features,k)) Only consider adding, if not already added. (idea from prof)
            if j in set_of_features:
                copy_set_of_features = cPickle.loads(cPickle.dumps(set_of_features))
                copy_set_of_features.remove(j)
                accuracy = leave_one_out_cross_validation(file, copy_set_of_features, number_of_instances)
                print ("\tUsing feature(s) ", copy_set_of_features, " accuracy is ", accuracy, "%")

                # (copied from prof's slides) 
                # Making sure that the iterated accuracy gets updated and printed below as well.  
                if updating_accuracy < accuracy:
                    updating_accuracy = accuracy
                    remove_features = j
                    
        # (copied from prof's slides)   
        set_of_features.remove(remove_features)

        if remove_features > 0:
            # Storing the "best" accuracy each time a better one comes up
            if best_accurate < updating_accuracy:  
                after_features_set.remove(remove_features)
                best_accurate = updating_accuracy 
                print ("Feature set ", set_of_features, " was best, accuracy is ", updating_accuracy, "%")
            else:
                print ("Warning, Accuracy has decreased! Continuing search in case of local maxima")
                print ("Feature set ", set_of_features, " was best, accuracy is ", updating_accuracy, "%")

    print ("Finished search. The best feature subset is", after_features_set, " which has an accuracy of: ", best_accurate, "%")


driver()