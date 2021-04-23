# Starter code for CS 165B HW2 Spring 2019
import numpy as np
import numpy.linalg as npla

def unpack_input(data):
    class_count = data[0][0]
    instance_counts = np.array(data[0][1:])
    parsed_data = np.asarray(data[1:])

    return [class_count, instance_counts, parsed_data]
    

def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.
    You are permitted to use the numpy library but you must write
    your own code for the linear classifier.

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED

        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """
    # metadata
    class_count = training_input[0][0]


    # TRAINING ========================================================================

    #training dataset
    training_dataset = unpack_input(training_input)
    tr_instance_counts = training_dataset[1]
    tr_data = training_dataset[2]

    # calculate centroids
    centroids = []
    for c in range(class_count):
        begin = tr_instance_counts[:c].sum()
        end = tr_instance_counts[:(c + 1)].sum()

        centroid = np.mean(tr_data[begin:end], axis=0)
        centroids.append(centroid)

    centroids = np.asarray(centroids)


    # TESTING ========================================================================

    # testing dataset
    testing_dataset = unpack_input(testing_input)
    te_instance_counts = testing_dataset[1]
    te_data = testing_dataset[2]

    c_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    # test data
    count = 0
    for c in range(class_count):
        begin = te_instance_counts[:c].sum()
        end = te_instance_counts[:(c + 1)].sum() - 1
        for i in range(begin, end+1):
            count += 1
            # test one instance, true answer is c
            instance = np.asarray(te_data[i])
            predicted_class = 0
            shortest_dist = 999999999
            
            # get shortest distance
            for j in range(len(centroids)):
                centroid = centroids[j]
                dist = npla.norm(centroid - instance)
                if dist < shortest_dist:
                    shortest_dist = dist
                    predicted_class = j
            
            # update confusion matrix
            c_matrix[predicted_class][c] += 1

    # info for stats
    total = np.sum(c_matrix)
    stats = np.zeros((3,5))

    # calculate stats
    for c in range(class_count):
        tp = c_matrix[c,c]
        fp = c_matrix[c,1 if c == 0 else 0] + c_matrix[c,1 if c == 2 else 2]
        fn = c_matrix[1 if c == 0 else 0,c] + c_matrix[1 if c == 2 else 2,c]
        tn = total - tp - fp - fn

        pos = tp + fn
        neg = fp + tn

        stats[c,0] = tp / pos
        stats[c,1] = fp / neg
        stats[c,2] = (fp + fn) / total
        stats[c,3] = (tp + tn) / total
        stats[c,4] = tp / (tp + fp)
         
    # package stats
    avg_stats = np.mean(stats, axis=0)
    tpr = avg_stats[0]
    fpr = avg_stats[1]
    error_rate = avg_stats[2]
    accuracy = avg_stats[3]
    precision = avg_stats[4]

    result = {
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
        "accuracy": accuracy,
        "precision": precision
    }

    return result
