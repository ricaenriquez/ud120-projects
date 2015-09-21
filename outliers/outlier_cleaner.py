#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    from operator import itemgetter
    
    cleaned_data = []

    ### your code goes here
    n_points = int(0.9*len(ages))
    for i in range(0, len(net_worths)):
        cleaned_data.append((ages[i][0], net_worths[i][0],
                             abs(predictions[i][0] - net_worths[i][0])))
    cleaned_data = sorted(cleaned_data, key=itemgetter(2))[0:n_points]
    return cleaned_data

