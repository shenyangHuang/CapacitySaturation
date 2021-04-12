'''
Use to measure k-ary classification accuracy for selected group of classes, the prediction from other classes will simply ignored 
'''

'''
predictions = softmax probability prediction by the NN for all classes 
true_labels = accuracy label for each sample predicted (classes not in classList will simply be ignored)
classList = list of classes that are being investigated (index starts from 0)
'''
def knaryAccuracy(predictions, true_labels, classList):
    
    predictions = predictions.tolist()
    true_labels = true_labels.tolist()
    total_counts = [0] * len(classList)
    correct_counts = [0] * len(classList)

    for i in range(0, len(predictions)):
        yTrue = true_labels[i].index(max(true_labels[i]))
        if (yTrue not in classList):        #ignore any classes that are not specified 
            continue
        else:
            probs = [predictions[i][x] for x in classList]
            yProbs = predictions[i].index(max(probs))
            Count_idx = classList.index(yTrue)
            total_counts[Count_idx] = total_counts[Count_idx] + 1
            if(yProbs == yTrue):
                correct_counts[Count_idx] = correct_counts[Count_idx] + 1

    print ("k-ary classification accuracy")
    for i in range(0, len(classList)):
        print ("classs " + str(classList[i]+1) + " is correct for " + str(correct_counts[i]) + " / " + str(total_counts[i]))
        print ("thus class " + str(classList[i]+1) + " has accuracy " + str(correct_counts[i]/total_counts[i]))
    































































