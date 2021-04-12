'''
design different curriculums for training and rehearsing of past data
'''
import numpy as np

'''
randomly pick a fixed number of past data and after a few epochs all data would have been passed through
returns (mask, curriculum, executeNum)
curriculum is used for the function to keep track of which samples have been passed through and thus should not be used
mask is the corresponding mask for that epoch (which is actually needed)
mask contains the index of the samples that should be used for that epoch

when first execute this funciton, pass an all 0 curriculum size of the lenTotal 
executeNum = 0
and executeNum stores which is the current epoch in an epoch cycle 
highestNum stores the length of the epoch cycle
'''
def random_curriculum(lenEpoch, lenTotal, executeNum, curriculum):
	if (lenEpoch > lenTotal):
		#set to use everything in the curriculum
		#print ("warning: you are using all of the dataset")
		lenEpoch = lenTotal
	highestNum = (lenTotal // lenEpoch)

	executeNum = executeNum + 1
	#go through a cycle
	if (executeNum > highestNum or executeNum > np.max(curriculum)):
		executeNum = 0

	if (executeNum == 0):
		#first time execution 
		if (np.count_nonzero(curriculum) == 0):
			phase = 1
			index = 0
			numLeft = lenTotal
			curriculum = np.zeros(lenTotal, dtype=int)
			while(numLeft >= lenEpoch):
				curriculum[index:index+lenEpoch] = phase
				index = index + lenEpoch
				phase = phase + 1
				numLeft = numLeft - lenEpoch
			#the ones that are still 0s are left overs
			np.random.shuffle(curriculum)
			executeNum = 1

		#come in full cycle, need to first use the leftover ones
		else:
			#how many additional element we need for this mask
			leftover = lenEpoch - (lenTotal - np.count_nonzero(curriculum))
			curriculum[curriculum > 0] = -1
			curriculum[curriculum == 0] = 1
			phase = 1
			#print (curriculum)
			(location,) = np.where(curriculum < 0)
			np.random.shuffle(location)
			location = location[0:leftover]
			curriculum[location] = phase
			#print (curriculum)
			curriculum[curriculum == -1] = 0
			numLeft = lenTotal - lenEpoch
			phase = phase + 1
			while(numLeft >= lenEpoch):
				(location,) = np.where(curriculum == 0)
				curriculum[location[0:lenEpoch]] = phase
				numLeft = numLeft - lenEpoch
				phase = phase + 1
			#print (curriculum)
			executeNum = 1

	(mask,) = np.where(curriculum == executeNum)
	return (mask, curriculum, executeNum)

'''
design a random curriculum without replacement
simply return a mask
'''
def random_with_replacement(lenEpoch, lenTotal):
    if (lenEpoch > lenTotal):
        lenEpoch = lenTotal
    curriculum = np.zeros(lenTotal, dtype=int)
    curriculum[0: lenEpoch] = 1
    np.random.shuffle(curriculum)
    
    mask = np.where(curriculum == 1)
    return mask





