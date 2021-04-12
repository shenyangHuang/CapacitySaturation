#this file stores architecture templates 

from Model.Layer_obj import Layer_obj



'''
architecture found by random search on 200 architectures
'''
'''
conv-3-320
maxpool
dropout-0.250000
conv-3-128
maxpool
dropout-0.500000
conv-3-224
maxpool
dropout-0.250000
conv-3-128
maxpool
dropout-0.250000
fc-576
dropout-0.100000
'''

def CNAS_start_arch():
    teacher_architecture = []
    teacher_architecture.append(Layer_obj([3,3,3,320],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))

    teacher_architecture.append(Layer_obj([3,3,320,128],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.5],'dropout'))

    teacher_architecture.append(Layer_obj([3,3,128,224],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))

    teacher_architecture.append(Layer_obj([3,3,224,128],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))

    teacher_architecture.append(Layer_obj([1024,576],'fc'))
    teacher_architecture.append(Layer_obj([0.1],'dropout'))
    return teacher_architecture



# # conv-3-272
# # maxpool
# # dropout-0.100000
# # conv-3-96
# # maxpool
# # dropout-0.500000
# # fc-896
# # fc-448

# def CNAS_2class_start_arch():
#     teacher_architecture = []
#     teacher_architecture.append(Layer_obj([3,3,3,272],'conv'))
#     teacher_architecture.append(Layer_obj([2],'maxpool'))
#     teacher_architecture.append(Layer_obj([0.1],'dropout'))

#     teacher_architecture.append(Layer_obj([3,3,272,96],'conv'))
#     teacher_architecture.append(Layer_obj([2],'maxpool'))
#     teacher_architecture.append(Layer_obj([0.5],'dropout'))

#     teacher_architecture.append(Layer_obj([1024,896],'fc'))
#     teacher_architecture.append(Layer_obj([896,448],'fc'))
#     return teacher_architecture



# conv-3-256
# maxpool
# dropout-0.250000
# conv-3-112
# maxpool
# dropout-0.250000
# conv-3-160
# conv-3-448
# maxpool
# dropout-0.100000
# fc-256
# fc-64
# fc-640
# fc-64
# fc-128
# dropout-0.100000
def CNAS_2class_start_arch():
    teacher_architecture = []
    teacher_architecture.append(Layer_obj([3,3,3,256],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))

    teacher_architecture.append(Layer_obj([3,3,256,112],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))

    teacher_architecture.append(Layer_obj([3,3,112,160],'conv'))
    teacher_architecture.append(Layer_obj([3,3,160,448],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.1],'dropout'))

    teacher_architecture.append(Layer_obj([1024,256],'fc'))
    teacher_architecture.append(Layer_obj([256,64],'fc'))
    teacher_architecture.append(Layer_obj([64,640],'fc'))
    teacher_architecture.append(Layer_obj([640,64],'fc'))
    teacher_architecture.append(Layer_obj([64,128],'fc'))
    teacher_architecture.append(Layer_obj([0.1],'dropout'))
    return teacher_architecture




'''
conv-3-368
maxpool
dropout-0.100000
conv-3-480
maxpool
dropout-0.500000
conv-3-144
maxpool
dropout-0.100000
Fc-704
'''

def CNAS_5class_start_arch():
    teacher_architecture = []
    teacher_architecture.append(Layer_obj([3,3,3,368],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.1],'dropout'))

    teacher_architecture.append(Layer_obj([3,3,368,480],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.5],'dropout'))

    teacher_architecture.append(Layer_obj([3,3,480,144],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.1],'dropout'))

    teacher_architecture.append(Layer_obj([1024,704],'fc'))
    return teacher_architecture






'''
architecture found by random search on 200 architectures
'''
'''
conv-3-320
maxpool
dropout-0.250000
conv-3-128
maxpool
dropout-0.500000
conv-3-224
maxpool
dropout-0.250000
conv-3-128
maxpool
dropout-0.250000
fc-576
dropout-0.100000
'''

def SA_10class():
    teacher_architecture = []
    teacher_architecture.append(Layer_obj([3,3,3,320],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))

    teacher_architecture.append(Layer_obj([3,3,320,128],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.5],'dropout'))

    teacher_architecture.append(Layer_obj([3,3,128,224],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))

    teacher_architecture.append(Layer_obj([3,3,224,128],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))

    teacher_architecture.append(Layer_obj([1024,576],'fc'))
    teacher_architecture.append(Layer_obj([0.1],'dropout'))
    return teacher_architecture


def SA_small():
    teacher_architecture = []
    teacher_architecture.append(Layer_obj([3,3,3,16],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))

    teacher_architecture.append(Layer_obj([3,3,3,16],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))

    teacher_architecture.append(Layer_obj([3,3,64,16],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))

    teacher_architecture.append(Layer_obj([3,3,3,16],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))

    teacher_architecture.append(Layer_obj([1024,112],'fc'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))
    return teacher_architecture



# def SA_small():
#     teacher_architecture = []
#     teacher_architecture.append(Layer_obj([3,3,3,16],'conv'))
#     teacher_architecture.append(Layer_obj([2],'maxpool'))
#     teacher_architecture.append(Layer_obj([0.25],'dropout'))

#     teacher_architecture.append(Layer_obj([3,3,64,16],'conv'))
#     teacher_architecture.append(Layer_obj([2],'maxpool'))
#     teacher_architecture.append(Layer_obj([0.25],'dropout'))

#     teacher_architecture.append(Layer_obj([1024,112],'fc'))
#     teacher_architecture.append(Layer_obj([0.25],'dropout'))
#     return teacher_architecture








def CNAS_old_start_arch():
    teacher_architecture = []
    teacher_architecture.append(Layer_obj([3,3,3,128],'conv'))
    teacher_architecture.append(Layer_obj([3,3,128,128],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.1],'dropout'))
    teacher_architecture.append(Layer_obj([3,3,64,128],'conv'))
    teacher_architecture.append(Layer_obj([3,3,64,128],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))
    teacher_architecture.append(Layer_obj([3,3,128,256],'conv'))
    teacher_architecture.append(Layer_obj([3,3,256,256],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.5],'dropout'))
    teacher_architecture.append(Layer_obj([1024,256],'fc'))
    teacher_architecture.append(Layer_obj([0.5],'dropout'))
    return teacher_architecture



def old_static_arch():
    teacher_architecture = []
    teacher_architecture.append(Layer_obj([3,3,3,128],'conv'))
    teacher_architecture.append(Layer_obj([3,3,128,128],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.1],'dropout'))
    teacher_architecture.append(Layer_obj([3,3,128,256],'conv'))
    teacher_architecture.append(Layer_obj([3,3,256,256],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))
    teacher_architecture.append(Layer_obj([3,3,256,512],'conv'))
    teacher_architecture.append(Layer_obj([3,3,512,512],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.5],'dropout'))
    teacher_architecture.append(Layer_obj([1024,512],'fc'))
    teacher_architecture.append(Layer_obj([0.5],'dropout'))
    return teacher_architecture


















































