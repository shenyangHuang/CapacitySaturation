#this file stores parameter templates


#training hyperparameter and search parameters
#standard params
p_s = {
	"lr": 0.0001,
	"epochs": 1000,
	"patience": 10,
	"data_augmentation": False,
	"batch_size": 128,
	"discrete_levels": (16,32,64,96,112,128,160,192,224,256,272,320,
    	384,448,512,576,640,704,768,832,896,960,1024),
	"entropy_penalty": 0.01,
	"RL_lr": 0.001,
	"max_duplicate": 3,
	"N_workers": 1,
	"epoch_limit": 20,
	"max_layers": 50,
	"num_input": 4,
	"fc_limit": 3
}











































