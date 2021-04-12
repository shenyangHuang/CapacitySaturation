# Canadian AI 2021
Code for Understanding Capacity Saturation in Incremental Learning


# To run the experiments:
1. Select which model to run, CNAS, SA, RAS
2. move the experiment scripts to the same same directory as README
3. adjust the related parameters in the script

example: in CNAS_2class_resevoir

mem_size = 1000

classes = datasets.Incremental_partition("cifar100", 0.017, order, normalization="numerical")

** ex. Replicate capacity saturation experiment in Experiment Section **

CUDA_VISIBLE_DEVICES=0 python -u CNAS_small_incremental |& tee CNAS_small.txt

CUDA_VISIBLE_DEVICES=0 python -u SA_small_2class |& tee SA_small.txt

Note that each experiment will utilize one GPU (preferrably with large memory)

.txt files stored the output of the experiment

small -- capacity saturation

2class -- 2-class incremental learning

10class -- 10-class incremental learning

mixed -- mixed class incremental learning (appendix)

fraction -- ablation study (appendix)


# Dependencies: (recommend installation with Anaconda) 
Python 3.6.7 :: Anaconda custom (64-bit)

Keras 2.2.4

Tensorflow 1.13.1 (gpu version)

numpy 1.14.2

pickle



