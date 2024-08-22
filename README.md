To run the code use this cmd example:
python train_MI.py --dataset cifar100 --dataroot=data/cifar-100-python --model resnet --lossfn ce --num-classes 100 --seed 42 --epochs 400 --weight-decay 0.05  --augment --job_array

This line will train ResNET18 model on CIFAR100 (full width) and will create and save a json file with the train\test metrics and MIA score for every epoch

Most of the code here is originaly from https://github.com/meghdadk/SCRUB