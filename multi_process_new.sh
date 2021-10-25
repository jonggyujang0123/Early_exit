#!/bin/bash

sess="Early_exit"

screen -dmS "$sess"

window=0

datasets="
CIFAR10
CIFAR100
EMNIST
MNIST"

models="
AlexNet
LeNet
ResNet"

for dataset in $datasets; do
	for model in $models; do
		if [ $window -ne 0 ]; then
			screen -S "$sess" -X screen $window
		fi
		command_1="conda activate Maritime \n" 
		command_2="CUDA_VISIBLE_DEVICES=$((window/6)) python tools/train.py --dataset $dataset --model $model \n"
		screen -S "$sess" -p $window -X stuff "$command_1"
		screen -S "$sess" -p $window -X stuff "$command_2"
		window=$(($window+1))
	done
done
