#!/bin/bash
while getopts c:j: flag
do
    case "${flag}" in
        c) control=${OPTARG};;
        j) setup=${OPTARG};;
    esac
done
echo "control: $control"
echo "setup: $setup"

if [[ $control == 0 ]]
then
	rm out_0_share.txt
	rm out_1_share.txt
	kubectl cp gpucontrol:/src/gpu-tester/out_0.txt out_0_share.txt &
	kubectl cp gpucontrol2:/src/gpu-tester/out_1.txt out_1_share.txt &
elif [[ $control == 1 ]]
then
	rm out_0_control.txt
	rm out_1_control.txt
	kubectl cp gpucontrol:/src/gpu-tester/out_0.txt out_0_control.txt &
	kubectl cp gpucontrol2:/src/gpu-tester/out_1.txt out_1_control.txt &
elif [[ $control == 2 ]]
then
	rm out_0_exclusive.txt
	rm out_1_exclusive.txt
	kubectl cp gpucontrol:/src/gpu-tester/out_0.txt out_0_exclusive.txt &
	kubectl cp gpucontrol2:/src/gpu-tester/out_1.txt out_1_exclusive.txt &
fi
