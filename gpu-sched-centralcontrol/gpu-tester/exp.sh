#!/bin/bash
# --c 1 means use controller, --j 2 means run job 0 and job 2 
# --c 0 means no controller, --j 1 means run job 0 and job 1 
# --c 2 means no controller, run job 0 exclusively first and then job 1 exclusively 
while getopts c:j: flag
do
    case "${flag}" in
        c) control=${OPTARG};;
	j) setup=${OPTARG};;
    esac
done
echo "control: $control"
echo "setup: $setup"
if [[ $control == 1 ]] 
then 
	if [[ $setup == 1 ]]
	then
		# run job 0 w control
		./run_test_nsight_control_0.sh &
		# run job 1 w control
		./run_test_nsight_control_1.sh
		#rename output files
		mv out_0.txt out_0_control.txt
		mv out_1.txt out_1_control.txt
	elif [[ $setup == 2 ]]
	then 
		# run job 0 w control
		./run_test_nsight_control_0.sh &
		# run job 2 w control
		./run_test_nsight_control_2.sh
		mv out_0.txt out_0_control.txt
		mv out_2.txt out_2_control.txt
	fi
elif [[ $control == 0 ]] 
then 
	if [[ $setup == 1 ]]
	then
		# run job 0
		./run_test_nsight_0.sh &
		# run job 1 
		./run_test_nsight_1.sh 
		#rename output files
		mv out_0.txt out_0_share.txt
		mv out_1.txt out_1_share.txt
	elif [[ $setup == 2 ]]
	then
		# run job 0  
		./run_test_nsight_0.sh &
		# run job 2
		./run_test_nsight_2.sh 
		#rename output files
		mv out_0.txt out_0_share.txt
		mv out_2.txt out_2_share.txt
	fi
elif [[ $control == 2 ]] 
then 
	# run job 0 exclusively 
	./run_test_nsight_0.sh 
	# run job 1 exclusively
	./run_test_nsight_1.sh 
	#rename output files
	mv out_0.txt out_0_exclusive.txt
	mv out_1.txt out_1_exclusive.txt
fi
