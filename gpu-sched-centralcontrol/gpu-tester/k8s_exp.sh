#!/bin/bash
# --c 1 means use controller, --j 2 means run job 0 on pod gpucontrol and job 2 on gpucontrol2
# --c 0 means no controller, --j 1 means run job 0 on pod gpucontrol and job 1 on gpucontrol2
# --c 2 means no controller, run job 0 on pod gpucontrol exclusively first and then job 1 exclusively on gpucontrol2
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
		# run job 0 on pod gpucontrol w control
		kubectl exec gpucontrol -- bash -c "cd src/gpu-tester && ./run_test_nsight_control_0.sh" &
		# run job 1 on pod gpucontrol2 w control
		kubectl exec gpucontrol2 -- bash -c "cd src/gpu-tester && ./run_test_nsight_control_1.sh"
	elif [[ $setup == 2 ]]
	then 
		# run job 0 on pod gpucontrol w control
		kubectl exec gpucontrol -- bash -c "cd src/gpu-tester && ./run_test_nsight_control_0.sh" &
		# run job 2 on pod gpucontrol2 w control
		kubectl exec gpucontrol2 -- bash -c "cd src/gpu-tester && ./run_test_nsight_control_2.sh"
	fi
elif [[ $control == 0 ]] 
then 
	if [[ $setup == 1 ]]
	then
		# run job 0 on pod gpucontrol
		kubectl exec gpucontrol -- bash -c "cd src/gpu-tester && ./run_test_nsight_0.sh" &
		# run job 1 on pod gpucontrol2 
		kubectl exec gpucontrol2 -- bash -c "cd src/gpu-tester && ./run_test_nsight_1.sh" 
	elif [[ $setup == 2 ]]
	then
		# run job 0 on pod gpucontrol 
		kubectl exec gpucontrol -- bash -c "cd src/gpu-tester && ./run_test_nsight_0.sh" &
		# run job 2 on pod gpucontrol2
		kubectl exec gpucontrol2 -- bash -c "cd src/gpu-tester && ./run_test_nsight_2.sh" 
	fi
elif [[ $control == 2 ]] 
then 
	# run job 0 on pod gpucontrol exclusively 
	kubectl exec gpucontrol -- bash -c "cd src/gpu-tester && ./run_test_nsight_0.sh" 
	# run job 1 on pod gpucontrol2 exclusively
	kubectl exec gpucontrol2 -- bash -c "cd src/gpu-tester && ./run_test_nsight_1.sh" 
fi
