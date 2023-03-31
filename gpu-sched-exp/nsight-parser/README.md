## To Setup the nsight parser's experiment 
0. You have already setup conda environment and replace transform.py file in torchvision with `gpu-core-exps/transform.py`. The path of the file that should be replaced is under `/home/USERNAME/miniconda3/envs/torch/lib/python3.9/site-packages/torchvision/models/detection/transform.py`

1. Activate conda environment, for this script is using matplotlib and numpy.


## To run the nsight parser
0. Under `gpu-sched-exp/gpu-tester/`, run the experiment with Nsight report. Copy `nsight_report.nsys-rep` and `models_pid.json` from there to this directory.

1. Turn the `nsight_report.nsys-rep` into csv file: <br /> `nsys stats -r kernexectrace --format csv -o . nsight_reportme.nsys-rep`\
It will generate `nsight_report_kernexectrace.csv` as output.

2. Parse the `nsight_report_kernexectrace.csv` and `models_pid.json ` using `parseNsysRep.py`:<br />
`python parseNsysRep.py -f nsight_report_kernexectrace.csv -p models_pid.json`\
Result summary logs, per model kernel timelines csv, and kernel execution time CDFs are stored under `nsight_report_kernexectrace.csv_TIMESTAMP` directory.