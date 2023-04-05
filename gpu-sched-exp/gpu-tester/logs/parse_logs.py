import statistics
import math


sync_list_p0 = [1, 4, 10, 20]
for sync in sync_list_p0:
    print("--------------------------------------------")
    print(f"control + sync, sync = {sync}")
    filename_model0 = f"model_A_sync{sync}.log"
    filename_model1 = f"model_B_sync{sync}.log"

    jct = []
    model = "model1"
    size = "[720, 1280]"
    f = open(filename_model1, 'r')
    lines = f.readlines()
    # Strips the newline character
    for line in lines:
        line = line.rstrip('\n').split(" ")
        # print(line)
        if len(line) > 4 and line[4] == "\"model_name\":":
            model = line[5]
        if (len(line) > 4 and line[4] == "\"resize\":") and line[5] == "true,":
            size = f"[{lines[i+2].strip()} {lines[i+3].strip()}]"
        if (line[0] == "JCT" and len(line) == 4) and float(line[3]) < 1000:
            jct.append(float(line[3]))
    print(f"{model} {size} | avg JCT = {sum(jct) / len(jct)} | SEM = {statistics.pstdev(jct) / math.sqrt(len(jct))}")
    f.close()


    jct = []
    model = "model0"
    size = "[720, 1280]"
    f = open(filename_model0, 'r')
    lines = f.readlines()
    # Strips the newline character
    for i, line in enumerate(lines):
        line = line.rstrip('\n').split(" ")
        # print(line)
        if len(line) > 4 and line[4] == "\"model_name\":":
            model = line[5]
        if (len(line) > 4 and line[4] == "\"resize\":") and line[5] == "true,":
            size = f"[{lines[i+2].strip()} {lines[i+3].strip()}]"
        if (line[0] == "JCT" and len(line) == 4) and float(line[3]) < 1000:
            jct.append(float(line[3]))
    
    print(f"{model} {size} | avg JCT = {sum(jct) / len(jct)}  | SEM = {statistics.pstdev(jct) / math.sqrt(len(jct))}")
    f.close()

    
            

