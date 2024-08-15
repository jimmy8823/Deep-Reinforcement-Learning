

total_step = 0

with open("result.txt", "r") as f:
    line = f.readline()
    while line:
        string = line.split(",",5)
        string = string[3]
        string = string.split(":")
        string = string[1]
        step = int(string)
        total_step += step
        line = f.readline()
    
    print(total_step)