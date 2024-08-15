

def main():
    position = 0
    total_time = 0
    max_time = 0
    min_time = 300
    success = 1e-6
    total_success = 0
    path = "D:\\CODE\\Python\\AirSim\\test_result.txt"
    with open(path,"r") as f:
        context = f.readline()
        while context:
            if context.startswith("-----"):
                position+=1
                if position == 1 :
                    context = f.readline()
                    continue
                mean_time = total_time/success
                success = success/15
                print("Position : {0} , Success : {1:.0%} , mean : {2}, max :{3}, min:{4}".format(position-1,success,mean_time,max_time,min_time))
                total_time = 0
                max_time = 0
                min_time = 300
                success = 1e-6
            elif context.startswith("[+]"):
                context = f.readline()
                continue
            else:
                episode_time , result = context.split(",")
                time = episode_time.split(":")[1]
                time = float(time)
                #print(time)
                terminate = result.split(":")[1].strip()
                #print(terminate)
                if terminate == "Success landing":
                    success+=1
                    total_success+=1
                    if time > max_time:
                        max_time = time
                    if time < min_time:
                        min_time = time
                    total_time += time
            context = f.readline()
        print("total success rate : " ,total_success/150)
main()