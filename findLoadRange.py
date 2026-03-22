#this Python file runs a series of load flow studies across 
#various MW and PF values to see which cases result in bus 5 voltage > 0.97pu
#------------------------------------------------------------------------------------

#libraries
import itertools
import csv 
import math
import time
import psspy

start_time = time.perf_counter()

#creating cases
variables = ["load size","power factor"]
values = [
    [int(num) for num in list(range(10,21,1))], #testing different MW values with the diff PFs below to see which systems result in bus 5 voltage > 0.97
    [PF/100 for PF in list(range(90,101))] #PF from 0.9 to 1.0, inclusive
]

cases = [dict(zip(variables, values)) for values in itertools.product(*values)] 
print(len(cases)) #should be 27
print(cases)

psspy.psseinit(50) #initializes environment for up to 50 buses
psspy.case(r"C:\Users\EmmaY\Documents\Eng + Ivey Year 5\Capstone\Dx Feeder Studies\PSSE Cases\Dx_Feeder_V1.sav") #opens case called "Dx_Feeder_V1.sav"

#STEPS: iterate through every case and do the following:
#1) calculate Q corresponding to P and PF
#2) enter the values for Q and P into Qload and Pload
#3) run the load flow
    # -if doesn't converge, will include
#4) export the following data into CSV:
    # -Load MW
    # -Load MVar
    # -Total load MVA
    # -PF
    # -Bus 5 voltage
    
with open('load_range_results.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Load MW', 'Load Mvar', 'Load MVA', 'PF', 'Bus 5 V (pu)', 'Bus 5 angle']) #header
        
    for case in cases:
        PF = case["power factor"]
        P = case["load size"] #active power
        Q = P*math.tan(math.acos(PF))
        S = math.sqrt(P*P+Q*Q)
        
        #change the Pload and Qload parameters in PSSE (constant P and constant Q fields)
        psspy.load_chng_7(5,r"""L1""",[_i,_i,_i,_i,_i,_i,_i],[P,Q,0,0,0,0,0,0],_s,_s)

        solution = psspy.fnsl([0,0,0,1,1,0,99,0]) #runs the load flow study
        
        #results from PSSE load flow
        vpu_err, vpu = psspy.abusreal(-1,1,'PU')
        vang_err, vang_rad = psspy.abusreal(-1,1,'ANGLE')
        vang_deg = [float(ang) * 57.29578 for ang in vang_rad[0]] #converting from radians to degrees

        
        print(vpu[0])
        print(vang_deg)
        
        if solution !=0: #solution did NOT converge
            writer.writerow(['NA','NA','NA','NA','NA','NA'])
        else: #solution DID converge
            writer.writerow([P,Q,S,PF,vpu[0][4],vang_deg[4]])

            
end_time = time.perf_counter()
program_time = end_time - start_time
print(program_time)
