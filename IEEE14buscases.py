#Version 1, Jan 2026
#-----------------------------------------------------------------------------------------------

#libraries
import csv
import psspy
import itertools 
import math
import time

psspy.psseinit(50) #PSSE initializes environment for up to 50 buses
psspy.read(0,r"C:\Users\jingx\Desktop\ieee14busemma.raw") #opens case called "Dx_Feeder_V1.sav"

start_time = time.clock()

zip_coeff_names = ['Pz', 'Pi', 'Pp', 'Qz', 'Qi', 'Qp']
zip_res = [0.055,0.945,0,1.94,0,-0.94]
zip_comm = [0.333, 0.333, 0.333,1.493,0,-0.493]
zip_ind = [0,0.19,0.81,2.92,0,-1.92]

zip1 = dict(zip(zip_coeff_names, zip_res))
zip2 = dict(zip(zip_coeff_names, zip_comm))
zip3 = dict(zip(zip_coeff_names, zip_ind))

from hourly_data import load_percent, very_sunny_Q, mod_sunny_Q, cloudy_Q

peak_MW = { #in pu
    2: [21.7,12.7],
    3: [94.2,19],
    4: [47.8, -1*3.9],
    5: [7.6,1.6],
    6: [11.2, 7.5],
    9: [29.5, 16.6],
    10: [9,5.8],
    11: [3.5,1.8],
    12: [6.1,1.6],
    13: [13.5,5.8],
    14: [14.9,5]
    }
    
print(peak_MW)

        #52.632 - bus 4 since commerical 
        #10.526 - buses 9 and 14 PV farm size since residential
        
PV_size_4 = 105.263
PV_size_914 = 0

with open('IEEE14_load_flow_results.csv','wb') as f:#opening a file called 'Dx_load_flow_results.csv"
    
    writer = csv.writer(f)
    
    #header
    writer.writerow(['hour','load 4 MW', 'load 4 Mvar', 'load 9 MW', 'load 9 Mvar','load 14 MW', 'load 14 Mvar',
                    'sun rating','Q limit 4', 'Q limit 914',
                    'Bus 4 pu', 'Bus 4 angle', 'Bus 9 pu', 'Bus 9 angle', 'Bus 14 pu' , 'Bus 14 angle',
                    #CVR:
                    'load 4 MW', 'load 4 Mvar', 'load 9 MW', 'load 9 Mvar','load 14 MW', 'load 14 Mvar',
                    'Bus 4 pu', 'Bus 4 angle', 'Bus 9 pu', 'Bus 9 angle', 'Bus 14 pu' , 'Bus 14 angle',
                    'Qgen 4', 'Qgen 9', 'Qgen 14',
                    'Load 4 MW reduction in P','Load 9 MW reduction in P','Load 14 MW reduction in P',
                    'Load 4 % reduction in P','% Load 9 reduction in P','Load 14 % reduction in P']) 
                   
    for hour in list(range(1,25)): #iterating through each hour for each case (hr 1 to hr 24)
        
        #determining Q limits based on the time of the day        
        Q_limit4 = cloudy_Q[hour]*PV_size_4
        Q_limit914 = cloudy_Q[hour]*PV_size_914
    
        load_bus_nums = [2,3,4,5,6,9,10,11,12,13,14]
        
        #===================================================================================
        #setting up load values - all constant P at first to get bus voltages
        for load in load_bus_nums:
            
            load_MW = load_percent[hour]*peak_MW[load][0]
            load_MVar = load_percent[hour]*peak_MW[load][1]
            print(load_MVar)
            psspy.load_chng_5(load,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[load_MW,load_MVar,0,0,0,0,0,0]) #constant P parameters

        #set up all the ZIP parameters of the load
        

        #ensuring all PQ buses are originally set to Code 1    
        psspy.bus_chng_4(4,0,[1,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(5,0,[1,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(9,0,[1,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(10,0,[1,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(11,0,[1,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(12,0,[1,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(13,0,[1,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(14,0,[1,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        
        psspy.machine_chng_2(4,r"""1""",[_i,_i,_i,_i,_i,_i],[0.0,0.0,0,0,0,0,_f,_f,_f,_f,_f,_f,_f,_f,_f])#,["",r"""GEN3"""])
        psspy.machine_chng_2(9,r"""1""",[_i,_i,_i,_i,_i,_i],[0.0,0.0,0,0,0,0,_f,_f,_f,_f,_f,_f,_f,_f,_f])#,["",r"""GEN3"""])
        psspy.machine_chng_2(14,r"""1""",[_i,_i,_i,_i,_i,_i],[0.0,0.0,0,0,0,0,_f,_f,_f,_f,_f,_f,_f,_f,_f])#,["",r"""GEN3"""])

        
        #===================================================================================
        
        #running base case load flow study and getting results
        solution = psspy.fnsl([0,0,0,1,1,0,99,0]) #runs the load flow study
        vpu_err, vpu = psspy.abusreal(-1,1,'PU') #all bus voltages in pu
        vang_err, vang_rad = psspy.abusreal(-1,1,'ANGLE')#all bus angles in radians
        vang_deg = [float(ang) * 57.29578 for ang in vang_rad[0]] #converting from radians to degrees
        load_err, load = psspy.aloadcplx(-1, 1, 'totalact') #gets the total value of the MVA drawn from bus 5
                
        print(vpu)
        print(vang_deg)
        print(load)
        
        #collect all pre-CVR data
        pre_CVR = [load[0][2].real, load[0][2].imag, load[0][5].real,load[0][5].imag, load[0][10].real, load[0][10].imag,
                    'very sunny', Q_limit4, Q_limit914,
                    vpu[0][3], vang_deg[3], vpu[0][8], vang_deg[8], vpu[0][13], vpu[0][13]]
        
        #=============================================================================================
        
        #setting up proper ZIP parameters
        v = vpu[0][1]
        Po = load[0][0].real/(v*0.19+0.81)
        Qo = load[0][0].imag/(v*v*2.92-1.92)        
        load_param = [0.81*Po, -1.92*Qo, 0.19*Po, 0, 0,-1*2.92*Qo]                    
        psspy.load_chng_5(2,r"""1""",[_i,_i,_i,_i,_i,_i,_i],load_param) #changing parameters in "Load" tab in Network Data
        
        v = vpu[0][2]
        Po = load[0][1].real/(v*0.19+0.81)
        Qo = load[0][1].imag/(v*v*2.92-1.92)        
        load_param = [0.81*Po, -1.92*Qo, 0.19*Po, 0, 0,-1*2.92*Qo]                    
        psspy.load_chng_5(3,r"""1""",[_i,_i,_i,_i,_i,_i,_i],load_param) #changing parameters in "Load" tab in Network Data

        v = vpu[0][3]
        Po = load[0][2].real/(v*v*0.333+v*0.33+0.33)
        Qo = load[0][2].imag/(v*v*1.493-0.493)        
        load_param = [0.33*Po, -0.493*Qo, 0.33*Po, 0, 0.33*Po,-1*1.493*Qo]        
        psspy.load_chng_5(4,r"""1""",[_i,_i,_i,_i,_i,_i,_i],load_param) #changing parameters in "Load" tab in Network Data
        
        v = vpu[0][4]
        Po = load[0][3].real/(v*v*0.333+v*0.33+0.33)
        Qo = load[0][3].imag/(v*v*1.493-0.493)        
        load_param = [0.33*Po, -0.493*Qo, 0.33*Po, 0, 0.33*Po,-1*1.493*Qo]        
        psspy.load_chng_5(5,r"""1""",[_i,_i,_i,_i,_i,_i,_i],load_param) #changing parameters in "Load" tab in Network Data

        v = vpu[0][5]
        Po = load[0][4].real/(v*v*0.055  +v*0.945)
        Qo = load[0][4].imag/(v*v*1.94-0.94) 
        load_param = [0, -0.94*Qo, 0.945*Po, 0, 0.055*Po, -1*1.94*Qo]
        psspy.load_chng_5(6,r"""1""",[_i,_i,_i,_i,_i,_i,_i],load_param) #changing parameters in "Load" tab in Network Data
        
        v = vpu[0][8]
        Po = load[0][5].real/(v*v*0.055  +v*0.945)
        Qo = load[0][5].imag/(v*v*1.94-0.94) 
        load_param = [0, -0.94*Qo, 0.945*Po, 0, 0.055*Po, -1*1.94*Qo]
        psspy.load_chng_5(9,r"""1""",[_i,_i,_i,_i,_i,_i,_i],load_param) #changing parameters in "Load" tab in Network Data
        
        v = vpu[0][9]
        Po = load[0][6].real/(v*v*0.055  +v*0.945)
        Qo = load[0][6].imag/(v*v*1.94-0.94) 
        load_param = [0, -0.94*Qo, 0.945*Po, 0, 0.055*Po, -1*1.94*Qo]
        psspy.load_chng_5(10,r"""1""",[_i,_i,_i,_i,_i,_i,_i],load_param) #changing parameters in "Load" tab in Network Data
        
        v = vpu[0][10]
        Po = load[0][7].real/(v*v*0.055  +v*0.945)
        Qo = load[0][7].imag/(v*v*1.94-0.94) 
        load_param = [0, -0.94*Qo, 0.945*Po, 0, 0.055*Po, -1*1.94*Qo]
        psspy.load_chng_5(11,r"""1""",[_i,_i,_i,_i,_i,_i,_i],load_param) #changing parameters in "Load" tab in Network Data
        
        v = vpu[0][11]
        Po = load[0][8].real/(v*v*0.055  +v*0.945)
        Qo = load[0][8].imag/(v*v*1.94-0.94) 
        load_param = [0, -0.94*Qo, 0.945*Po, 0, 0.055*Po, -1*1.94*Qo]
        psspy.load_chng_5(12,r"""1""",[_i,_i,_i,_i,_i,_i,_i],load_param) #changing parameters in "Load" tab in Network Data
        
        v = vpu[0][12]
        Po = load[0][9].real/(v*v*0.055  +v*0.945)
        Qo = load[0][9].imag/(v*v*1.94-0.94) 
        load_param = [0, -0.94*Qo, 0.945*Po, 0, 0.055*Po, -1*1.94*Qo]
        psspy.load_chng_5(13,r"""1""",[_i,_i,_i,_i,_i,_i,_i],load_param) #changing parameters in "Load" tab in Network Data
        
        v = vpu[0][13]
        Po = load[0][10].real/(v*v*0.055  +v*0.945)
        Qo = load[0][10].imag/(v*v*1.94-0.94) 
        load_param = [0, -0.94*Qo, 0.945*Po, 0, 0.055*Po, -1*1.94*Qo]
        psspy.load_chng_5(14,r"""1""",[_i,_i,_i,_i,_i,_i,_i],load_param) #changing parameters in "Load" tab in Network Data
        
        #change buses 4, 9,14 to codee 2 and Q limits
        
        psspy.bus_chng_4(4,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(9,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(14,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        
        psspy.machine_chng_2(4,r"""1""",[_i,_i,_i,_i,_i,_i],[0.0,0.0,Q_limit4, -1*Q_limit4,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])#,["",r"""GEN3"""])
        psspy.machine_chng_2(9,r"""1""",[_i,_i,_i,_i,_i,_i],[0.0,0.0,Q_limit914, -1*Q_limit914,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])#,["",r"""GEN3"""])
        psspy.machine_chng_2(14 ,r"""1""",[_i,_i,_i,_i,_i,_i],[0.0,0.0,Q_limit914, -1*Q_limit914,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])#,["",r"""GEN3"""])

        psspy.plant_chng_4(4,0,[_i,_i],[0.97,_f])
        psspy.plant_chng_4(9,0,[_i,_i],[0.97,_f])
        psspy.plant_chng_4(14,0,[_i,_i],[0.97,_f])
              

        solution = psspy.fnsl([0,0,0,1,1,0,99,0]) #only need to run the load flow study once
        
        #after all round 2 load flow studies done, collect load information and Qgen of PV bus
        CVR_load_err, CVR_load = psspy.aloadcplx(-1, 1, 'totalact')
        CVR_p_load = CVR_load[0][0].real
        CVR_q_load = CVR_load[0][0].imag
        qerr, Qgen = psspy.agenbusreal(-1, 1, 'QGEN')
        PV_Qgen = Qgen[0][1] #will always be the second element since only slack bus and PV bus will have any MVar   
        
        print(Qgen)
        
        vpu_err, vpu = psspy.abusreal(-1,1,'PU') #all bus voltages in pu
        vang_err, vang_rad = psspy.abusreal(-1,1,'ANGLE')#all bus angles in radians
        vang_deg = [float(ang) * 57.29578 for ang in vang_rad[0]] #converting from radians to degrees
        
        post_CVR = [CVR_load[0][2].real, CVR_load[0][2].imag, CVR_load[0][5].real,CVR_load[0][5].imag, CVR_load[0][10].real, CVR_load[0][10].imag,
                    vpu[0][3], vang_deg[3], vpu[0][8], vang_deg[8], vpu[0][13], vang_deg[13], Qgen[0][3], Qgen[0][6], 
                    Qgen[0][7]]
                    
        writer.writerow([hour] + pre_CVR + post_CVR)
            
            
end_time = time.clock()
program_time = end_time - start_time
print("program time: {} seconds".format(round(program_time,3)))
print("program time: {} minutes".format(round(program_time/60,3)))