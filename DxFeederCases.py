#This program automates load flow studies for the Dx feeder case in PSSE
#Author: Emma Wang

#Version 1, Jan 2026
#-----------------------------------------------------------------------------------------------

#libraries
import csv
import psspy
import itertools
import math
import time

psspy.psseinit(50) #PSSE initializes environment for up to 50 buses
psspy.case(r"C:\Users\EmmaY\Documents\Eng + Ivey Year 5\Capstone\Dx Feeder Studies\PSSE Cases\Dx_Feeder_V1.sav") #opens case called "Dx_Feeder_V1.sav"

start_time = time.perf_counter()

zip_coeff_names = ['Pz', 'Pi', 'Pp', 'Qz', 'Qi', 'Qp']
zip_res = [0.055,0.945,0,1.94,0,-0.94]
zip_comm = [0.33,0.33,0.33,1.493,0,-0.493]

zip1 = dict(zip_coeff_names, zip_res)
zip2 = dict(zip_coeff_names, zip_comm)

#print(zip_coeff_1)
#print(zip_coeff_2)

#creating cases - round 1
variables = ['load type','ZIP parameters',"power factor","PV farm location","PV farm size", "sun rating"]
values = [
    #"load type" --> ["summer", "winter"],
    ["ZIP"],
    [zip1]  #ZIP co-efficients in order of Pz, Pi, Pp, Qz, Qi, Qp
    #["Z", "I"],
    [0.9, 0.95, 0.98],
    [3,4,5],
    [5.263, 10.526], #corresponds to MW output of 5MW and 10MW from PV farm, assuming 0.95PF (IEEE 2800)
    ["very sunny","moderate sun", "cloudy"] #maybe want to show 
]

cases = [dict(zip(variables, values)) for values in itertools.product(*values)] 
print(len(cases)) #54 cases --> x24 hrs = 2592 cases
print(cases)

#importing relevant hourly data
from hourly_data import load_percent, very_sunny_Q, mod_sunny_Q, cloudy_Q
# print(load_percent)
# print(very_sunny_Q)
# print(mod_sunny_Q)
# print(cloudy_Q)

peak_MW = 10 #using peak MW as 10MW based on Excel analysis (satisifes all PFs from 0.9 to 1.0)

with open('Dx_load_flow_results.csv','w',newline='') as f:#opening a file called 'Dx_load_flow_results.csv"
    
    writer = csv.writer(f)
    
    #header
    writer.writerow(['hour','load MW', 'load Mvar','PF','load type','sun rating','PV_size (MVA)', 'Q limit (MVar)', 
                    'PV bus #', 'PV bus V pu (no CVR)', 'PV bus angle (no CVR)', 'Load bus (5) pu (no CVR)','Load bus (5) angle (no CVR)',
                    'PV bus V pu (CVR)', 'PV bus angle (CVR)', 'Load bus (5) pu (CVR)','Load bus (5) angle (CVR)',
                    'PV bus Q (MVar)', 'PV bus Q limit (MVar)', 'load MW CVR', 'load MVar CVR','MW reduction in P', '% reduction in P']) 

    for currcase in cases[0:1]: #iterating through all the cases
        
        print(currcase)
        
        for hour in list(range(1,2)): #iterating through each hour for each case (hr 1 to hr 24)
            
            #getting parameters for load flow studies
            #print([hour, currcase])
            
            PF = currcase["power factor"] #power factor of the load, determines Q
            load_MW = load_percent[hour] * peak_MW #grabbing % from load data and multiplying by peak load
            load_MVar = load_MW*math.tan(math.acos(PF)) #calculating Q based on P and PF

            sun_rating = currcase["sun rating"]
            PV_size = currcase["PV farm size"] #rated MVA of PV farm

            if sun_rating == "very sunny":
                Q_limit = very_sunny_Q[hour]*PV_size #if "very sunny", access proper Q availability corresponding to hour
            elif sun_rating == "moderate sun":
                Q_limit = mod_sunny_Q[hour]*PV_size
            else: #cloudy
                Q_limit = cloudy_Q[hour]*PV_size
            
            #putting constant P parameters into PSSE for initial run (to get bus 5 voltage to determine ZIP parameters)
            PV_bus = currcase["PV farm location"] #either bus 3,4 or 5
            psspy.load_chng_7(5,r"""L1""",[_i,_i,_i,_i,_i,_i,_i],[load_MW,load_MVar,0,0,0,0,0,0],_s,_s) #constant P parameters
            
            #reset buses 3,4,5 to code 1 to ensure they are PQ buses with no voltage regulation
            psspy.bus_chng_4(3,0,[1,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
            psspy.bus_chng_4(4,0,[1,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
            psspy.bus_chng_4(5,0,[1,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
            
            #psspy.machine_chng_5(PV_bus,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[_f,0.0,Q_limit,-1*Q_limit,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f],["",r"""GEN3"""])
                    
                
            #running base case load flow study and getting results
            solution = psspy.fnsl([0,0,0,1,1,0,99,0]) #runs the load flow study
            vpu_err, vpu = psspy.abusreal(-1,1,'PU') #all bus voltages in pu
            vang_err, vang_rad = psspy.abusreal(-1,1,'ANGLE')#all bus angles in radians
            vang_deg = [float(ang) * 57.29578 for ang in vang_rad[0]] #converting from radians to degrees
            load_err, load = psspy.aloadcplx(-1, 1, 'totalact') #gets the total value of the MVA drawn from bus 5
            
            #base case values (i.e., no CVR implemented)
            PV_bus_pu = vpu[0][PV_bus-1]
            PV_bus_ang = vang_deg[PV_bus-1]
            bus5_pu = vpu[0][4]
            bus5_ang = vang_deg[4]
            p_load = load[0][0].real
            q_load = load[0][0].imag
            
            print("initial PV bus voltage")
            print(vpu[0][PV_bus-1])

            #SUBSEQUENT RUNS - RUNNING UNTIL BUS 5 VOLTAGE WITHIN 0.01PU OF 0.97PU-------------------------------
            
            #calculating proper parameters for load and setting them in PSSE - sets up the NEXT ROUNDS OF LOAD FLOWS (when trying to reduce bus 5 to 0.97pu)
            load_type = currcase["load type"]
            
            #if constant Z load:
            if load_type == "Z":
                P_inp = load_MW/(vpu[0][4]*vpu[0][4]) 
                Q_inp = -1*load_MVar/(vpu[0][4]*vpu[0][4]) #need negative sign for PSSE convention (Power World is opposite with just positive sign)
                psspy.load_chng_7(5,r"""L1""",[_i,_i,_i,_i,_i,_i,_i],[0,0,0,0,P_inp,Q_inp,0,0],_s,_s) #changing parameters in "Load" tab in Network Data
                #print(Q_inp)

            #if constant I load:
            elif load_type == "I":
                P_inp = load_MW/(vpu[0][4]) 
                Q_inp = load_MVar/(vpu[0][4]) 
                psspy.load_chng_7(5,r"""L1""",[_i,_i,_i,_i,_i,_i,_i],[0,0,P_inp,Q_inp,0,0,0,0],_s,_s) #changing parameters in "Load" tab in Network Data
                
            #if ZIP load
            else:
                v5 = vpu[0][4]
                coeffs = currcase['ZIP parameters'] #gets a diciontary with all the values
                
                Po = load_MW/(coeffs['Pz']*v5*v5 + coeffs['Pi']*v5 + coeffs['Pp'])
                Qo = load_MVar/(coeffs['Qz']*v5*v5 + coeffs['Qi']*v5 + coeffs['Qp'])

                load_param = [coeffs['Pp']*Po, coeffs['Qp']*Qo, coeffs['Pi']*Po, coeffs['Qi']*Qo, coeffs['Pz']*Po, -1*coeffs['Qz']*Qo,0,0]

                psspy.load_chng_7(5,r"""L1""",[_i,_i,_i,_i,_i,_i,_i],load_param,_s,_s) #changing parameters in "Load" tab in Network Data


            #changing PV_Bus code to 2 from code 1
            psspy.bus_chng_4(PV_bus,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
            
            #Setting Q limits on PV bus basesd on Q_limit
            #psspy.machine_chng_5(PV_bus,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[_f,_f,Q_limit,-1*Q_limit,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f],["",r"""GEN4"""])
            psspy.machine_chng_5(PV_bus,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[_f,0.0,Q_limit,-1*Q_limit,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f],["",r"""GEN3"""])
                                
            #Setting initial Vsched as current PV bus voltage minus 0.01 or 0.01 (DECIDE) 
            if PV_bus == 5: #can directly set Vsched to 0.97
                psspy.plant_chng_4(5,0,[_i,_i],[0.97,_f])
                solution = psspy.fnsl([0,0,0,1,1,0,99,0]) #only need to run the load flow study once


            else: #bus 3 or 4 = PV bus  
                #print("in loop")
                
                # #Keep looping until 0.97 hit within what % tolerance? (DECIDE) - what is considered close enough?
                # #abs(vpu[0][5] - 0.97) <= 0.001
                qerr, Qgen = psspy.agenbusreal(-1, 1, 'QGEN')
                PV_Qgen = Qgen[0][1] #will always be the second element since only slack bus and PV bus will have any MVar   
                # print("bus 5 pu:")
                # print(vpu[0][4])
                # print("PV bus Q:")
                # print(PV_Qgen)
                # print("PV bus pu:")
                # print(vpu[0][PV_bus-1])
                
                num_iterations = 0
                                       
                while abs(vpu[0][4]-0.97)>=(0.01) and abs(PV_Qgen)<Q_limit and num_iterations<=5: #limiting the # of iterations
                    
                    
                    initial_V = vpu[0][PV_bus-1] - (0.01) #initial guess for PV bus voltage is value - 0.01
                    # print("V guess: ")
                    # print(initial_V)
                    
                    #resetting PV bus voltage, QGen, and Qlimits so load flow runs properly
                    psspy.plant_chng_4(PV_bus,0,[_i,_i],[initial_V,_f])
                    psspy.machine_chng_5(PV_bus,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[_f,0.0,Q_limit,-1*Q_limit,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f],["",r"""GEN3"""])
                    
                    #running the load flow and gathering the new results
                    solution = psspy.fnsl([0,0,0,1,1,0,99,0]) #only need to run the load flow study once
                    vpu_err, vpu = psspy.abusreal(-1,1,'PU') #all bus voltages in pu
                    qerr, Qgen = psspy.agenbusreal(-1, 1, 'QGEN')
                    PV_Qgen = Qgen[0][1] #will always be the second element since only slack bus and PV bus will have any MVar   
                   
                    num_iterations += 1
                    #print(num_iterations)
                   
            #after all round 2 load flow studies done, collect load information and Qgen of PV bus
            CVR_load_err, CVR_load = psspy.aloadcplx(-1, 1, 'totalact')
            CVR_p_load = CVR_load[0][0].real
            CVR_q_load = CVR_load[0][0].imag
            qerr, Qgen = psspy.agenbusreal(-1, 1, 'QGEN')
            PV_Qgen = Qgen[0][1] #will always be the second element since only slack bus and PV bus will have any MVar   
            
            vpu_err, vpu = psspy.abusreal(-1,1,'PU') #all bus voltages in pu
            vang_err, vang_rad = psspy.abusreal(-1,1,'ANGLE')#all bus angles in radians
            vang_deg = [float(ang) * 57.29578 for ang in vang_rad[0]] #converting from radians to degrees
                
                #get Q required to bring PV bus to final voltage (so bus 5 voltage = 0.97pu)
               
            #new values of PV bus and load bus after CVR 
            CVR_PV_bus_pu = vpu[0][PV_bus-1]
            CVR_PV_bus_ang = vang_deg[PV_bus-1]
            CVR_bus5_pu = vpu[0][4]
            CVR_bus5_ang = vang_deg[4]
            
            p_load_reduction = (CVR_p_load-p_load)
            
            if solution !=0: #solution did NOT converge
                writer.writerow(['NA'])
            else: #solution DID converge
                #if PV_bus == 5:
                writer.writerow([hour, p_load, q_load,PF, load_type, sun_rating,PV_size,Q_limit,
                                PV_bus, PV_bus_pu, PV_bus_ang, bus5_pu, bus5_ang,
                                CVR_PV_bus_pu, CVR_PV_bus_ang, CVR_bus5_pu, CVR_bus5_ang,
                                PV_Qgen, Q_limit, CVR_p_load, CVR_q_load, p_load_reduction, p_load_reduction/p_load])
                # else:
                    # writer.writerow([hour, p_load, q_load,PF, load_type, sun_rating,PV_size,Q_limit,
                                    # PV_bus, PV_bus_pu, PV_bus_ang, bus5_pu, bus5_ang])
                
        # keep track of % reduction in MW throughout ENTIRE DAY!! and on a per hour basis

end_time = time.perf_counter()
program_time = end_time - start_time
print("program time: {} seconds".format(round(program_time,3)))
print("program time: {} minutes".format(round(program_time/60,3)))



#NOTES:
#load size will always be 10MW peak, then multiply by % --> this ensures bus 5 voltage always > 0.97pu so actually can implement CVR

#--------------------------------------------------------------------------------------------------------------------


#parameters are as folllowing:
# psspy.fnsl([ 
    # flat,      # 0 = flat start, 1 = no flat start
    # tap,       # tap adjustment locked = 0
    # shunt,     # switched shunts enabled = 0
    # var,       # VAR limits automatically applied = 1
    # phase,     # phase shifters disabled = 1
    # dc,        # DC taps adjust allowed = 0
    # iter,      # max iterations = 99
    # # nondiv     # non-divergent option = not enabled
# # ])
# solution = psspy.fnsl([0,0,0,1,1,0,99,0]) 


# #in case solution did not converge
# if solution !=0:
    # print("solution did not converge!")
     
# #accessing data from load flow study results
# busnum_err, bus_nums = psspy.abusint(-1,1,'NUMBER')
# vmag_err, vmag = psspy.abusreal(-1,1,'KV')
# vpu_err, vpu = psspy.abusreal(-1,1,'PU')
# vang_err, vang_rad = psspy.abusreal(-1,1,'ANGLE')
# vang_deg = [float(ang) * 57.29578 for ang in vang_rad[0]] #converting from radians to degrees

# #HAVING ERRORS WITH THIS - FIX!!!
# p_err, p_load = psspy.aloadint(-1,1,'STATUS')
# q_err, q_load = psspy.aloadreal(-1,1,'PL')

# #note that the above are a list within a list so need to access index 0
# print(bus_nums[0])
# print(vmag[0])
# print(vpu[0])
# print(vang_deg)

# with open('Dx_load_flow_results.csv','w',newline='') as f:
    # writer = csv.writer(f)
    # writer.writerow(['Bus #','kV','pu','angle (deg)'])
    # for bnum in bus_nums[0]:
        # writer.writerow([bnum])
