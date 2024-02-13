import gurobipy as gp
from gurobipy import GRB
import numpy as np
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import sys

Data_File_Real = sys.argv[1]
Data_File_Pred = sys.argv[2]

def read_need(adresse):
    out = []
    with open(adresse) as file:
        for ligne in file:
            new_line = []
            for elt in ligne:
                if elt not in ['[',",","]"," ","\n"]:
                    new_line.append(int(elt))
            out.append(new_line)
    return out
def shift_maker(Nb_Time_Step, Nb_Pif,Morning, Break, After):
    Shifts = []
    for p1 in tqdm(range(1,Nb_Pif+1)):
        for p2 in range(1,Nb_Pif+1):
            for i in range(Nb_Time_Step-(Morning+Break+After-1)):
                newline = [0 for j in range(Nb_Time_Step)]
                for k1 in range(Morning):
                    newline[i+k1] = p1
                for k2 in range(After):
                    newline[i+k2+Morning+Break] = p2
                Shifts.append(newline)
    return Shifts
def S_maker(Shift, TIME, PIF):
    return [[[int(Shift[i][t] == p+1) for t in range(TIME)]for p in range(PIF)]for i in tqdm(range (len(Shift)))]

def write_file_schedule(optim_out,shifts,Name):
    
    File_Name = Name + ".csv"
    with open(File_Name,"w") as file:
        for i in range (len(optim_out)):
            if optim_out[i] != 0:
                for j in range(optim_out[i]):
                    line = ""
                    for k in range(len(shifts[i])-1):
                        line += str(shifts[i][k])
                        line +=","
                    line += str(shifts[i][-1])
                    line += "\n"
                    file.write(line)

def convert_Shift(var,s,Time,Pif):
    Sched = [[0 for i in range(Time)] for p in range(Pif)]
    for i in range(len(var)):
        if var[i] != 0 :
            shift = s[i]
            for j in range(len(shift)) :
                if shift[j] != 0:
                    Sched[shift[j]-1][j] += var[i]
    return Sched

def graph_sched(need, sched,name1 = "Need",name2 = "Schedule"):
    
    fig, axs = plt.subplots(3, 4, figsize = (30,30))
    
    for i in range(len(need)):
        f = i//4
        a = i -4*f
        ##plt.figure(figsize = (30,30))
        X = np.arange(len(need[0]))
        Y1 = need[i]
        Y2 = [elt + 0.1 for elt in sched[i]]
        #plt.figure(figsize=(50,50))
        axs[f,a].step(X,Y1,label = name1)
        axs[f,a].step(X,Y2,label = name2)
        axs[f,a].legend()
        axs[f,a].set_title("PIF N°" + str(i+1))
    ##fig.savefig('Need_vs_Schedule')
    plt.show()


def create_agent_schedule(File,MORNING,BREAK,AFTER,display=False,save_file=False):
    Need = read_need(File)
    PIF = len(Need)
    TIME = len(Need[0])

    SHIFTS = shift_maker(TIME, PIF, MORNING,BREAK,AFTER)
    S = S_maker(SHIFTS,TIME,PIF)

    m = gp.Model("agent")
    n = m.addVars(len(SHIFTS),vtype=GRB.INTEGER, name = 'n')
    m.update()
    m.setObjective(n.sum(),GRB.MINIMIZE)
    for p in tqdm(range(PIF)):
        for t in range(TIME):
            if Need[p][t]>0:
                m.addConstr(gp.quicksum(n[i]*S[i][p][t] for i in range(len(SHIFTS)))>=Need[p][t])   
    m.update()
    m.optimize()

    OUT = [int(v.X) for v in m.getVars()]

    if display:
        graph_sched(Need,convert_Shift(OUT,SHIFTS,TIME,PIF))
    if save_file:
        write_file_schedule(OUT,SHIFTS,"Schedule_Test")

    return OUT


def multi_graph(Data,Names,save_fig=False):
    
    fig, axs = plt.subplots(3, 4, figsize = (30,30))
    Rooms = ['C2A-A', 'CT2-A', 'C2C-C', 'CT1-B', 'C2D-D', 'C2E-JETE', 'C2E-S3', 'C2E-S4', 'C2F-F1', 'C2F-F2', 'Unknown', 'C2B-B']
    
    for i in range(len(Data[0])):
        f = i//4
        a = i -4*f
        ##plt.figure(figsize = (30,30))
        X = np.arange(len(Data[0][0]))
        for k in range(len(Data)):
            Y =  [elt + 0.05*k for elt in Data[k][i]]
            axs[f,a].step(X,Y,label = Names[k])
        #plt.figure(figsize=(50,50))
        axs[f,a].legend()
        axs[f,a].set_title(Rooms[i])
    if save_fig : 
        fig.savefig('Need_vs_Real_vs_Pred')    
    plt.show()


def list_difference(l1,l2):
    out = []
    for i in range(len(l1)):
        dif_pif = []
        for j in range(len(l1[1])):
            dif_pif.append(l1[i][j]-l2[i][j])
        out.append(dif_pif)
    return out



def compare_schedule(file_real,file_pred,MORNING,BREAK,AFTER,name1,name2,comparison = False,display=False,save_file=False):

    out_real = create_agent_schedule(file_real,MORNING,BREAK,AFTER,display,save_file)
    out_pred = create_agent_schedule(file_pred,MORNING,BREAK,AFTER,display,save_file)




    
    Need = read_need(file_real)
    PIF = len(Need)
    TIME = len(Need[0])
    SHIFTS = shift_maker(TIME, PIF, MORNING,BREAK,AFTER)

    if comparison:
        need_vs_pred = list_difference(convert_Shift(out_pred,SHIFTS,TIME,PIF),Need)
        real_vs_pred = list_difference(convert_Shift(out_pred,SHIFTS,TIME,PIF),convert_Shift(out_real,SHIFTS,TIME,PIF))
        real_vs_need = list_difference(convert_Shift(out_real,SHIFTS,TIME,PIF),Need)

        multi_graph([need_vs_pred],["Need vs Pred"])
        multi_graph([real_vs_pred],["Real vs Pred"])
        multi_graph([real_vs_need],["Real vs Need"])


    ##graph_sched(convert_Shift(out_real,SHIFTS,TIME,PIF),convert_Shift(out_pred,SHIFTS,TIME,PIF),name1,name2)

    data = [Need,convert_Shift(out_real,SHIFTS,TIME,PIF),convert_Shift(out_pred,SHIFTS,TIME,PIF)]
    names = ["Need",name1,name2]

    multi_graph(data,names,save_file)



compare_schedule(Data_File_Real,Data_File_Pred,4,1,4,"Real","Pred",True,False,True)

