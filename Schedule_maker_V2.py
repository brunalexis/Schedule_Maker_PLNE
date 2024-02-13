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
    
    fig, axs = plt.subplots(4, 4, figsize = (30,30))
    
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
    
    fig = plt.figure(figsize=(45,45))
    Rooms = ['C16-X', 'C2B-B', 'C2F-F2', 'Unknown', 'C2E-S3', 'C2E-JETE', 'C2F-F1', 'C2E-S4', 'C2D-D', 'CT1-B', 'C15-Y', 'CT2-A', 'C11-U']
    
    for i in range(len(Data[0])):
        f = i//4
        a = i -4*f
        ax = fig.add_subplot(4,4,i+1)
        X = np.arange(len(Data[0][0]))
        for k in range(len(Data)):
            Y =  [elt + 0.05*k for elt in Data[k][i]]
            ax.step(X,Y,label = Names[k],linewidth=4.0)
            
            
        #plt.figure(figsize=(50,50))
        ax.legend(fontsize=25)
        ax.set_title(Rooms[i],fontsize = 40)
    plt.subplots_adjust(left=0.025, bottom=0.03, right=0.99, top=0.97, wspace=0.07, hspace=0.1)
    if save_fig : 
        fig.savefig('Need_vs_Real_vs_Pred_new')    
    plt.show()

def cumulative_demand(Data,Names,savefig=False):
    cumuled = [[0 for _ in range(len(Data[0][0]))] for k in range(len(Data))]
    for i in range(len(Data)):
        for j in range(len(Data[i])):
            for k in range(len(Data[i][j])):
                cumuled[i][k] += Data[i][j][k]

    X = np.arange(len(Data[0][0]))
    fig = plt.figure(figsize=(20,20))
    for i in range(len(cumuled)):
        plt.step(X,cumuled[i],label = Names[i],linewidth=5.0)
    plt.legend(fontsize=40)
    plt.subplots_adjust(left=0.03, bottom=0.045, right=0.99, top=0.99, wspace=None, hspace=None)
    if savefig :
        plt.savefig("Cumulative_Demand")
    plt.show()


def list_un_moins_list_deux(l1,l2):
    for i in range(len(l1)):
        l1[i] = l1[i] - l2[i]
    return l1


def compare_schedule(file_real,file_pred,MORNING,BREAK,AFTER,name1,name2,display=False,save_file=False):

    out_real = create_agent_schedule(file_real,MORNING,BREAK,AFTER,display,save_file)
    out_pred = create_agent_schedule(file_pred,MORNING,BREAK,AFTER,display,save_file)


    
    Need = read_need(file_real)
    PIF = len(Need)
    TIME = len(Need[0])
    SHIFTS = shift_maker(TIME, PIF, MORNING,BREAK,AFTER)

    ##graph_sched(convert_Shift(out_real,SHIFTS,TIME,PIF),convert_Shift(out_pred,SHIFTS,TIME,PIF),name1,name2)

    data = [Need,convert_Shift(out_real,SHIFTS,TIME,PIF),convert_Shift(out_pred,SHIFTS,TIME,PIF)]
    names = ["Need",name1,name2]

    data_cumul = [data[0],data[2]]
    names_cumul = ["Need",name2]

    cumulative_demand(data_cumul,names_cumul,save_file)

    ##difference = list_un_moins_list_deux(Need,data[2])
    ##cumulative_demand([difference],["Pred - Need"],save_file)

    multi_graph(data,names,save_file)



compare_schedule(Data_File_Real,Data_File_Pred,4,1,4,"Real","Pred",False,True)

