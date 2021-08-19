# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 21:15:18 2021

@author: HP
"""

import pickle

import os

import math





def entropy_calc(prob_a,prob_b):
    a=-(prob_a)*math.log(prob_a,2)
    b=-(prob_b)*math.log(prob_b,2)
    return (a+b)






os.chdir('./m2_number_of_event')
for i in range(2088):
    file_name2="m2_number_of_event"+" "+str(i)
    number_of_event_file=pickle.load(open(file_name2,"rb")) # for a particular group
    dict1={}   # this dictioanry is for each group
    for key in number_of_event_file:  # iterate for every member of the event file
        member_id=key
        list1=[] # store the entropy for member member id
        for j in range(len(number_of_event_file[member_id])):
            if(number_of_event_file[member_id][j]==0) or (number_of_event_file[member_id][j]==5):
                list1.append(0)
            else:
                number_of_event=number_of_event_file[member_id][j]
                prob_a=float(number_of_event/5)
                prob_b=1-prob_a
                entropy=entropy_calc(prob_a,prob_b)
                list1.append(entropy)
        dict1[key]=list1
    file_name="m4_entropy"+" "+str(i)
    pickle.dump( dict1, open( file_name, "wb" ))
    
    dict1={}                          
                
                
            
        
print("The end")