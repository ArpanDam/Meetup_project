# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:43:46 2021

@author: HP
"""

import numpy as np
import pickle

import os
import shelve







#total_degree_of_member_windowwise=pickle.load(open("total_degree_of_member_windowwise21","rb"))


'''list1=[[5620833, 6657235, 8978228, 9041705, 9061588], [ 9420414, 10290000, 11350459, 11413753, 11738963, 12351926]]
list2=0
list1.append(list2)

list2=[12,4]

list1.append(list2)'''


#tag_new_slide1=pickle.load(open("tag_new_slide1","rb"))

grp_event=pickle.load(open("grp_event_common_2088_groups","rb"))

ersvp=pickle.load(open("ersvp","rb"))


groupjoin=pickle.load(open("groupjoin_common_2088_groups","rb"))

          
os.chdir('./m2_number_of_event')

#m2_number_of_event_2085=pickle.load(open("m2_number_of_event 2085","rb"))

'''iteration=0

for key in grp_event:
    iteration=iteration+1
    if(iteration==2086):
        break'''


iteration=0
for key in grp_event:
    '''if(iteration<=1354):
        iteration=iteration+1
        continue'''
    dict1={}
    member_id=[] # used for storing all members of the group
    for i in range(len(groupjoin[key])):
        member_id.append(groupjoin[key][i][0])
    list_of_events=[]
    for i in range(len(grp_event[key])):
        list_of_events.append(grp_event[key][i][0])
    list1=[] # used for storing the list of 5 events    
    for i in range(len(list_of_events)-9):
        
        
        list1=list_of_events[i:i+5]   # list1 contains list of 5 events
        
        
        # find the list of members present in event present in list1
        
        #dict1={}
        list_total_minus_5_event=[]
        list_of_members_in_5_events=[]
        for j in range(len(list1)):
            event=list1[j]                 # event is each event of the 5 event
            
            try:
                for k in range(len(ersvp[event])):
                    list_of_members_in_5_events.append(ersvp[event][k][0])     # list of members for event list1[j] that is event
            except:
                pass
        # substract list_of_members_in_5_events
        list_of_members_in_5_events=set(list_of_members_in_5_events)
        list_of_members_in_5_events=list(list_of_members_in_5_events)
        list_total_minus_5_event = [x for x in member_id if x not in list_of_members_in_5_events]
        for j in range(len(list_total_minus_5_event)):   # this loop is for the members which are not present in any of the 5 events
            # extract the data then append yet to be done
            if(list_total_minus_5_event[j] in dict1):
                tuple1=dict1[list_total_minus_5_event[j]]
                #tuple1=list(tuple1)
                list2=[0]
                #tuple1.append(list2)
                #tuple1.append(tuple(list2))
                #tuple1=tuple(tuple1)
                dict1[list_total_minus_5_event[j]]=tuple1+list2
                list2=[]
                #print("")
                
            else:
                list2=[0]
                #tuple1=[]
                #tuple1.append(list2)
                
                
                dict1[list_total_minus_5_event[j]]=list2  
                list2=[]
                tuple1=[]
        list_total_minus_5_event=[]
        
        
        for j in range(len(list_of_members_in_5_events)): # iterate for each member who attended the 5 events
            member=list_of_members_in_5_events[j]
            number_of_event=0  # this is the number of event for member with id member
            for k in range(len(list1)): # iterate for each event of the group
                event_id=list1[k]
                if(event_id in ersvp):
                    for m in range(len(ersvp[event_id])):
                        if(member==ersvp[event_id][m][0]):
                            number_of_event=number_of_event+1
                            break
            if(member in dict1):
                tuple1=dict1[member]
                list2=[]
                list2.append(number_of_event)
                #tuple1.append(list2)
                dict1[member]=tuple1+list2
                list2=[]
            else:
                list2=[]
                list2.append(number_of_event)
                dict1[member]=list2
                list2=[]
                
                
                            
    file_name="m2_number_of_event"+" "+str(iteration)
    pickle.dump( dict1, open( file_name, "wb" ))
    iteration=iteration+1
    dict1={}                                   
                        
                
            
        
        
print("The end")        
        
        
        
        
        
        
        
         
            
            
            
        
                
                
                