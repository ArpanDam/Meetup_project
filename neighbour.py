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




grp_event=pickle.load(open("grp_event_common_2088_groups","rb"))

ersvp=pickle.load(open("ersvp","rb"))


groupjoin=pickle.load(open("groupjoin_common_2088_groups","rb"))

          
os.chdir('./neighbours')
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
    list1=[]    
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
        list_total_minus_5_event = [x for x in member_id if x not in list_of_members_in_5_events]
        for j in range(len(list_total_minus_5_event)):   # this loop is for the members which are not present in any of the 5 events
            # extract the data then append yet to be done
            if(list_total_minus_5_event[j] in dict1):
                tuple1=dict1[list_total_minus_5_event[j]]
                tuple1=list(tuple1)
                list2=[0]
                tuple1.append(tuple(list2))
                tuple1=tuple(tuple1)
                dict1[list_total_minus_5_event[j]]=tuple1
                #print("")
                
            else:
                list2=[0]
                
                dict1[list_total_minus_5_event[j]]=tuple(list2)    
        list_total_minus_5_event=[]
        
        
        
        dict2={}
        for j in range(len(list1)):
            event_id=list1[j]  # the event among the 5 events
            list_of_members_in_event_id=[]
            if(event_id in ersvp):
                for k in range(len(ersvp[event_id])):
                    list_of_members_in_event_id.append(ersvp[event_id][k][0])
            for k in range(len(list_of_members_in_event_id)):
                list_member=[]
                list_member.append(list_of_members_in_event_id[k])
                list_remaining_members=[]
                list_remaining_members = [x for x in list_of_members_in_event_id if x not in list_member]
                if(list_member[0] in dict2):
                    list_temp=[]
                    
                    tuple1=dict2[list_member[0]][0]
                    tuple1=list(tuple1)
                    #tuple1.append(list_remaining_members)
                    list3=tuple1 + list_remaining_members
                    list3=set(list3)
                    list3=list(list3)
                    list3=tuple(list3)
                    list_temp.append(list3)
                    dict2[list_member[0]]=tuple(list_temp)
                    list_temp=[]
                    #print("")
                else:
                    list_temp=[]
                    list_remaining_members=tuple(list_remaining_members)
                    list_temp.append(list_remaining_members)
                    dict2[list_member[0]]=tuple(list_temp)
                    list_temp=[]
        for key in dict2:
            if(key in dict1):
                tuple1=dict1[key]  
                dict1[key]=tuple1+dict2[key]
            else:
                dict1[key]=dict2[key]
        dict2={}    
            
    file_name="group_number"+" "+str(iteration)
    pickle.dump( dict1, open( file_name, "wb" ))
    iteration=iteration+1
    dict1={}       
            
            
            
        
                
                
                