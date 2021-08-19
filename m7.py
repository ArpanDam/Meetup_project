# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 17:46:16 2021

@author: HP
"""

import pickle

import os

#tag_new_slide1=pickle.load(open("tag_new_slide1","rb"))

groupjoin_common=pickle.load(open("groupjoin_common_2088_groups","rb"))

grp_event_common=pickle.load(open("grp_event_common_2088_groups","rb"))

'''match=0
dict1={}  # for groupjoin_common
dict2={}  # for grp_event_common 
for key in tag_new_slide1:
    for key1 in grp_event_common:
        if(key == key1):
            match=match+1
            dict1[key]=groupjoin_common[key]
            dict2[key]=grp_event_common[key]
            break

pickle.dump( dict1, open( "groupjoin_common_2088_groups", "wb" ))

pickle.dump( dict2, open( "grp_event_common_2088_groups", "wb" ))            
print(match) '''           
            
os.chdir('./neighbour_groupwise_tagwise')

'''iteration=0
for key in grp_event_common:
    iteration=iteration+1
    if iteration==1345:
        break
group_id=key    

m7_avg_degree_1344=pickle.load(open("m7_avg_degree 1344","rb"))

group_number_1344=pickle.load(open("group_number 1344","rb"))

for j in range(len(groupjoin_common[group_id])):
    if(22730769 == groupjoin_common[group_id][j][0] ):
        print("")'''

for i in range(2088): # iterate for each group
    file_name1="group_number"+" "+str(i)
    neighbours=pickle.load(open(file_name1,"rb")) # for a particular group
    file_name2="total_degree_list"+" "+str(i)
    degree_file=pickle.load(open(file_name2,"rb")) # for a particular group
    dict1={}   # for each group
    # find the neighbours of key
    for key in neighbours:  # iterate for each member of a group
        list1=[]  # this liost will store the avg degree of neighbours of member_id in a window wise manner
        member_id=key
        for j in range(len(neighbours[key])):  # iterate for each window of the neighbour so window number = j
            #print("")
            if(neighbours[member_id][j]==0):
                #list2=[0]
                list1.append(0)
            else:
                #print(key)
                #print(j)
                try:
                    if(neighbours[member_id][j][0] == 0):
                        list1.append(0)
                    else:
                        #################################
                        sum1=0 # sum of degree of the neighbours
                        avg=0
                        for k in range(len(neighbours[member_id][j])): # iterate for each neighbour of member_id at window j
                            neighbour_id=neighbours[member_id][j][k]
                            #the window is j
                            # need to find degree of the member neighbour_id at window j
                            sum1=sum1+degree_file[neighbour_id][j]
                        avg=sum1/len(neighbours[member_id][j])
                        list1.append(avg)
                            
                            #degree=degree_file[neighbour_id][j][k]  # get the degree of the neighbour of member_id at window j from degree file
                            
                       
                        #print("")
                except:
                    list1.append(0) 
        dict1[key]=list1
    file_name="m7_avg_degree"+" "+str(i)
    pickle.dump( dict1, open( file_name, "wb" ))
    dict1={}                     
                        
            
    '''file_name2="total_degree"+" "+str(i)
    degree_file=pickle.load(open(file_name2,"rb")) # for a particular group
    for key in degree_file:  # iterate for each member of the group
        list1=[]   
        for i in range(len(degree_file[key])): 
            if (degree_file[key][i][0]==0):
                list1.append(0)
            else:
                print("")'''
               
               
        #print("")