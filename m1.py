# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 20:52:09 2021

@author: HP
"""


import pickle

import os

'''tag_new_slide1=pickle.load(open("tag_new_slide1","rb"))

groupjoin=pickle.load(open("groupjoin_common_2088_groups","rb"))

grp_event_common=pickle.load(open("grp_event_common_2088_groups","rb"))

ersvp=pickle.load(open("ersvp","rb"))

os.chdir('./neighbour_groupwise_tagwise')

group_number_2072=pickle.load(open("group_number 2064","rb"))


iteration=0
for key in groupjoin:
    iteration=iteration+1
    if(iteration==2065):
        break
print("") '''   
    
    

'''all_event_we_tag5=pickle.load(open("all_event_we_tag5","rb"))

all_event_we_grp5=pickle.load(open("all_event_we_grp5","rb"))
all_event_we_window5=pickle.load(open("all_event_we_window5","rb"))
for key in grp_event_common:
    if (len(grp_event_common[key])==100):
        print(key)
        break
group_id=key

for key in all_event_we_grp5:
    if(all_event_we_grp5[key]== group_id):
        print(key)'''
    

#all_event_we_x5=pickle.load(open("all_event_we_x5","rb"))



os.chdir('./neighbour_groupwise_tagwise')

#ersvp=pickle.load(open("total_degree 408","rb"))

for i in range(2088): # iterate for each group and there are 2088 groups

    
    file_name="group_number"+" "+str(i)
    
    dict1={} # used to store the total degree of member of a particular group[
    neighbours=pickle.load(open(file_name,"rb")) # for a particular group
    #os.chdir('..')
    for key in neighbours: # iterate for member of the group
        
        list1=[]
        for j in range(len(neighbours[key])): # iterate for each window of the member
            
            if(neighbours[key][j]==0):
                #list2=[0]
                list1.append(0)
            else:
                #print(key)
                #print(j)
                try:
                    if(neighbours[key][j][0] == 0):
                        list1.append(0)
                    else:
                        degree=len(neighbours[key][j])
                        list1.append(degree)    
                except:
                    list1.append(0)
                    
                '''list_temp.append(degree)
                list_temp=tuple(list_temp)
                list1.append(list_temp)
                list_temp=[]'''
        #print("")
        list_final=[]
        '''for j in range(len(list1)):
            list_temp=[]
            list_temp.append(list1[j])
            list_temp=tuple(list_temp)
            list_final.append(list_temp)
            list_temp=list(list_temp)
            list_temp=[]'''
        dict1[key]=list1
        list_final=[]
        list_temp=[]
    #os.chdir('..')
    #os.chdir('./m1_total_degree_of_member_2088')    
    file_name="total_degree_list"+" "+str(i)
    #os.chdir('..')
    #pickle.dump( dict1, open( file_name, "wb" ))
    dict1={}        
            
            
            
            
            
    #print("")
print("The end")    