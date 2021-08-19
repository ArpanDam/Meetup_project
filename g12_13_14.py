# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 19:42:16 2021

@author: HP
"""
import pickle





import os

def variance(data):  # for calculating variance
    # Number of observations
    n = len(data)
    # Mean of the data
    if(n > 0):
        mean = sum(data) / n
     # Square deviations
        deviations = [(x - mean) ** 2 for x in data]
     # Variance
        variance = sum(deviations) / n
        return variance
    else:
        return 0
new_grp_event=pickle.load(open("grp_event_common_2088_groups","rb"))

hcount=pickle.load(open("hcount","rb"))



os.chdir('./group_wise_pickle')

dict1={}  # for storing sum of RSVP
dict2={}   # for stroring average RSVP
dict3={}    # variance of rsvp
for key in new_grp_event:
    list1=[]
    list2=[]
    list3=[]
    list_of_events=[]
    for j in range(len(new_grp_event[key])):
        
        list_of_events.append(new_grp_event[key][j][0])
    list_to_store_5_events=[]
    for i in range(len(list_of_events)-9):  # 
        list_to_store_5_events=list_of_events[i : i+5]
        list_to_calculate_variance=[]
        sum1=0
        for k in range(len(list_to_store_5_events)):  #for each event of the window
        
            if(list_to_store_5_events[k] in hcount):
                sum1=sum1+hcount[list_to_store_5_events[k]]
                list_to_calculate_variance.append(hcount[list_to_store_5_events[k]])    
        
            '''try:
                sum1=sum1+hcount[list_to_store_5_events[k]]
                list_to_calculate_variance.append(hcount[list_to_store_5_events[k]])
            except:
                print("")'''
        variance1=variance(list_to_calculate_variance)
        avg=sum1/5
        list3.append(sum1)
        list2.append(variance1)
        list1.append(avg)
    dict1[key]=list1     # avg   
    dict2[key]=list2     # variance
    dict3[key]=list3        # sum
pickle.dump( dict1, open( "g12_average_window_wise_2088", "wb")) 
pickle.dump( dict2, open( "g13_variance_window_wise_2088", "wb"))
pickle.dump( dict3, open( "g14_sum_window_wise_2088", "wb"))                  


'''#pickle.dump( dict1, open( "g12_window_wise_protocol_2", "wb"), protocol=2) 
pickle.dump( dict2, open( "g13_window_wise_protocol_2", "wb"), protocol=2)
#pickle.dump( dict3, open( "g14_window_wise,protocol_2", "wb"), protocol=2)  '''                
                