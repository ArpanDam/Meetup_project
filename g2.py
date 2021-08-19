# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 20:34:14 2020

@author: HP
"""


import pickle


from math import radians, cos, sin, asin, sqrt

#from mputil import mpu

from geopy.distance import geodesic

elat=pickle.load(open("elat","rb"))

elon=pickle.load(open("elon","rb"))

#groupjoin=pickle.load(open("groupjoin","rb"))

#egroup=pickle.load(open("egroup","rb"))

group_event=pickle.load(open("grp_event_common_2088_groups","rb"))


dict_to_store_groupidwith_average_event_distance={}

dict_to_store_variance={}


def variance_calc(data):
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


def distance(lat1, lat2, lon1, lon2):   # lat1 and lon1 should be postive , this function is used for calculating distance between 2 location giving there latitude and longitude
      
    # The math module contains a function named 
    # radians which converts from degrees to radians. 
    lon1 = radians(lon1) 
    lon2 = radians(lon2) 
    lat1 = radians(lat1) 
    lat2 = radians(lat2) 
       
    # Haversine formula  
    dlon = lon2 - lon1  
    dlat = lat2 - lat1 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  
    c = 2 * asin(sqrt(a))  
     
    # Radius of earth in kilometers. Use 3956 for miles 
    r = 6371
       
    # calculate the result 
    return(c * r) 


for key in group_event:
    list1=[]
    list2=[]
    list_to_store_avg_distance=[]
    list_to_store_variance=[]
    
    
    # calculating average distance of all the distance 
    
    for i in range(len(group_event[key])):
        list2.append(group_event[key][i][0])# list1 contains list of events
    
    for i in range(len(list2)-9):
        list1=list2[i : i+5]
        k=0
    
    
        list_lat=[]
        list_lon=[]
    
        for k in range(len(list1)):
            try:
                list_lat.append(elat[list1[k]])
                list_lon.append(elon[list1[k]])
            except:
                pass
        
        if(len(list_lat)==1) or(len(list_lat)==0):   # is no elat or elon found
            list_to_store_avg_distance.append(0)
            list_to_store_variance.append(0)
            
        
        else:  # if more than 1 elat elon found on 5 events of list1  
            
            m=0
            dis_sum=0
            temp_list_to_store_variance=[]
            for m in range(len(list_lat)-1):
                n=m+1
                for n in range(len(list_lat)):
                #dis=distance(list_lat[m], list_lat[n], list_lon[m], list_lon[n])
                #dis = mpu.haversine_distance((list_lat[m], list_lon[m]), (list_lat[n], list_lon[n]))
                    origin = (list_lat[m], list_lon[m])  # (latitude, longitude) don't confuse
                    dist = (list_lat[n], list_lon[n])
                    dis=geodesic(origin, dist).kilometers   # Using geopy libary to calculate distance , the function distance(lat1, lat2, lon1, lon2) can also be used both process gives same result
                    #dis_km=geodesic(origin, dist).kilometers
                    temp_list_to_store_variance.append(dis)
                    dis_sum=dis_sum+dis
                
            avg_dis=dis_sum/len(list_lat)
            variance1=variance_calc(temp_list_to_store_variance)
            list_to_store_variance.append(variance1)
            temp_list_to_store_variance=[]
            
            list_to_store_avg_distance.append(avg_dis)
            
    dict_to_store_groupidwith_average_event_distance[key]=list_to_store_avg_distance
    dict_to_store_variance[key]=list_to_store_variance
    #print("")    
        
        
pickle.dump( dict_to_store_groupidwith_average_event_distance, open( "g2_avg_distance_windowwise_tagwise_2088", "wb" ) )

pickle.dump( dict_to_store_variance, open( "g3_variance_windowwise_tagwise_2088", "wb" ) )

print("The end")