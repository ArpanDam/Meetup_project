# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 11:40:08 2021

@author: HP
"""
import pickle
import os
import numpy as np
import json
#from numpy.linalg import svd
import numpy
from sklearn.decomposition import TruncatedSVD

os.chdir('./rank_redunced_4_features_json')

f = open('rank4 2',)
  
# returns JSON object as 
# a dictionary
data = json.load(f)


grp_event_common_2088_groups=pickle.load(open("grp_event_common_2088_groups","rb"))
groupjoin=pickle.load(open("groupjoin_common_2088_groups","rb"))

g2=pickle.load(open("g2_avg_distance_windowwise_tagwise_random_sample_2088","rb"))
g3=pickle.load(open("g3_variance_windowwise_tagwise_random_sample_2088","rb"))

os.chdir('./group_wise_pickle')




g12=pickle.load(open("g12_average_window_wise_2088","rb"))
g13=pickle.load(open("g13_variance_window_wise_2088","rb"))
g14=pickle.load(open("g14_sum_window_wise_2088","rb"))




for m in range(2088): # if m=0 then 1st group
    os.chdir('../m2_number_of_event')
    m2_file="m2_number_of_event"+" "+str(m)
    m4_file="m4_entropy"+" "+str(m)
    m8_file="m8_avg_event_1_hop"+" "+str(m)
    m11_file="m11_avg_entropy_1_hop"+" "+str(m)
    
    
    m2=pickle.load(open(m2_file,"rb"))
    m4=pickle.load(open(m4_file,"rb"))
    m8=pickle.load(open(m8_file,"rb"))
    m11=pickle.load(open(m11_file,"rb"))
    
    #os.chdir("../neighbour_groupwise_tagwise")
    m1_file="total_degree_list"+" "+str(m)
    m7_file="m7_avg_degree"+" "+str(m)
    neigbour_file="group_number"+" "+str(m)
    
    m1=pickle.load(open(m1_file,"rb"))
    m7=pickle.load(open(m7_file,"rb"))
    neighbour=pickle.load(open(neigbour_file,"rb"))
    
    iteration=0
    dict1={}  # this dictionary is for each group stores the reduced rank
    '''for key in grp_event_common_2088_groups:
        if (iteration ==m):
            break;
        iteration=iteration+1    
    group_id=key'''
    #print("")
    # find the number of windows in mn2
    for key in m2:
        window_length=len(m2[key])
        
        
    for i in range((window_length)):
        final_list=[]
        list_members=[]
        for member_id in m2:
        #member_id=key
            member_id_feature=[]
            list_members.append(member_id)
            
            
            member_id_feature=[]
            m1_feature=m1[member_id][i]
            m2_feature=m2[member_id][i]
            m4_feature=m4[member_id][i]
            m7_feature=m7[member_id][i]
            m8_feature=m8[member_id][i]
            m11_feature=m11[member_id][i]
            '''neighbour_id=neighbour[key][window]
            if(neighbour_id == (0,)):
                neighbour_id=0    # no neighbour of the member member_id for the window window'''
            
            member_id_feature.append(m1_feature)
            member_id_feature.append(m2_feature)
            member_id_feature.append(m4_feature)
            member_id_feature.append(m7_feature)
            member_id_feature.append(m8_feature)
            member_id_feature.append(m11_feature)
            final_list.append(member_id_feature)
            member_id_feature=[]
        # window over
        arr = numpy.array(final_list)
        svd = TruncatedSVD(n_components=4)
        svd.fit(arr)
        result = svd.transform(arr)
        #print(result)
        
        for j in range(len(list_members)):
            if(list_members[j] in dict1):
                list1=[]
                for n in range(len(dict1[list_members[j]])):
                    tuple1=dict1[list_members[j]][n]  # previously stored data
                    list1.append(tuple1)
                 
                #new_tuple=tuple(result[j])    # new data from the reduced rank
                #list1=[]
                #combined_tuple=tuple2,new_tuple
                list1.append(tuple(result[j]) )
                #list1.append(combined_tuple)
                #dict1[list_members[j]]=tuple2,new_tuple  # string both data
                #dict1[list_members[j]]=list(combined_tuple)  # string both data
                dict1[list_members[j]]=list1
                
                list1=[]
                #print("")
            else:
                list1=[]
                #result[j]=tuple(result[j])
                list1.append(tuple(result[j]))
                #tuple2=tuple(result[j])
                dict1[list_members[j]]=list1
                list1=[]
                #print("")
                
    # get the groupwise dict1
    os.chdir('../rank_redunced_4_features_json')
    output_file_name="rank4"+" "+str(m)
    #pickle.dump( dict1, open( output_file_name, "wb" ) )           
    with open(output_file_name, "w") as fp:
        json.dump(dict1,fp)             
    '''final_list=[]
        U, S, VT = svd(arr)
        U_new=U[:,:4]  # extracting first 4 collums
        a=np.zeros((4, 4), float)
        S_new=S[0:4]
        a[0][0]=float(S_new[0])
        a[1][1]=float(S_new[1])
        a[2][2]=float(S_new[2])
        a[3][3]=float(S_new[3])
        VT_new=VT[:4,:]
        res = np.dot(U_new,a)
        ans=np.dot(res,VT_new)'''
        
        