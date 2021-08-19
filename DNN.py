# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 12:16:18 2021

@author: HP
"""
import math
import numpy as np
    
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import random

import csv
import pickle

import os

import time
import json

####################################################################################################################################################
# model
    
class model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 10)  
        self.layer11 = nn.Linear(10, 10)  
        self.layer2 = nn.Linear(10, 4)   
        self.layer3 = nn.Sequential(
            nn.Linear(5,3),
            #nn.ReLU(),,
            #nn.Linear(4,2)
            #nn.Softmax()
            nn.Sigmoid()
            
            
            
            
            
        )
        self.layer4 = nn.Sequential(
           nn.Linear(7,4),
           nn.Sigmoid(),
           nn.Linear(4,3),
           nn.Softmax()
            
        )
    
    def first_group(self,xb):
        #xb = xb.reshape(-1, 5)
        activation1=self.layer3(xb)
        return activation1
    def forward_half1(self, xb):
        #xb = xb.reshape(-1, 4)
        activation2 = self.layer1(xb)
        activation2 = F.sigmoid(activation2)
        return activation2
    def forward_half11(self, xb):
        #xb = xb.reshape(-1, 10)   
        activation3 = self.layer11(xb)
        activation3 = F.sigmoid(activation3)
        return activation3
    def forward_half2(self, xb):    
        activation4 = self.layer2(xb)
        activation4=F.softmax(activation4)
        return activation4
    def forward(self, x, path='all'):
        if path=='all':
            x = self.forward_half1(x)
            x = self.forward_half2(x)
        elif path=='half1':
            x = self.forward_half1(x)
        elif path=='half11':
            x = self.forward_half11(x)    
        elif path=='half2':
            x = self.forward_half2(x)
        elif path=='group1':
            x = self.first_group(x)    
        else:
            raise NotImplementedError
        return x
    
    def training_step(self,neighbour,groupjoin,g2,g3,g12,g13,g14,tag_feature,window,rank4_features):
        start = time. time()
        list_of_members=[]
        role_vector={}
        for key in rank4_features:
            list_of_members.append(key)    
        #f_final={}
        f_final=[0,0,0,0]
        f_final = np.array(f_final,dtype='float32')
        f_final = torch.from_numpy(f_final)
        
        for L in range(4):
            #f[L]=[0,0,0,0,0,0]
            for member_id_str in list_of_members:
                member_id=int(member_id_str)
                if(L==0):
                    role_vector_member_id=[]
                
                    #role_vector_member_id=list(rank4_features[member_id][window])
                    role_vector_member_id=rank4_features[member_id_str][window]
                    
                    role_vector[member_id]=role_vector_member_id
                sum1=[0,0,0,0] # used to store the SIGMA part
                neighbour_id=neighbour[member_id][window]  # finding the neighbour of the member
                if((neighbour_id == (0,) or neighbour_id == 0) and (L==0)): # if no neighbour and layer =0
                    sum1=[0,0,0,0]
                    v1 = [ sum1[x] + role_vector[member_id][x] for x in range (len (sum1))]
                    v1 = np.array(v1,dtype='float32') 
                if((neighbour_id == (0,) or neighbour_id == 0) and (L!=0)): # if no neighbour and layer >0
                    sum1=[0,0,0,0,0,0,0,0,0,0]
                    sum1 = np.array(sum1,dtype='float32')
                    sum1 = torch.from_numpy(sum1)
                    #role_vector[member_id]=role_vector[member_id].tolist()[0]
                    v1=torch.add(sum1, role_vector[member_id])
                
                if((neighbour_id != (0,) and neighbour_id != 0)):  # if neighbour is present
                    #sum1=[0,0,0,0,0,0]
                    if(L==0): # if 1st layer
                        sum1=[0,0,0,0]
                        for i in range(len(neighbour_id)):  # for calculating sigma part
                            neighbour_role=[]
                            #sum1=[0,0,0,0,0,0]
                            individual_neighbour_id=neighbour_id[i]
                            #neighbour_role=list(rank4_features[individual_neighbour_id][window])
                            neighbour_role=rank4_features[str(individual_neighbour_id)][window]
                            sum1 = [ sum1[x] + neighbour_role[x] for x in range (len (sum1))]
                        v1 = [ sum1[x] + role_vector[member_id][x] for x in range (len (sum1))]
                        v1 = np.array(v1,dtype='float32')    
                        #v1 = [ sum1[x] + role_vector[member_id][x] for x in range (len (sum1))]    
                    else:  # if not first layer
                        sum1=[0,0,0,0,0,0,0,0,0,0]
                        sum1 = np.array(sum1,dtype='float32')
                        sum1 = torch.from_numpy(sum1)
                        for i in range(len(neighbour_id)):
                            neighbour_role=[]
                            individual_neighbour_id=neighbour_id[i]
                            neighbour_role=role_vector[individual_neighbour_id]
                            #neighbour_role=neighbour_role.tolist()[0]
                            sum1=torch.add(sum1, neighbour_role)
                            #sum1 = [ sum1[x] + neighbour_role[x] for x in range (len (sum1))]
                        #role_vector[member_id]=role_vector[member_id].tolist()[0]
                        v1=torch.add(sum1, role_vector[member_id])
                #role_vector[member_id]=role_vector[member_id].tolist()[0]        
                #v1 = [ sum1[x] + role_vector[member_id][x] for x in range (len (sum1))]
                #v1 = np.array(v1,dtype='float32')
                if(len(v1)==4):
                    v1 = torch.from_numpy(v1)
                #if(v1.shape)
                    out1 = model(v1, path='half1')  # out1 is v2
                if(len(v1)==1):  # if v1 is a tensor that is if layer >0
                    #v1 = torch.from_numpy(v1)
                #if(v1.shape)
                    out1 = model(v1, path='half11')    
                role_vector[member_id]=out1
                out2 = model(out1, path='half2')  # out2 is softmax(v2TWl)
                #out2=out2.tolist()[0] # converting tensor to list
                f_final=torch.add(out2, f_final)
                #f_final=[ f_final[x] + out2[x] for x in range (len (f_final))] # this is for a particular member
        #print(" ")
        ############################################## for group feature #############################
        #f_final = np.array(f_final,dtype='float32')
        #f_final = torch.from_numpy(f_final) # changing f_final
        group_features=[]
        group_features.append(g2)
        group_features.append(g3)
        group_features.append(g12)
        group_features.append(g13)
        group_features.append(g14)
        group_features = np.array(group_features,dtype='float32')
        group_features = torch.from_numpy(group_features)
        out3 = model(group_features, path='group1')
        #out3=self.layer3(group_features)
        # concatenate out3 and f_final
        #out4 = torch.cat((f_final, out3),1)
        out4 = torch.cat((f_final, out3))
        out5=self.layer4(out4) # out5 is the final output which is to be used for calculating loss  here error
        
        tag_feature_list=[]
        
        if(tag_feature==-1):
           new_tag_feature=0
        if(tag_feature==0):
           new_tag_feature=1
        if(tag_feature==1):
           new_tag_feature=2
        tag_feature_list.append(new_tag_feature)   
        label = np.array(tag_feature_list,dtype='float32')
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)
        out5 = out5.reshape(-1, 3) # changing shape of out5 # added later
        end = time. time()
        #print("Time to execute the training function without calculating loss is",end-start)
        loss = F.cross_entropy(out5, label)
        
        
        return(loss)                
                        
       
            
   
            
    def validation_step(self,neighbour,groupjoin,g2,g3,g12,g13,g14,tag_feature,window,rank4_features):
            list_of_members=[]
            role_vector={}
            for key in rank4_features:
                list_of_members.append(key)    
            #f_final={}
            f_final=[0,0,0,0]
            f_final = np.array(f_final,dtype='float32')
            f_final = torch.from_numpy(f_final)
        
            for L in range(3):
                #f[L]=[0,0,0,0,0,0]
                for member_id in list_of_members:
                    if(L==0):
                        role_vector_member_id=[]
                
                        role_vector_member_id=list(rank4_features[member_id][window])    
                        
                        role_vector[member_id]=role_vector_member_id
                    sum1=[0,0,0,0] # used to store the SIGMA part
                    neighbour_id=neighbour[member_id][window]  # finding the neighbour of the member
                    if((neighbour_id == (0,) or neighbour_id == 0) and (L==0)): # if no neighbour and layer =0
                        sum1=[0,0,0,0]
                        v1 = [ sum1[x] + role_vector[member_id][x] for x in range (len (sum1))]
                        v1 = np.array(v1,dtype='float32') 
                    if((neighbour_id == (0,) or neighbour_id == 0) and (L!=0)): # if no neighbour and layer >0
                        sum1=[0,0,0,0,0,0,0,0,0,0]
                        sum1 = np.array(sum1,dtype='float32')
                        sum1 = torch.from_numpy(sum1)
                        #role_vector[member_id]=role_vector[member_id].tolist()[0]
                        v1=torch.add(sum1, role_vector[member_id])
                
                    if((neighbour_id != (0,) and neighbour_id != 0)):  # if neighbour is present
                        #sum1=[0,0,0,0,0,0]
                        if(L==0): # if 1st layer
                            sum1=[0,0,0,0]
                            for i in range(len(neighbour_id)):  # for calculating sigma part
                                neighbour_role=[]
                                #sum1=[0,0,0,0,0,0]
                                individual_neighbour_id=neighbour_id[i]
                                neighbour_role=list(rank4_features[individual_neighbour_id][window])
                                
                                sum1 = [ sum1[x] + neighbour_role[x] for x in range (len (sum1))]
                            v1 = [ sum1[x] + role_vector[member_id][x] for x in range (len (sum1))]
                            v1 = np.array(v1,dtype='float32')    
                            #v1 = [ sum1[x] + role_vector[member_id][x] for x in range (len (sum1))]    
                        else:  # if not first layer
                            sum1=[0,0,0,0,0,0,0,0,0,0]
                            sum1 = np.array(sum1,dtype='float32')
                            sum1 = torch.from_numpy(sum1)
                            for i in range(len(neighbour_id)):
                                neighbour_role=[]
                                individual_neighbour_id=neighbour_id[i]
                                neighbour_role=role_vector[individual_neighbour_id]
                                #neighbour_role=neighbour_role.tolist()[0]
                                sum1=torch.add(sum1, neighbour_role)
                                #sum1 = [ sum1[x] + neighbour_role[x] for x in range (len (sum1))]
                            #role_vector[member_id]=role_vector[member_id].tolist()[0]
                            v1=torch.add(sum1, role_vector[member_id])
                    #role_vector[member_id]=role_vector[member_id].tolist()[0]        
                    #v1 = [ sum1[x] + role_vector[member_id][x] for x in range (len (sum1))]
                    #v1 = np.array(v1,dtype='float32')
                    if(len(v1)==4):
                        v1 = torch.from_numpy(v1)
                    #if(v1.shape)
                        out1 = model(v1, path='half1')  # out1 is v2
                    if(len(v1)==1):  # if v1 is a tensor that is if layer >0
                        #v1 = torch.from_numpy(v1)
                    #if(v1.shape)
                        out1 = model(v1, path='half11')    
                    role_vector[member_id]=out1
                    out2 = model(out1, path='half2')  # out2 is softmax(v2TWl)
                    #out2=out2.tolist()[0] # converting tensor to list
                    f_final=torch.add(out2, f_final)
                    #f_final=[ f_final[x] + out2[x] for x in range (len (f_final))] # this is for a particular member
            #print(" ")
            ############################################## for group feature #############################
            #f_final = np.array(f_final,dtype='float32')
           #f_final = torch.from_numpy(f_final) # changing f_final
            group_features=[]
            group_features.append(g2)
            group_features.append(g3)
            group_features.append(g12)
            group_features.append(g13)
            group_features.append(g14)
            group_features = np.array(group_features,dtype='float32')
            group_features = torch.from_numpy(group_features)
            out3 = model(group_features, path='group1')
            #out3=self.layer3(group_features)
            # concatenate out3 and f_final
            out4 = torch.cat((f_final, out3),1)
            out5=self.layer4(out4) # out5 is the final output which is to be used for calculating loss  here error
        
            tag_feature_list=[]
        
            if(tag_feature==-1):
               new_tag_feature=0
            if(tag_feature==0):
               new_tag_feature=1
            if(tag_feature==1):
               new_tag_feature=2
            tag_feature_list.append(new_tag_feature)   
            label = np.array(tag_feature_list,dtype='float32')
            label = torch.from_numpy(label)
            label = label.type(torch.LongTensor)
            #out5 = out5.reshape(-1, 3) # changing shape of out5
            #loss = F.cross_entropy(out5, label)
            acc = accuracy(out5, label)
        
            return(acc)                        
        
        
       





def accuracy(outputs, label_test):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == label_test).item() / len(preds))

def evaluate(model, neighbour,groupjoin,g2,g3,g12,g13,g14,tag_feature,window,rank4_features):
    #outputs = [model.validation_step(batch) for batch in val_loader]
    outputs=model.validation_step(neighbour,groupjoin,g2,g3,g12,g13,g14,tag_feature,window,rank4_features)
    #return model.validation_epoch_end(outputs)
    return outputs.tolist() 


def fit(epochs, lr, model,neighbour,groupjoin,g2,g3,g12,g13,g14,tag_feature,window,rank4_features,opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = [] # for recording epoch-wise results
    
    for epoch in range(epochs):
        
        # Training Phase
        # the below code is for batches
        '''for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()'''
        # the below code is not for batches
        loss = model.training_step(neighbour,groupjoin,g2,g3,g12,g13,g14,tag_feature,window,rank4_features)
        #print(loss)
        start = time. time()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end = time. time()
        #print("Time to execute the optimiser is",end-start)
        return(loss)
        
        
        # Validation phase
        #result = evaluate(model, val_loader)
        #model.epoch_end(epoch, result)
        #history.append(result)

    return loss









neighbour={}



model = model1()



#########tag 20 labels ############################

#tag=pickle.load(open("tag_20_groups","rb"))
 

#tag=pickle.load(open("tag_testing_5_groups","rb"))#

tag=pickle.load(open("tag_20_groups","rb"))#  # this tag_2-groups conatins labels of just 20 groups this is used for training purpose

tag_test=pickle.load(open("tag_testing_5_groups","rb")) # this tag_2-groups conatins labels of just 5 groups this is used for testing purpose




dict1={}
###############################
length=0
for key in tag:
    length=length+len(tag[key])
    
    
print("Total number of labels in training is",length)

##################################

grp_event_common_2088_groups=pickle.load(open("grp_event_common_2088_groups","rb"))
groupjoin=pickle.load(open("groupjoin_common_2088_groups","rb"))

g2=pickle.load(open("g2_avg_distance_windowwise_tagwise_random_sample_2088","rb"))
g3=pickle.load(open("g3_variance_windowwise_tagwise_random_sample_2088","rb"))

number_of_groups=0

os.chdir('./group_wise_pickle')




g12=pickle.load(open("g12_average_window_wise_2088","rb"))
g13=pickle.load(open("g13_variance_window_wise_2088","rb"))
g14=pickle.load(open("g14_sum_window_wise_2088","rb"))

#3544262
iteration_for_neural_network=20
total_epoch=0
#os.chdir('../rank_redunced_4_features')
for iteration_number in range(iteration_for_neural_network):
    start = time. time()
    print("iteration number",iteration_number)
    for m in range(20): # if m=0 then 1st group
        print("m =",m)
        if(m==2):
            print("")
        os.chdir('../m2_number_of_event')
        
        #os.chdir('../m2_number_of_event')    
        neigbour_file="group_number"+" "+str(m)
        with open(neigbour_file, 'rb') as f:
            # file opened
            neighbour = pickle.load(f) 
        
        os.chdir('../rank_redunced_4_features_json')
        rank4_file="rank4"+" "+str(m)
        iteration=0
        for key in tag:
            if (iteration ==m):
                break;
            iteration=iteration+1    
        group_id=key
        f = open(rank4_file)
        rank4_features = json.load(f)          
        f.close
        # extracting member features from rank4 pickle files from rank_redunced_4_features folder 
        '''with open(rank4_file,'rb') as f: 
            rank4_features=pickle.load(f)'''
            
        
        
        # find the window length
        for key in rank4_features:
            window_length=len(rank4_features[key])
            break
        
        
        for window in range(window_length):        # extracting the group features 
            g12_feature=g12[group_id][window]
            g13_feature=g13[group_id][window]
            g14_feature=g14[group_id][window]
            g2_feature=g2[group_id][window]
            g3_feature=g3[group_id][window]        
            tag_feature=tag[group_id][window]
            end = time. time()
            #print("time to execute the iteration is",end - start)
            history = fit(1, 0.2, model,neighbour,groupjoin[group_id],g2_feature,g3_feature,g12_feature,g13_feature,g14_feature,tag_feature,window,rank4_features )
            #print("total_epoch",total_epoch,"history",history)
            '''for parameter in model.parameters():
                    print(parameter)'''
            total_epoch=total_epoch+1
            if((total_epoch>1733) and (total_epoch % 1733 <=30)):
                    print("epoch=",total_epoch,"group_id=",group_id,"window number",window," loss = ",history)
        
    #history1 = fit(5000, 0.01, model,neighbour,dummy_group_member,dummy_member_role,dummy_group_level_features )
    



print("End of training")


# for testing purpose after model
total_iteration=0
list_of_predicted_outputs=[]        
for m in range(5): # if m=0 then 1st group
        print("m =",m)
        
        #os.chdir('../m2_number_of_event')
        
        os.chdir('../m2_number_of_event')    
        neigbour_file="group_number"+" "+str(m+20)
        with open(neigbour_file, 'rb') as f:
            # file opened
            neighbour = pickle.load(f) 
        
        os.chdir('../rank_redunced_4_features')
        rank4_file="rank4"+" "+str(m+20)
        iteration=0
        for key in tag_test:
            if (iteration ==(m)):
                break;
            iteration=iteration+1    
        group_id=key
        rank4_features=pickle.load(open(rank4_file,"rb"))
        
        for key in rank4_features:
            window_length=len(rank4_features[key])
            break
        
        for key in rank4_features:
            #member_id=key
            member_id_feature=[]
            for window in range(window_length):
                g12_feature=g12[group_id][window]
                g13_feature=g13[group_id][window]
                g14_feature=g14[group_id][window]
                g2_feature=g2[group_id][window]
                g3_feature=g3[group_id][window]
                tag_feature=tag_test[group_id][window]
            #tag_feature=tag[key][window]
            
                
                history = evaluate(model,neighbour,groupjoin[group_id],g2_feature,g3_feature,g12_feature,g13_feature,g14_feature,tag_feature,window,rank4_features )
                #evaluate(model, neighbour,groupjoin,g2,g3,g12,g13,g14,tag_feature,window,rank4_features):
                list_of_predicted_outputs.append(history)
                if(total_iteration==3):
                    
                    break
                total_iteration=total_iteration+1
                '''for parameter in model.parameters():
                    print(parameter)'''
                print("test loss = ",history)    
                '''total_epoch=total_epoch+1
                if((total_epoch>1500) and (total_epoch % 1500 <=20)):
                    print("epoch=",total_epoch," loss = ",history)'''       
accuracy_total=0

for i in range(len(list_of_predicted_outputs)):
    accuracy_total=accuracy_total+ list_of_predicted_outputs[i]   
print(accuracy_total)        


                
                
               
        
        
           
            
            
            

    
    
