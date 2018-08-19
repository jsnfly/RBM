# -*- coding: utf-8 -*-
"""
Created on Mon May 07 12:48:31 2018

@author: schillam
"""

import numpy as np




########################################## functions to calculate dim normed distance matrix

def calculate_dim_pairwise_distances(mat,label_vec):
    dists=[]
    label_combi=[]
    for i in range(len(mat)):
        for j in np.arange(i+1, len(mat)):
            dists.append(np.sqrt((mat[i]-mat[j])**2))
            label_combi.append([label_vec[i],label_vec[j]])
    #print len(dists), dists
    #hallo        
    return np.array(dists), np.array(label_combi)    


def dim_normed_dist_vec(data_mat,label_vec):
    n=len(data_mat[:,0])  
    
    if len(data_mat[:,0])!=len(label_vec):
        error('Labelsize does not fit data size')
    
    dim_size=np.shape(data_mat)[1]
    
    dim_distances=np.zeros([(n*(n-1))/2,dim_size])    
    dim_norm_distances=np.zeros([(n*(n-1))/2,dim_size]) 
    
    for d in np.arange(dim_size):
        dim_distances[:,d],label_combi=calculate_dim_pairwise_distances(data_mat[:,d],label_vec)
        dim_norm_distances[:,d]=dim_distances[:,d]/np.mean(dim_distances[:,d])  #dimensionweises normieren auf den mittlere Abstand i.e teile Abstand in jeder Dimension 
        
    #print dim_distances, dim_norm_distances
    #hallo
    dist_vec=np.sqrt(np.sum(dim_norm_distances**2,axis=1))
     

    return dist_vec, label_combi, dim_size


#############################################################functions to calculate distnace matrix with dimension z-transfomed data points

def dim_wise_z_transform(data_mat, dim_size):
    for i in range(dim_size):
        mean = np.mean(data_mat[:,i])
        std_dev = np.std(data_mat[:,i])
        data_mat[:,i]=(data_mat[:,i]-mean)/std_dev
    return data_mat


def calculate_pairwise_distances(mat,label_vec):
    dists=[]
    label_combi=[]
    #print 'hallo'
    for i in range(len(mat)):
        for j in np.arange(i+1, len(mat)):
            dists.append(np.sqrt(sum((mat[i,:]-mat[j,:])**2)))
            label_combi.append([label_vec[i],label_vec[j]])
#         print(i)    
    #print len(dists), dists
    #hallo        
    return np.array(dists), np.array(label_combi)    




def calculate_dist_vec(data_mat, label_vec):
    n=len(data_mat[:,0])  
    
    if len(data_mat[:,0])!=len(label_vec):
        error('Labelsize does not fit data size')
    
    dim_size=np.shape(data_mat)[1]
    
    data_mat=dim_wise_z_transform(data_mat, dim_size)
    
    
    dist_vec, label_combi = calculate_pairwise_distances(data_mat,label_vec)


    return dist_vec, label_combi, dim_size     










############################################################## functions to calculate discrimination value 
    

def intra_value(dist_vec, label_combi,label):
    temp=0
    count=0
    #hallo
    for i in range(len(dist_vec)):
        if np.array_equal(label_combi[i],[label,label]):
            count=count+1
            #print label_combi[i],label, dist_vec[i]
            temp+=dist_vec[i]
    if temp==0:
        count=1        
    return temp/count       
        
def calculate_intra_sum(labels, dist_vec, label_combi):
    intra_sum=0 # sum of intercluster distances
    for i in range(len(labels)):
        intra_sum+=intra_value(dist_vec, label_combi,labels[i])
    return intra_sum


def extra_value(dist_vec, label_combi,combination_of_label):
    temp=0
    count=0
    #print 'hallo' ,combination_of_label[::-1]
    for i in range(len(dist_vec)):
        #print dist_vec[i]
        if np.array_equal(label_combi[i],combination_of_label) or np.array_equal(label_combi[i],combination_of_label[::-1]):
            count=count+1
            temp+=dist_vec[i]        
    return temp/count  





def calculate_extra_sum(labels,dist_vec,labels_combi):
    combi_temp=[]    
    for i in range(len(labels)):
        for j in np.arange(i,len(labels)):
            combi_temp.append([labels[i],labels[j]])
    
    combi_temp2=[]
    for i in range(len(combi_temp)):
        if combi_temp[i][0]!=combi_temp[i][1]:
            #print 'combi', combi_temp[i]
            combi_temp2.append(combi_temp[i])  
    
    combi_temp2=np.array(combi_temp2)       
    
    extra_sum= 0   
    
    for j in range(len(combi_temp2[:,0])):
        temp=extra_value(dist_vec, labels_combi,combi_temp2[j,:])
        if temp is None:
            pass
        else:
            extra_sum+=temp
    return extra_sum        
        
    
    
def discrimination_value_calc(dist_vec, label_combi,label_vec,dim_size):
    labels=np.unique(label_vec)
    
    intra_sum=calculate_intra_sum(labels, dist_vec, label_combi)
    
    extra_sum=calculate_extra_sum(labels,dist_vec,label_combi)
    
    L=len(labels)
    
    disc_value =np.sqrt(1.0/dim_size)*(intra_sum-(2.0/(L-1.0))*extra_sum)
    
    
    print('intra_sum', intra_sum, 'extra_sum', extra_sum, 'discrimination_value', disc_value)
    
    return disc_value

########################################################## main function        
def discrimination_value(data_mat,label_vec,norm):
    if norm=='dim':  
        dist_vec, label_combi, dim_size = dim_normed_dist_vec(data_mat,label_vec)
        disc_value=discrimination_value_calc(dist_vec, label_combi,label_vec,dim_size) 
    if norm=='z':
        # remove zero dimensions:
        #######
        non_zero_dim_indices = []
        for i in range(data_mat.shape[1]):
            std_dev = np.std(data_mat[:,i])
            if std_dev != 0:
                non_zero_dim_indices.append(i)
        print('num zero dimensions: ', data_mat.shape[1]-len(non_zero_dim_indices))
        data_mat = data_mat[:,non_zero_dim_indices]
        #######
        dist_vec, label_combi, dim_size = calculate_dist_vec(data_mat, label_vec)
        disc_value=discrimination_value_calc(dist_vec, label_combi,label_vec,dim_size) 
    
    return disc_value    

     
##########################################################     
#    
#a=np.array([[1,1],[1.1,1.1],[1.2,1.2]])
#b=np.array([[0,0],[0.1,0.1],[0.2,0.2]])
##data=np.array([[1,1.1],[1.1,1.1],[0.1,0.2],[0,0],[2.1,2.1],[2.2,2.2]])
#
##label_vec=np.array([1,1,2,2,3,3])
#data=np.array([[0.1,0.0,0.0],[0.1,0.0,0.0],[-0.0,-0.1,-0.0],[-0.0,-0.1,-0.0],[-0.0,-0.0,-1],[-0.0,-0.0,-1]])
#data=np.array([[0.1,0.1],[0.0,0.0]])
#label_vec=np.array([2,2,1,1,3,3])
#label_vec=np.array([2,1])
#
#data=np.random.sample([100,100])
#data[0:50]=data[0:50]+np.ones([50, 100])*100
#label_vec=np.zeros(100)
#print 'Random'
#label_vec[20:70]=1
#
#
#
##disc_dim=discrimination_value(data,label_vec,'dim')
#disc_z=discrimination_value(data,label_vec,'z')
#
#print 'ooooooooooooooooooooooooooooooooooooooooooo'
#print disc_z

