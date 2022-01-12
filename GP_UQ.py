# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:37:13 2021

@author: Chensen_Ding
"""
# In[1] Importing Libraries
from src.vtkReader import vtkReader
import numpy as np
import scipy.io
from scipy.linalg import svd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
import meshio

# In[2] Importing data
print('Importing data...')
# Simple example reading Jean's vtk file
# Put ''.vtk'' files in folder "dune_files". Here I have just made a copy of the one provided by Jean, so we have more than one.
# Here is the code, it reads in the mesh to start with from the first file, it then checks it is the same in the other files.
# It then reads in the solutions which for this one is either the solution or the stress. Here you can get these
# using : data.solution["U"] - gives a list of solution for each file data.solution["STRESS"] - gices a list of stress for each file
# so if you want solution in file 0 you would have data.solution["U"][0] - this returns numNodes by 3 matrix for the displacement solution
# data = vtkReader()
# folder_path = "dune_files/"
# data.readMeshes(folder_path)
# data.readSolutions()
# Displacement_file1 = data.solution["U"][0]

# Initial model and solution
data_input = vtkReader()
folder_path = "dune_files/"
data_input.readMeshes(folder_path)
data_input.readSolutions()

xFeatures = np.array(data_input.inputfeatures["gcoord"]) # Feature
Displacement = np.array(data_input.solution["U"])        # Response

# Generating testing and testing data
NumSample, NumNodes, NumDimen = np.shape (xFeatures)    

NumTraining = 12 # Number of testing sample
if NumTraining < NumSample:
    NumPred = NumSample - NumTraining
else:
    print('Error: Training sample exceeds total!')

Num_indexTraining = [i for i in range(NumSample)] 
np.random.shuffle(Num_indexTraining)

Index_training = Num_indexTraining[0:NumTraining]
Index_testing = Num_indexTraining[NumTraining:]

# data for GP    
xFeatures_data = xFeatures.reshape (NumSample,-1)
Displacement_data = Displacement.reshape(NumSample,-1)

xFeatures_training = xFeatures_data[Index_training,:]   # Training input
xFeatures_pred = xFeatures_data[Index_testing,:]         # Training output

Displacement_training = Displacement_data[Index_training,:]  # Testing input
Displacement_ref = Displacement_data[Index_testing,:]        # Testing output (Reference)

# # Export training and testing data to Matlab
# scipy.io.savemat('C_section.mat', mdict={ 'xFeatures_training':xFeatures_training,
#                                     'xFeatures_pred':xFeatures_pred,
#                                     'Displacement_training':Displacement_training,
#                                     'Displacement_ref':Displacement_ref})

# In[3] Decouple the correlation amoung inputs and output data via SVD
print ('Decoupling the data via SVD...')
# Decouple input correlation
U_in, s_in, V_in= svd(xFeatures_training.T,full_matrices=False)
plt.plot(np.square(s_in)/np.sum(np.square(s_in)),'r*')
# Determin the reduced dimension of U_in
tor = 1- 10**-10
for i in range(s_in.shape[0]):
    percent = np.sum(np.square(s_in[0:i+1])) / np.sum(np.square(s_in))
    if percent >= tor:
        reducedBasisNoIn = i+1
        print ("reduced basis of Input:", U_in.shape[0], i + 1)
        break
Phi_in= U_in[:,:reducedBasisNoIn]
plt.show()

# Decouple output correlation
U_out, s_out, V_out= svd(Displacement_training.T,full_matrices=False)
plt.plot(np.square(s_out)/np.sum(np.square(s_out)),'bo')
# Determin the reduced dimension of U_out
tor = 1- 10**-10
for i in range(s_out.shape[0]):
    percent = np.sum(np.square(s_out[0:i+1])) / np.sum(np.square(s_out))
    if percent >= tor:
        reducedBasisNoOut = i+1
        print ("Reduced basis of Output:", U_out.shape[0], i + 1)
        break
Phi_out= U_out[:,:reducedBasisNoOut]
plt.show()

# In[4] Transfer initial map to the map from input basis coefficients of  to output basis coefficients
# Input basis coefficients 
reducedCoefIn_training = np.zeros((xFeatures_training.shape[0],reducedBasisNoIn))  # Features * samples
reducedCoefIn_testing = np.zeros((xFeatures_pred.shape[0],reducedBasisNoIn))  # Features * samples
for i in range(reducedBasisNoIn):
    reducedCoefIn_training[:,i] =np.dot(Phi_in[:,i].T,xFeatures_training.T)
    reducedCoefIn_testing[:,i] =np.dot(Phi_in[:,i].T,xFeatures_pred.T)
    
# Output basis coefficients 
reducedCoefOut = np.zeros((Displacement_training.shape[0],reducedBasisNoOut))  # Features * samples
for i in range(reducedBasisNoOut):
    reducedCoefOut[:,i] =np.dot(Phi_out[:,i].T,Displacement_training.T)

# Generating data(reduced coefficients) for GP
x_training = reducedCoefIn_training
y_training = reducedCoefOut

x_testing = reducedCoefIn_testing

    # Number of output feature/parameter, training samples, predicating samples
Num_trainingSamp, Num_yFeature = np.shape(y_training)
Num_testSamp = x_testing.shape[0]

# In[4] Build GP regression
print('Building GP...')
# kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
kernels = [ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)),
           1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 10.0)),
            1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
            1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                length_scale_bounds=(0.1, 10.0),
                                periodicity_bounds=(1.0, 10.0)),
            ConstantKernel(0.1, (0.01, 10.0))
                * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
            1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                        nu=1.5)]

# Initializing GP 
GPR_cell = np.empty((Num_yFeature),dtype=object)
for i in range(Num_yFeature):      # For Each Feature/parameter, each Feature has each GP
    GPR_cell[i] = GaussianProcessRegressor(ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3)))
    # GPR_cell[i] = GaussianProcessRegressor(kernel=kernels[4],
    #                                 alpha               = 1e-10,
    #                                 copy_X_train        = True,
    #                                 optimizer           = "fmin_l_bfgs_b",
    #                                 n_restarts_optimizer= 25,
    #                                 normalize_y         = False,
    #                                 random_state        = None)   
                               
# Fit to data using Maximum Likelihood Estimation of the parameters
    y_trainingI = y_training[:,i].reshape(-1,1) # Each Feature has each GP
    GPR_cell[i].fit(x_training, y_trainingI)  
    print (i)
        
# In[5] Predicating 
print('Predicating GP...')

# Predicate the reduced coefficients
Pred_mean = np.zeros((Num_testSamp,Num_yFeature))
Pred_std = np.zeros((Num_testSamp,Num_yFeature))
for i in range(Num_testSamp):      # For Each predicating sample
    x_testingI = x_testing [i,:].reshape(1,-1)
    for j in range (Num_yFeature):
        y_pre, y_std= GPR_cell[j].predict(x_testingI, return_std=True)  
        
        Pred_mean [i,j] = y_pre
        Pred_std [i,j] = y_std

Pred_ref =  np.dot(Phi_out.T,Displacement_ref.T)

# Predicate the corresponding full field displacement
GP_pred_disp_mean = np.dot(Phi_out,Pred_mean.T)
GP_pred_disp_std = np.dot(Phi_out,Pred_std.T)

# Error between GP and Reference displacement
a_array =  GP_pred_disp_mean - Displacement_ref.T
b_array = Displacement_ref.T
Error = np.divide(a_array, b_array, out=np.zeros_like(a_array, dtype=np.float64), where=b_array!=0)

# Find where the large error appears
print(np.argmax(Error, axis=0)) # 按每列求出最小值的索引
print(np.argmin(Error, axis=0)) # 按每行求出最小值的索引
print ('Finished')
a_array[np.argmax(Error, axis=0)]

# In[7] Plot results
# Compare the Reduced coefficients of testing output
x_axis = np.arange(0, Num_yFeature)+1
Pred_Upper = Pred_mean + 3*Pred_std
Pred_Low = Pred_mean - 3*Pred_std
for i in range (Num_testSamp):    
    # plt.subplot (2,Num_testSamp/2,i+1)
    # plt.suptitle('Reduced basis coefficents')
    plt.title('Sample %d'%(i+1))
    plt.plot (x_axis, Pred_mean[i,:],'k*',linestyle = "-",label='GP_mean')
    plt.plot (x_axis, Pred_Upper[i,:],'g',linestyle = "--",label='GP_upper(99%)')
    plt.plot (x_axis, Pred_Low[i,:],'b',linestyle = "--",label='GP_lower(99%)')   
    plt.plot (x_axis, Pred_ref[:,i],'r',linestyle = "--",label='Ref')    
    plt.xlabel('Index of features')    
    plt.legend()
    plt.grid()
    plt.show()

# # Compare the Reduced coefficients of testing output
# x_axis = np.arange(0, Num_yFeature)+1
# for i in range (Num_testSamp):    
#     plt.subplot (2,Num_testSamp/2,i+1)
#     plt.suptitle('Reduced basis coefficents')
#     plt.title('Sample %d'%(i+1))
#     plt.plot (x_axis, Pred_mean[i,:],'ro',linestyle = "--",label='GP')
#     plt.plot (x_axis, Pred_ref[:,i],'b*',linestyle = "--",label='Ref')
#     plt.xlabel('Index of reduced basis coefficents')        
#     plt.legend()
#     plt.grid()
# plt.show()

# # Compare the Displacement of testing output
# x_axis = np.arange(0, np.size(GP_pred_disp_mean,0))+1
# for i in range (Num_testSamp):    
#     plt.subplot (2,Num_testSamp/2,i+1)
#     plt.suptitle('Full-field displacement')
#     plt.plot (x_axis, GP_pred_disp_mean[:,i],'ro', label='GP')
#     plt.plot (x_axis, Displacement_ref[i,:],'b*', label='Ref')    
#     plt.xlabel('Index of degree of freedom')    
#     plt.legend()
#     plt.grid()         
# plt.show()

# Z = Error
# size=Z.shape
# Y=np.arange(0,size[0],1)     
# X=np.arange(0,size[1],1)
# X,Y=np.meshgrid(X,Y)    

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm,
#                        linewidth=0, antialiased=False)
# # Customize the z axis.
# # ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(plt.LinearLocator(10))
# ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))

# In[7] Export to vtk. file
for i in range (Num_testSamp):       
    points = xFeatures[Index_testing[i],:,:] # gcoord 
    cells = [("hexahedron",data_input.elements)]
    # Reference solution
    point_data = {"Disaplacement_Reference": Displacement[Index_testing[i],:,:]}
    meshio.Mesh(points,cells,point_data).write("C-section_Ref%d.vtk" %(i))
    # GP solution 
    Displacement_GP = GP_pred_disp_mean[:,i].reshape(NumNodes, NumDimen)
    point_data = {"Disaplacement_GP": Displacement_GP}
    meshio.Mesh(points,cells,point_data).write("C-section_GP%d.vtk" %(i)) 
    # Error between Reference and GP solution     
    Error = (Displacement[Index_testing[i],:,:]-Displacement_GP)
    point_data = {"Disaplacement_Error": Error}
    meshio.Mesh(points,cells,point_data).write("C-section_Eorror%d.vtk" %(i))
    
#!!!!!!!!!!! Please Check    
    # Relative Error between Reference and GP solution     
    Error = (Displacement[Index_testing[i],:,:]-Displacement_GP)/Displacement[Index_testing[i],:,:]
    Error[np.isnan(Error)]=0
    Error[np.isinf(Error)]=0
    point_data = {"Disaplacement_RelativeError": Error}
    meshio.Mesh(points,cells,point_data).write("C-section_RelativeEorror%d.vtk" %(i))
    # L2 Error between Reference and GP solution     
    ErrorL2 = (Displacement[Index_testing[i],:,:]-Displacement_GP)**2/(Displacement[Index_testing[i],:,:])**2
    ErrorL2[np.isnan(ErrorL2)]=0
    ErrorL2[np.isinf(ErrorL2)]=0        # /np.linalg.norm(Displacement[Index_testing[i],:,:])
    point_data = {"Disaplacement_ErrorL2": ErrorL2}
    meshio.Mesh(points,cells,point_data).write("C-section_RelativeEorror%d.vtk" %(i))
        
    
plt.show()