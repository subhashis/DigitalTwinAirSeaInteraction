import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import netCDF4 as nc
#from saveNCfile import savenc


def load_test_data(FF,lead):

  SSH = np.asarray(FF['msshf'])
  ocean_grid_size = np.count_nonzero(~np.isnan(SSH[0,:,:]))
##### convert Nans to zero ####
  SSH[np.isnan(SSH)]=0.0
################################
  SSH=SSH[0:,:,:]
  
  uv = np.zeros([np.size(SSH,0),1,np.size(SSH,1),np.size(SSH,2)])

  uv [:,0,:,:] = SSH

  uv_test_input = uv[0:np.size(SSH,0)-lead,:,:,:]
  uv_test_label = uv[lead:np.size(SSH,0),:,:,:]
 


## convert to torch tensor
  uv_test_input_torch = torch.from_numpy(uv_test_input).float()
  uv_test_label_torch = torch.from_numpy(uv_test_label).float()

  return uv_test_input_torch, uv_test_label_torch,ocean_grid_size


def load_test_data_DTAS(FF, lead, varname):

  AS_field = np.asarray(FF[varname])
  ocean_grid_size = np.count_nonzero(~np.isnan(AS_field[0,:,:]))
  ##### convert Nans to zero ####
  AS_field[np.isnan(AS_field)]=0.0
  ################################
  AS_field=AS_field[0:,:,:]
  
  uv = np.zeros([np.size(AS_field,0),1,np.size(AS_field,1),np.size(AS_field,2)])

  uv [:,0,:,:] = AS_field

  uv_test_input = uv[0:np.size(AS_field,0)-lead,:,:,:]
  uv_test_label = uv[lead:np.size(AS_field,0),:,:,:]
 


## convert to torch tensor
  uv_test_input_torch = torch.from_numpy(uv_test_input).float()
  uv_test_label_torch = torch.from_numpy(uv_test_label).float()

  return uv_test_input_torch, uv_test_label_torch,ocean_grid_size

def load_train_data(GG, lead,trainN):
  
     ucur=np.asarray(GG['msshf'])
#### remove NANS###########
     ocean_grid_size = np.count_nonzero(~np.isnan(ucur[0,:,:]))

     SSH=np.asarray(GG['msshf'])
     SSH[np.isnan(SSH)]=0
     SSH=SSH[0:trainN,:,:]

     uv = np.zeros([np.size(SSH,0),1,np.size(SSH,1),np.size(SSH,2)])
     uv[:,0,:,:] = SSH

     uv_train_input = uv[0:np.size(SSH,0)-lead,:,:,:]
     uv_train_label = uv[lead:np.size(SSH,0),:,:,:]

     uv_train_input_torch = torch.from_numpy(uv_train_input).float()
     uv_train_label_torch = torch.from_numpy(uv_train_label).float()

     return uv_train_input_torch, uv_train_label_torch, ocean_grid_size


def load_train_data_DTAS(GG, lead, trainN, varname):
  
     ucur=np.asarray(GG[varname])
     #### remove NANS###########
     ocean_grid_size = np.count_nonzero(~np.isnan(ucur[0,:,:]))

     AS_field=np.asarray(GG[varname])
     AS_field[np.isnan(AS_field)]=0
     AS_field=AS_field[0:trainN,:,:]

     uv = np.zeros([np.size(AS_field,0),1,np.size(AS_field,1),np.size(AS_field,2)])
     uv[:,0,:,:] = AS_field

     uv_train_input = uv[0:np.size(AS_field,0)-lead,:,:,:]
     uv_train_label = uv[lead:np.size(AS_field,0),:,:,:]

     uv_train_input_torch = torch.from_numpy(uv_train_input).float()
     uv_train_label_torch = torch.from_numpy(uv_train_label).float()

     return uv_train_input_torch, uv_train_label_torch, ocean_grid_size