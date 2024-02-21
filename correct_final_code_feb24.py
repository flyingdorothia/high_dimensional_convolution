import numpy as np
from numpy.linalg import norm, inv
import time
import math
from tensorflow import keras
from matplotlib.pyplot import imread
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


def conv_by_definition_Nov18(image, kernel_given):
    kernel_given = np.flip(kernel_given)
    mat_should = np.zeros(image.shape, dtype = np.complex)
    input_image = np.pad(image,((1, 1), (1, 1)), 'constant')
    for xi in range(image.shape[0]): 
        for yi in range(image.shape[1]):
            img_patch = input_image[xi:xi+3, yi:yi+3]
            mat_should[xi, yi] = np.sum(kernel_given*img_patch)
    return mat_should


def kernel_to_array_0217(in_dim, kernel_given):
    # it is only how we address it.
    k_dim = kernel_given.shape[0]
    z_dim = in_dim - k_dim
    # now we know how many columns we need to do
    kernel_arr_pos = np.concatenate([[kernel_given[1,2]], np.zeros(z_dim), kernel_given[2,:]])
    kernel_arr_neg = np.concatenate([kernel_given[0,:], np.zeros(z_dim), [kernel_given[1,0]]])
    kernel_arr_neg = kernel_arr_neg[::-1]
    kernel_0 = kernel_given[1,1]
    return kernel_arr_neg, kernel_arr_pos, kernel_0
# so we don't need to do the flip anymore. Just let it be


def populate_matrix_posNeg(dim, XvecNeg, XvecPos, X0):
    Xmat = np.identity(dim)*X0
    i = np.arange(0, dim)
    for ik in range(XvecPos.shape[0]):
        i1 = i[:-ik-1]
        Xmat[i1, i1+ik+1] = XvecNeg[ik]
        Xmat[i1+ik+1, i1] = XvecPos[ik]
    return Xmat


def givenKerBig_conv_results_Nov18(image, kernel_big_mat_final):
    input_image = np.pad(image,((1, 1), (1, 1)),'constant')
    img_arr = input_image.flatten()
    mat_ops_arr_final = kernel_big_mat_final.dot(img_arr)

    mat_output = mat_ops_arr_final.reshape(-1, input_image.shape[1])
    mat_output = mat_output[1:-1,1:-1]
    return mat_output, mat_ops_arr_final


# inverse part

def expand_Xvec_0718(XvecPos, XvecNeg, X0, theta):
    XvecNeg = XvecNeg[::-1]
    totalK = XvecPos.shape[0] + XvecNeg.shape[0] + 1
    kSeries = np.linspace(-XvecNeg.shape[0], XvecPos.shape[0], totalK)
    kSeries = np.reshape(kSeries, (-1, 1))
    theta = theta.reshape(1, -1)
    x_complex = np.concatenate((XvecNeg, [X0], XvecPos), axis=0)
    x_complex = np.reshape(x_complex, (-1, 1))
    output = x_complex*np.exp(1j*kSeries*theta)
    return np.sum(output, axis=0)


def inv_trans_matrix_0718(XvecPos, XvecNeg, X0, YDim):
    N_theta = 40000
    theta_sample = np.linspace(0, 2*np.pi, N_theta, endpoint = False) + 1E-3
    theta_sample = theta_sample.reshape(-1, 1)
    yn = np.arange(0, YDim-1)
    yn = yn.reshape(1, -1)
    denom = expand_Xvec_0718(XvecPos, XvecNeg, X0, theta_sample)
    denom = denom.reshape(-1, 1)
    vec_tempPos = np.exp(-1j*(yn+1)*theta_sample)/denom
    vec_tempNeg = np.exp(1j*(yn+1)*theta_sample)/denom

    YposVec = 1/2/np.pi*np.trapz(vec_tempPos, theta_sample, axis = 0)
    YnegVec = 1/2/np.pi*np.trapz(vec_tempNeg, theta_sample, axis = 0)
    theta_sample = np.linspace(0, 2*np.pi, N_theta, endpoint = False) + 1E-3
    vec_tempZero = 1/expand_Xvec_0718(XvecPos, XvecNeg, X0, theta_sample)
    Y0 = 1/2/np.pi*np.trapz(vec_tempZero, theta_sample, axis = 0)
    return YposVec, YnegVec, Y0

# get into the angle part and without angle


def from_S_get_kappa_0718(kernel_arr_pos, kernel_arr_neg, kernel_0, norm_fac, YDim):
    kernel_0 = kernel_0/norm_fac - 1
    kernel_arr_pos = kernel_arr_pos/norm_fac
    kernel_arr_neg = kernel_arr_neg/norm_fac
    YposVec, YnegVec, Y0 = inv_trans_matrix_0718(kernel_arr_pos, kernel_arr_neg, kernel_0, YDim)
    kappa0 = 1j*Y0*2
    kappa_pos = 1j*YposVec*2
    kappa_neg = 1j*YnegVec*2
    return kappa_pos, kappa_neg, kappa0



# done
def find_modulate_AmpPha_0718_without_angle(kappa_pos, kappa_neg, kappa0):
    # kappa matrix itself
    A_list = np.zeros(len(kappa_pos))
    B_list = np.zeros(len(kappa_pos))
    comp_scale = np.max(abs(kappa_pos))
    # now we take the phase as some fixed value
    
    for m in np.arange(0, len(kappa_pos)):
        kpm = kappa_pos[m]
        kmm = kappa_neg[m]
        A_list[m] = np.real(-1j*(kpm + np.conjugate(kmm)))
        B_list[m] = np.real(-1j*(kpm - np.conjugate(kmm)))
    
    gamma0 = 1j*kappa0-1
    return A_list, B_list, gamma0



def forward_build_Xvec_0718_without_angle(A_list, B_list, gamma0):
    kappa_pos = np.zeros(A_list.shape[0], dtype=complex)
    kappa_neg = np.zeros(A_list.shape[0], dtype=complex)
    for mi in range(kappa_pos.shape[0]):
        Am = A_list[mi]
        Bm = B_list[mi]
        kappa_pos[mi] = Am/2*np.exp(1j*np.pi/2) + Bm/2*np.exp(1j*np.pi/2)
        kappa_neg[mi] = Am/2*np.exp(-1j*np.pi/2) - Bm/2*np.exp(-1j*np.pi/2)
    kappa0 = -1j*(gamma0+1)
    
    return kappa_pos, kappa_neg, kappa0
    
def from_kappa_get_S_0718(kappa_pos, kappa_neg, kappa0, norm_fac, dim):
    
    kernel_arr_posBuild, kernel_arr_negBuild, kernel_0Build = inv_trans_matrix_0718(kappa_pos, kappa_neg, kappa0, dim)

    kernel_0Build = (2j*kernel_0Build+1)*norm_fac
    kernel_arr_posBuild = 2j*kernel_arr_posBuild*norm_fac
    kernel_arr_negBuild = 2j*kernel_arr_negBuild*norm_fac

    return kernel_arr_posBuild, kernel_arr_negBuild, kernel_0Build


# with angle

def find_modulate_AmpPha_0718_with_angle(kappa_pos, kappa_neg, kappa0):
    # kappa matrix itself
    A_list = np.zeros(len(kappa_pos), dtype = complex)
    B_list = np.zeros(len(kappa_pos), dtype = complex)
    alpha_list = np.zeros(len(kappa_pos))
    beta_list = np.zeros(len(kappa_pos))
    # now we take the phase as some fixed value
    
    for m in np.arange(0, len(kappa_pos)):
        
        kpm = kappa_pos[m]
        kmm = kappa_neg[m]
        
        alpha_list[m] = np.angle(kpm + np.conjugate(kmm))
        beta_list[m] = np.angle(kpm - np.conjugate(kmm))
        
        A_list[m] = (kpm + np.conjugate(kmm))/np.exp(1j*alpha_list[m])# in units of t_R gamma_e
        B_list[m] = (kpm - np.conjugate(kmm))/np.exp(1j*beta_list[m])# in units of t_R gamma_e
        
    gamma0 = 1j*kappa0-1      # in units of gamma_e
    
    return A_list, B_list, alpha_list, beta_list, gamma0


def forward_build_Xvec_0718_with_angle(A_list, B_list, alpha_list, beta_list,  gamma0):
    
    kappa_pos = np.zeros(A_list.shape[0], dtype=complex)
    kappa_neg = np.zeros(A_list.shape[0], dtype=complex)
    
    for mi in range(kappa_pos.shape[0]):
        Am = A_list[mi]
        Bm = B_list[mi]
        alpha_m = alpha_list[mi]
        beta_m = beta_list[mi]
        
        kappa_pos[mi] = Am/2*np.exp(1j*alpha_m) + Bm/2*np.exp(1j*beta_m)
        kappa_neg[mi] = Am/2*np.exp(-1j*alpha_m) - Bm/2*np.exp(-1j*beta_m)
        
    kappa0 = -1j*(gamma0+1)# all right, checked. 
    
    return kappa_pos, kappa_neg, kappa0


# causality

def causality_metric(gamma0, B_list, beta_list):
    t_list = np.arange(0, 2*np.pi, 0.01)
    result_list = np.zeros(t_list.shape, dtype = complex)
    
    # for all the t find the max of this
    index_list = np.arange(0, len(B_list),1)+1
    for ii, t in enumerate(t_list):
        result_list[ii] = sum(B_list*np.sin(index_list*t + beta_list))
    constraint = max(result_list)
    return gamma0 + 1/2 > constraint



def input_output_image_conv_large(input_image, L, K):

    W = input_image.shape[1]
    H = input_image.shape[0]
    dim = int((W-(K - 1))/(L-(K - 1)))*L*H

    loop_times = int((W - (K - 1))/(L-(K - 1)))
    stride = int(L - (K - 1))
#     print(dim)

    i = np.arange(0, dim)
#     print(loop_times)
    for i in range(loop_times):
        arr1 = input_image[:, stride*i:L+stride*i].flatten()
        if i == 0:
            arr_input = arr1
        else:
            arr_input = np.concatenate((arr_input, arr1))
            
    return arr_input, dim


def input_output_slice_output_0301(Smat, arr_input, K, input_image, L):
    H = input_image.shape[0]
    W = input_image.shape[1]
    brr = Smat.dot(arr_input)
    
    # so far so good
    shift_step = int((K-1)/2)
    loop_times = int((W - (K - 1))/(L-(K - 1)))
    stride = int(L - (K - 1))
#     print(loop_times)
    for j in range(loop_times):
        d0 = j*H*L
#         print(H-(K-1))
        for i in range(H-(K-1)):
#             print(i)
#             print(np.arange(d0+(i+shift_step)*L+shift_step, d0+(i+shift_step)*L+stride+shift_step))
            arr1 = brr[d0+(i+shift_step)*L + shift_step: d0+(i+shift_step)*L+stride + shift_step]
#             print(arr1)
            if i == 0:
                arr = arr1
            else:
                arr = np.vstack((arr, arr1))
                
#         print(arr)
        if j == 0:
            bmat = arr
        else:
            bmat = np.hstack((bmat, arr))
# now this code is not very ideal
    return brr, bmat



def kernel_flatten_3D(input_image, kernel_given):
    P1 = int((kernel_given.shape[0]-1)/2)
    P2 = int((kernel_given.shape[1]-1)/2)
    P3 = int((kernel_given.shape[2]-1)/2)
    
    L1 = input_image.shape[0]
    L2 = input_image.shape[1]
    L3 = input_image.shape[2]
    s_pos = np.zeros(P1*L3*L2 + P2*L3 + P3)
    s_neg = np.zeros(P1*L3*L2 + P2*L3 + P3)
    
    for p in np.arange(-P1, P1+1):
        for q in np.arange(-P2, P2+1):
            for t in np.arange(-P3, P3+1):
                
                m = p*L3*L2 + q*L3 + t
#                 print(p, q, t,m)
#                 print(p, q, t,m, P1-p, P2-q, P3-t)
                if m > 0:
                    s_neg[m-1] = kernel_given[P1-p, P2-q, P3-t]
                elif m < 0:
                    s_pos[-m-1] = kernel_given[P1-p, P2-q, P3-t]
                    
                else:
                    s0 = kernel_given[P1, P2, P3]
                    
    return s_neg, s_pos, s0



def conv_by_definition_3D_Mar7(image, kernel_given):
    kernel_new = np.flip(kernel_given)
    mat_should = np.zeros(image.shape, dtype = np.complex)
    input_image = np.pad(image,((1, 1), (1, 1), (1, 1)), 'constant')
    for xi in range(image.shape[0]): 
        for yi in range(image.shape[1]):
            for zi in range(image.shape[2]):
                img_patch = input_image[xi:xi+3, yi:yi+3, zi:zi+3]
                mat_should[xi, yi, zi] = np.sum(kernel_new*img_patch)
    return mat_should



def plot_matrix(mat, figsize, cmap=plt.cm.coolwarm):
    f = plt.figure(figsize=figsize)
    ax = plt.axes([0.05, 0.05, 0.6, 0.6]) #left, bottom, width, height
    #note that we are forcing width:height=1:1 here, 
    #as 0.9*8 : 0.9*8 = 1:1, the figure size is (8,8)
    #if the figure size changes, the width:height ratio here also need to be changed
    im = ax.imshow(mat, interpolation='nearest', cmap=cmap)
    ax.grid(False)
    cax = plt.axes([0.7, 0.05, 0.05, 0.6])
    plt.colorbar(mappable=im, cax=cax)
    return ax, cax