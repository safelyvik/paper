##util_target_detection
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import hadamard
from scipy.stats import norm

N = 64                                                  #number of RIS elements
et=0.01                                                 #transmit energy
f = 3e9                                                 #carrier frequency
fs = 25e6 
B=25e6
tp=1.28e-6                                              #sampling frequency
L = 32                                                  #fast time sample number tp*fs    
P = 0.25e6                                              #transmit power
c = 3e8                                                 #speed of light
theta_r_RIS = 45                                        # angle between radar and normal of RIS

lambda_ = c / f                                         # Wavelength
#d_RIS = lambda_/2  
d_RIS = 0.6                                      #spacing bw RIS elements
sigma_RIS_n = 4 * np.pi * N * d_RIS**2 / lambda_**2     # RCS of the RIS n-th element
sigma_E=1                                               #exceptation of target RCS(0dbm)
sigma_E_k=10**-2                                        #expectation of clutter RCS(-20dbm)
Gt = 1                                                  #Gain of target,RIS and clutter
Gris = 1
G_c_k = 1

num_epochs = 1000                                    # Max epoch
batch_size = 500        # Number of samples in batch
learning_rate = 5e-3                                    # Learning rate
lr_decay_epoch = 200
K=1                                                     #number of clutters
a = 1                                                   # Constant (not used in the computation but kept for consistency)

#transmit signal
#H = torch.tensor(hadamard(L))    # Generate the Hadamard matrix
#s_old = torch.sqrt(torch.tensor(et / L)) * H[:,14]   # Compute s
#s = s_old.unsqueeze(1).repeat(batch_size, 1, 1)
#s_complex = s.to(dtype=torch.complex64)

Ks = 25e6/1.28e-6


# Generate n values for each sequence
n_values = torch.arange(L, dtype=torch.float32).view(1, -1, 1)  # Shape: (1, N, 1)

# Expand n_values to match batch size
n_values = n_values.expand(batch_size, -1, -1)  # Shape: (500, 32, 1)

# Compute the sequence s_0,n for each batch
s_complex = (1 / torch.sqrt(torch.tensor(L, dtype=torch.float32))) * \
        torch.exp(1j * torch.pi * Ks * (n_values / fs) ** 2)  # Shape: (500, 32, 1)

#Random noise
Rn_old = 0.01 * torch.eye(N)
Rn = Rn_old.unsqueeze(0).repeat(batch_size, 1, 1)

#distance bw radar and target
d_r_t = 70                                       
d_r_t_expanded = np.full((batch_size, N, 1), d_r_t)  # Shape (500, 64, 1)
d_r_t_expanded = torch.tensor(d_r_t_expanded)
#distance between radar and RIS center
d_r = 14.1421                                           

#angle between clutter, radar and RIS normal
theta_clutter = 35 
#theta_clutter_t = torch.full((batch_size, 1), theta_clutter)
theta_clutter_rad = torch.deg2rad(torch.tensor(theta_clutter))  # Use the current value from theta_t_r_RIS_np
theta_clutter_rad_t = torch.full((batch_size, 1), theta_clutter_rad)  
                                   
#distance bw radar and clutter
dr_ck=60
#dr_ck_expanded = torch.full((batch_size, N, 1), dr_ck)  # Shape (500, 64, 1)
dr_ck_expanded = np.full((batch_size, N, 1), dr_ck)  # Shape (500, 64, 1)
dr_ck_expanded = torch.tensor(dr_ck_expanded)


# Distances from RIS to target for each RIS element and radar to RIS elements
d_r_n = np.zeros(N)                                     
for n in range(N):
    d_r_n[n] = np.sqrt(d_r**2 + ((n - (N + 1) / 2) * d_RIS**2))
d_r_n_expanded = np.expand_dims(d_r_n, axis=0).reshape(1,N,1)  # Shape (1, 64, 1)
d_r_n_expanded = np.tile(d_r_n_expanded, (batch_size, 1, 1))  # Shape (500, 64, 1)
d_r_n_expanded = torch.tensor(d_r_n_expanded)

# Construct the shift matrix J0
J0 = torch.zeros((N, L))
for m in range(N):
    for n in range(L):
        if m - n == 0:
          J0[m, n] = 1
          
# Initialize an array to hold ris_vec values
ris_vec_list = []              
for n in range(N):
    ris_vec =(n - (N + 1) / 2) * d_RIS
    ris_vec_list.append(ris_vec)
ris_vec_array = np.array(ris_vec_list).reshape(N, 1)  # creating an array to transpose
#print("Shape of the ris_vec_array:",ris_vec_array.shape)
ris_vec_transpose = ris_vec_array.T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Decay learning_rate
def lr_scheduler(optimizer, epoch, init_lr, lr_decay_epoch):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
    return optimizer

# Define a simple Deep Neural Network (DNN) model
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(1, 2)
        self.bn1 = nn.BatchNorm1d(2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2, 4)
        self.bn2 = nn.BatchNorm1d(4)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4, 8)
        self.bn3 = nn.BatchNorm1d(8)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(8, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(16, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(32, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(64, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(128, 64)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        a = self.fc1(x)
        b = self.bn1(a)
        c = self.relu1(b)
        d = self.fc2(c)
        e = self.bn2(d)
        f = self.relu2(e)
        g = self.fc3(f)
        h = self.bn3(g)
        i = self.relu3(h)
        j = self.fc4(i)
        k = self.bn4(j)
        l = self.relu4(k)
        m = self.fc5(l)
        n = self.bn5(m)
        o = self.relu5(n)
        p = self.fc6(o)
        q = self.bn6(p)
        r = self.relu6(q)
        s = self.fc7(r)
        t = self.bn7(s)
        u = self.relu7(t)
        v = self.fc8(u)
        w = self.sig(v)
        output = w * math.pi
        return output
    
    import math

def calculate_angle(a, b, C):

    # Convert C from degrees to radians for math.cos
    C_rad = np.radians(C)
    
    # Using the law of cosines to calculate the third side c
    c = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(C_rad))
    
    # Using the law of cosines to calculate angle A
    A_rad = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
    
    # Convert A back to degrees
    A_deg1 = np.degrees(A_rad)    #Calculated angle A in degrees
    A_deg = A_deg1 - theta_r_RIS
    
    return A_deg 


def construct_Ht(theta_t_r_RIS,phi_vector):
   
    # Convert angles from degrees to radians
    theta_RIS_t = calculate_angle(d_r_t, d_r, theta_t_r_RIS)
    theta_RIS_t_rad = np.deg2rad(theta_RIS_t)    ## batch_size x 1 
    theta_t_r_RIS_rad = np.deg2rad(theta_t_r_RIS) # Use the current value from theta_t_r_RIS_np  ## batch_size x 1 

    # Distance between RIS and target
    d_t = np.sqrt(d_r_t ** 2 + d_r**2 - 2 * d_r_t * d_r * np.cos(theta_t_r_RIS_rad))   ## batch_size x 1     
    tensor_d_t = np.repeat(d_t, N, axis=1).reshape(batch_size, N, 1)      #batch_size x N X 1
    # print("Shape of the tensor_d_t:",tensor_d_t.shape)

    
    d_t_n = tensor_d_t + (np.sin(theta_RIS_t_rad)*ris_vec_transpose).reshape(batch_size,N,1)   #### batch_sizexNx1(500x64x1)
    #d_t_n = tensor_d_t + (np.sin(theta_RIS_t_rad)*ris_vec_transpose)

    
    #finding channel propogation coefficients and phases for LoS
    alpha = torch.sqrt((P * Gt**2 * lambda_**2) / ((4 * torch.pi)**3 * d_r_t_expanded**4))  
    phi = -4 * torch.pi * d_r_t_expanded / lambda_  
    
    #finding channel propogation coefficients and phases for nlos
    alpha_n = torch.sqrt((P * Gt * Gris * lambda_**2 * sigma_RIS_n) / 
                     ((4 * torch.pi)**4 * d_r_t_expanded**2 * d_t_n**2 * d_r_n_expanded**2))
    phi_ = d_r_t_expanded + d_r_n_expanded + d_t_n
    phi_n = -2 * torch.pi * phi_ / lambda_  
   
    # Range gate delay
    rn = np.ceil((d_r_n_expanded + d_t_n + d_r_t_expanded) * fs / c) - np.ceil((2 * d_r_t_expanded * fs) / c)
    
    # Create the shift matrix Jrn
    Jrn = torch.zeros((N, L))  # Initialize Jrn as a tensor of zeros

    for m in range(N):  
        for k in range(L):  
            if torch.any(torch.eq(torch.tensor(m - k), rn)): # Convert m - k to tensor and Check if the value matches
                Jrn[m, k] = 1
    
    phi_vector = phi_vector.reshape(batch_size, N, 1)   # Shape (500, 64, 1)
    
    # Construct the matrix Ht(phi)
    Ht = alpha * torch.exp(1j * phi) * J0
 
    Ht += alpha_n * phi_vector * torch.exp(1j * phi_n) * Jrn
    
    return Ht

# Construct Hck(phi)
def construct_Hck(phi_vector):

    # Convert angles from degrees to radians
    theta_RIS_c =calculate_angle(dr_ck,d_r,theta_clutter )
    theta_RIS_c_rad = np.deg2rad(theta_RIS_c)
    theta_RIS_c_rad_t = torch.full((batch_size, 1), theta_RIS_c_rad)

    # Calculate the distance from the RIS to the clutter using the given formula
    d_k = np.sqrt(dr_ck**2 - d_r**2 - 2 * dr_ck * d_r * np.cos(theta_clutter_rad_t))  
    tensor_d_k = np.repeat(d_k, N, axis=1).reshape(batch_size, N, 1) 
 
    
    # Distances from RIS to clutter for each RIS element
    d_ck_n = tensor_d_k + (np.sin(theta_RIS_c_rad_t)*ris_vec_transpose).reshape(batch_size,N,1) 

    # Range gate delay
    rkn = np.ceil((d_r_n_expanded + d_ck_n + dr_ck_expanded) * fs / c) - np.ceil((2 * dr_ck_expanded * fs) / c)
   
    # Compute the channel propagation coefficients for LoS
    alpha_k = torch.sqrt((P * G_c_k**2 * lambda_**2) / ((4 * torch.pi)**3 * dr_ck_expanded**4))

    # Calculate alpha_k_n for each RIS element  
    alpha_k_n= torch.sqrt((P * G_c_k * Gris * lambda_**2 * sigma_RIS_n) / 
                     ((4 * torch.pi)**4 * dr_ck_expanded**2 * d_ck_n**2 * d_r_n_expanded**2))

    # Phase terms
    phi_k = -4 * torch.pi * dr_ck_expanded / lambda_

    # Calculate phi_nk for each RIS element
    phi_ = dr_ck_expanded + d_r_n_expanded + d_ck_n
    phi_nk = -2 * torch.pi * phi_ / lambda_
   

    # Construct the shift matrix Jrkn
    
    Jrkn = torch.zeros((N, L))  # Initialize Jrn as a tensor of zeros

    for m in range(N):  # or another appropriate range for m
        for k in range(L):  # or another appropriate range for k
            if torch.any(torch.eq(torch.tensor(m - k), rkn)): # Convert m - k to tensor and Check if the value matches
                Jrkn[m, k] = 1

    phi_vector = phi_vector.reshape(batch_size, N, 1)   # Shape (500, 64, 1)
    # Construct the matrix Hck(phi)
    Hck = alpha_k * torch.exp(1j * phi_k) * J0
    Hck += alpha_k_n * phi_vector *  np.exp(1j * phi_nk) * Jrkn
   
    return Hck

def compute_w(Ht, Hck):


    Hck_s = torch.matmul(Hck, s_complex) 
    #print("Shape of the Hck_s:",Hck_s.shape)
    Hck_H = Hck_s.conj().transpose(-2, -1)
    #print("Shape of the Hck_H:",Hck_H.shape)

    # Step 4: Final multiplication
    zi = sigma_E * torch.matmul(Hck_s, Hck_H) + Rn  # Shape: [500, 64, 64]
    #print("Shape of the zi:",zi.shape)
    
    # Compute the batched inverse
    zi_inv = torch.linalg.inv(zi)  # Shape: [batch_size, N, N]
    #print("Shape of the zi_inv:",zi_inv.shape)

    # Perform matrix multiplications: inv(zi) @ Ht @ s
    Ht_s = torch.matmul(Ht, s_complex)  # Shape: [batch_size, N, 1]
    zi_inv_Ht_s = torch.matmul(zi_inv, Ht_s)  # Shape: [batch_size, N, 1]

    # Scale by a
    w = a * zi_inv_Ht_s  # Shape: [batch_size, N, 1]
   
    return w

def compute_sinr(w, Ht, Hck ):

    #print('Rn',Rn)
    '''
    print('s_complex',s_complex)
    print('Hck',Hck)
    print('Ht',Ht)
    print('w',w)
    print(f"Hck shape : {Hck.shape},s shape : {s_complex.shape},Rn shape : {Rn.shape},Ht shape :{Ht.shape},w shape : {w.shape}")
    '''
    # Numerator: Signal power
    w_t = w.conj().transpose(-2,-1)
    signal_power = sigma_E * torch.abs(torch.matmul(torch.matmul(w_t, Ht), s_complex))**2
    
    Hck_s = torch.matmul(Hck, s_complex) 
    Hck_H = Hck_s.conj().transpose(-2, -1)
    
    interference_matrix = sigma_E_k * torch.matmul(Hck_s, Hck_H) + Rn
    wt_I = torch.matmul(w_t, interference_matrix)
    denominator = torch.matmul(wt_I, w)  # Shape: [batch_size, 1, 1]
    denominator_new = abs(denominator)
    
    sinr= signal_power / denominator_new 
    
    # print(f" sinr shape : {sinr.shape}")
    return sinr


#sigma_squared_values = [1,0.316,0.1,0.0316,0.01]

sigma_squared_values = [0.01]          

def loss_function(sinr):
    epsilon = 1e-8  # Small constant to avoid log(0) or division by zero
    #loss = -torch.sum(torch.log2(1 + sinr))  # Maximize SINR (equivalent to minimizing negative log SINR)
    loss = -torch.sum(sinr)
    return loss / batch_size  # Normalize by batch sizeO
