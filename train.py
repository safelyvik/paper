##train_target_detection
#### DNN Nov 24_2024
from __future__ import print_function
import time
from util_target_detection_24Dec import*
import os
import matplotlib.pyplot as plt
import torch.distributions as dist


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

channel = scipy.io.loadmat('C:/Users/Arpita-1/Desktop/Divya/theta.mat')
theta = torch.tensor(channel['random_numbers'], dtype=torch.float32)

#train_loader = torch.utils.data.DataLoader((csi_h[half_size:,:]), batch_size=batch_size, shuffle=True, num_workers=0)
train_loader = torch.utils.data.DataLoader(theta, batch_size=batch_size, shuffle=True, num_workers=0)   


# Function to calculate Q(x), the CCDF of the normal distribution
def Q(x): return 0.5 * torch.erfc(x / torch.sqrt(torch.tensor(2.0)))
def Q_inv(p_fa): return dist.Normal(0, 1).icdf(p_fa)
    
# Training Loop
for sigma_squared in sigma_squared_values:
    plt.figure(figsize=(10, 6))
    train_losses = []  # Reset losses for each sigma_squared    
    sinr_values = []  # To store average SINR per epoch
    pd_values = []  # To store average PD per epoch  
    dnn = DNN().to(device)
    optimizer = optim.Adam(dnn.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        running_sinr = 0.0  # To accumulate SINR across batches
        running_pd = 0.0  # To accumulate PD across batches
        total_samples = 0   # To count the number of samples for averaging SINR and PD
        
        for i,data in enumerate(train_loader): 
            data = data.to(device)
            dnn.zero_grad()
            optimizer.zero_grad()
            # Forward pass
            output = dnn(data)
            Ht = construct_Ht(data,output)
            Hc = construct_Hck(output)
            W = compute_w(Hc,Ht)
            sinr= compute_sinr(W,Ht,Hc)
                              
            # Calculate loss
            loss = loss_function(sinr)
            running_loss += loss.item()
                        
             # Accumulate SINR 
            running_sinr += sinr.sum().item()  # Sum SINR across the batch
            total_samples += sinr.numel()  # Total number of samples for averaging
            
            # Backpropagation
            loss.backward()
            optimizer.step()
        
        # Compute PD based on SINR using Marcum-Q
        pfa = torch.tensor(1e-4, dtype=torch.float32)
        # Calculate Q^-1(P_fa)
        q_inv = Q_inv(pfa)
        # Calculate the term under the square root
        sqrt_term = np.sqrt(2 * running_sinr)
        # Calculate P_d using the formula
        pd = Q(q_inv - sqrt_term)
            
        #print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'%(epoch+1, num_epochs, i+1, len(train_loader),running_loss))
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.5f},  SINR: {10 * np.log10(running_sinr):.2f} dB,  PD: {running_pd:.4f}')
        
        # Calculate the average loss for the epoch and store it
        avg_loss = running_loss / len(train_loader)
        avg_sinr = running_sinr / len(train_loader)  # Average SINR for the epoch
        #avg_pd = pd / num_epochs  # Average PD for the epoch
       
        train_losses.append(avg_loss)
        sinr_values.append(avg_sinr)  # Store average SINR
        pd_values.append(pd)  # Store average PD
        
        # Update learning rate
        optimizer = lr_scheduler(optimizer, epoch, learning_rate, lr_decay_epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.5f}, Avg SINR: {10 * np.log10(avg_sinr):.2f} dB,  PD: {pd:.4f}') 
        plt.plot(train_losses, label=f'sigma_squared = {sigma_squared}', linestyle='-', color='black')
        
    
    
    
    # Save the model state to a checkpoint file after the last epoch
    checkpoint_dir = 'checkpoint_dnn_2024'
    # Create the directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir,f'checkpoint_{sigma_squared}.pth')
    # Save the model state to the checkpoint file
    with open(checkpoint_path, 'wb') as f:
        torch.save(dnn.state_dict(), f)

    # Add labels, title, and legend
    
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Epoch for Sigma Squared Value {}'.format(sigma_squared))
    plt.grid(True)
    plt.savefig(f'training_plot_{sigma_squared}.png') # Save the plot to a PNG file
    plt.clf() # Clear the plot for the next sigma_squared value
   
   # Plot SINR vs PD
    
    pd_values_tensor = torch.tensor(pd_values)
    plt.plot(10 * np.log10(sinr_values),pd_values_tensor, label='Average SINR (dB)', linestyle='--', color='blue')
    plt.xlabel('Average SINR (dB)')
    plt.ylabel('Average Detection Probability (PD)')
    plt.title(f'Average SINR per Average Detection Probability (PD) for Sigma Squared Value {sigma_squared}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'sinr_pd_plot_{sigma_squared}.png')
    plt.clf()  # Clear the plot for the next sigma_squared value
       
    '''            

        def Q(x):
        return 0.5 * torch.erfc(x / torch.sqrt(torch.tensor(2.0)))

            # Function to calculate the inverse of Q(x), i.e., Q^-1(P_fa)
            def Q_inv(p_fa):
            # The inverse of Q corresponds to the inverse of the normal CDF
            normal_dist = dist.Normal(0, 1)
            return normal_dist.icdf(p_fa)
            # Define a Marcum-Q function to compute PD based on SINR
                def marcum_q(m, a, b):
                    return torch.special.i0e(a * b)  # Approximation for the Marcum-Q function
             # a = torch.sqrt(2 * sinr)  # Ensure SINR is treated as a tensor
             # b = torch.sqrt(-2 * torch.log(pfa))
             # pd = marcum_q(1, a, b)  # Detection Probability for this batch

    '''
