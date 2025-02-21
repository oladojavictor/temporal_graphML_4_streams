import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, temporal_signal_split
from tqdm import tqdm
from torch_geometric_temporal.nn.recurrent import A3TGCN2


# GPU support
DEVICE = torch.device('cuda') # cuda
shuffle=True
batch_size = 2


def normalization(array):
    '''
    Parameters
    ----------
    array: np.ndarray (B,N,F,T)
    Returns
    ----------
    array_norm : np.ndarray,
                shape is the same as original
    '''
    mean = array.mean(axis=(0,1,3), keepdims=True)
    std = array.std(axis=(0,1,3), keepdims=True)

    def normalize(x):
        return (x - mean) / std

    array_norm = normalize(array)

    return array_norm


edge_list = np.load("surface_edge_list.npy")
subsurface_adj = np.load("surface_adj.npy")
stacked_features = np.load("stacked_features.npy")
stacked_targets = np.load("stacked_targets.npy")[:,2:,:]

number_target_features = stacked_targets.shape[1]

sequence_length = 365
indices = [(i, i + sequence_length)
           for i in range(0,stacked_features.shape[2], 365)
        ]
        
# Generate observations
features, targets = [], []
for i, j in indices:
    features.append(stacked_features[:,:,i:j])
    targets.append(stacked_targets[:,:,i:j])

# edges

edges = []
for e in edge_list:
    edges.append(tuple(e))
edges = np.array(edges).T
    
    
edge_weights = np.ones(edges.shape[1])

dataset = StaticGraphTemporalSignal(
            edges, edge_weights, features, targets
        )
        
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.70)

# create validation dataset
random.seed(42)
val_indices = random.sample([i for i in range(len(test_dataset.features))], int(len(test_dataset.features)/2))
test_indices = [i for i in range(len(test_dataset.features)) if i not in val_indices]


val_input, val_target, test_input, test_target  = [],[],[],[]

for i in val_indices:
    val_input.append(test_dataset.features[i])
    val_target.append(test_dataset.targets[i])
for i in test_indices:
    test_input.append(test_dataset.features[i])
    test_target.append(test_dataset.targets[i])
    
train_input = np.array(train_dataset.features) # (4, 740, 58, 365)
train_target_norm  = np.array(train_dataset.targets) # (4, 740,18, 365)
val_input = np.array(val_input)
val_target = np.array(val_target)
test_input = np.array(test_input)
test_target = np.array(test_target)

train_input_norm = normalization(train_input)
val_input_norm = normalization(val_input)

train_input = 0
val_input = 0
train_input_norm = np.nan_to_num(train_input_norm, nan=0)
val_input_norm = np.nan_to_num(val_input_norm, nan=0)
#train_target_norm = np.nan_to_num(train_target_norm, nan=0)


train_x_tensor = torch.from_numpy(train_input_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=True,drop_last=True)

val_x_tensor = torch.from_numpy(val_input_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
val_dataset_new = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)
val_loader = torch.utils.data.DataLoader(val_dataset_new, batch_size=batch_size, shuffle=True,drop_last=True)


# model construction
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features,  out_channels=32, periods=periods,batch_size=batch_size) # node_features=58, periods=365
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods*number_target_features)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index) # x [b, 740, 58, 365]  returns h [b, 740, 18*365]
        h = F.relu(h) 
        h = self.linear(h)
        h = h.reshape(batch_size,x.shape[1], number_target_features, x.shape[3])
        return h

TemporalGNN(node_features=40, periods=365, batch_size=batch_size)

# Create model and optimizers
model = TemporalGNN(node_features=40, periods=365, batch_size=batch_size).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, amsgrad=True)
loss_fn = torch.nn.MSELoss()

for snapshot in train_dataset:
    static_edge_index = snapshot.edge_index.to(DEVICE)
    break;


num_epoch = 300

epoch_losses = []
val_losses = []
errors = []
ground_truth_pred = []


for epoch in range(num_epoch):

    model.train() #set model to training mode
    #step = 0
    cost = 0 #errors
    mini_batch = 0 #errors
    loss_list = []
    for encoder_inputs, labels in train_loader:
        mini_batch += 1
        y_hat = model(encoder_inputs, static_edge_index)         # Get model predictions
        loss = loss_fn(y_hat, labels) # Mean squared error #loss = torch.mean((y_hat-labels)**2)  sqrt to change it to rmse
        cost = (y_hat - labels) #errors
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #step= step+ 1
        loss_list.append(loss.item())
        if epoch % 100 == 0 and mini_batch % 25 == 0:
            ground_truth_pred.append((labels.cpu().detach(), y_hat.cpu().detach(), cost.cpu().detach()))
        #if step % 100 == 0 :
            #print(sum(loss_list)/len(loss_list))
    avg_train_loss = sum(loss_list) / len(loss_list)
    epoch_losses.append(avg_train_loss)
    
    #validation
    model.eval() # set model to evaluation mode
    val_loss_list = []
    with torch.no_grad(): # disable gradient computation for validation
        for encoder_inputs, labels in val_loader:  
            y_hat = model(encoder_inputs, static_edge_index)  # Get model predictions
            val_loss = loss_fn(y_hat, labels)  # Compute validation loss
            val_loss_list.append(val_loss.item())
    avg_val_loss = sum(val_loss_list) / len(val_loss_list)
    val_losses.append(avg_val_loss)

    # save the best model based on validation loss
    if avg_val_loss == min(val_losses):
        torch.save(model.state_dict(), "s_best_model.pth")
            
    print("Epoch {} train MSE: {:.4f}".format(epoch, avg_train_loss))
    print("Epoch {} validation MSE: {:.4f}".format(epoch, avg_val_loss))
    print()
    
np.save("training_losses.npy", epoch_losses)
np.save("validation_losses.npy", val_losses)


temporal_attention = model.state_dict()['tgnn._attention'].cpu().detach()
np.save("temporal_attention_val.npy", temporal_attention)