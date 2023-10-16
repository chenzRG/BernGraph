import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch_geometric.nn import SAGEConv
from egsage import EGraphSage
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset, DataLoader, Batch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


import os
import csv
from itertools import combinations
from scipy import sparse
from tqdm import tqdm
from torch_geometric.transforms import RandomLinkSplit
from sklearn.model_selection import train_test_split
# For reproducibility
torch.manual_seed(0)
np.random.seed(0)
import warnings
warnings.filterwarnings("ignore")
# Check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('data.csv', index_col=0)
raw_data_matrix = data.values

label_data = pd.read_csv('label.csv', index_col=0)
label_data = label_data.values

# Split the data into train, validation, and test sets
train_ratio = 0.6
valid_ratio = 0.2
test_ratio = 0.2

# First, split the data into train and remaining sets
train_data, remaining_data, train_labels, remaining_labels = train_test_split(raw_data_matrix, label_data, test_size=1-train_ratio)

# Then, split the remaining data into validation and test sets
test_ratio_adjusted = test_ratio / (valid_ratio + test_ratio)  # Adjust the test ratio
valid_data, test_data, valid_labels, test_labels = train_test_split(remaining_data, remaining_labels, test_size=test_ratio_adjusted)

def generate_graph(raw_data_matrix):
    num_samples, num_nodes = raw_data_matrix.shape

    # Calculate proportions based on the entire dataset
    proportions = np.mean(raw_data_matrix, axis=0)

    # Generate posterior probabilities matrix based on the dataset
    denominator = proportions[:, None] + proportions[None, :]
    denominator[denominator == 0] = 1
    matrix_x = proportions[:, None] / denominator

    # Replace NaN values with 0
    matrix_x = np.nan_to_num(matrix_x, nan=0)

    # Generate edge list for fully connected graph
    edge_list = list(combinations(range(num_nodes), 2))

    return proportions, matrix_x, edge_list


def create_data_object(raw_data_batch, proportions, matrix_x, edge_list):
    # Create graph edges table
    graph_edges = np.array(edge_list)
    #node_labels = torch.tensor(raw_data_batch, dtype=torch.float) # convert to tensor here
    graph_edges_dict = {
        'src': graph_edges[:, 0].tolist(),
        'dst': graph_edges[:, 1].tolist(),
        'weight': np.round(matrix_x[graph_edges[:, 0], graph_edges[:, 1]], 3).tolist()
    }
    graph_edges_df = pd.DataFrame(graph_edges_dict)

    edge_index = torch.tensor(graph_edges_df[['src', 'dst']].values.T, dtype=torch.long)
    edge_attr = torch.tensor(graph_edges_df['weight'].tolist(), dtype=torch.float)

    data = Data(x=None, edge_index=edge_index, edge_attr=edge_attr, y=None)
    data = data.to(device)  # Move data object to device once
    
    return data

class GraphDataset(Dataset):
    def __init__(self, raw_data, labels, train_proportions, matrix_x, edge_list, graph_data):
        super(GraphDataset, self).__init__()
        self.raw_data = raw_data
        self.labels = labels
        self.proportions = torch.tensor(train_proportions, dtype=torch.float).to(device) # move to device here
        self.matrix_x = matrix_x
        self.edge_list = edge_list
        self.graph_data = graph_data


    def get(self, idx):
        raw_data = self.raw_data[idx]
        label = self.labels[idx]

        # Update the node embeddings and labels of the graph data object
        node_labels = torch.tensor(raw_data, dtype=torch.float).unsqueeze(0).to(self.proportions.device) # move to the same device as proportions
        node_labels = torch.where(node_labels == 1, self.proportions, 1 - self.proportions)
        self.graph_data.x = node_labels

        processed_label = torch.tensor(label, dtype=torch.float)
        self.graph_data.y = processed_label

        return self.graph_data

    def len(self):
        return len(self.labels)


# Define MLP model
class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiTaskMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
#             nn.Dropout(p=0.5), 
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Add Sigmoid activation function
        )

    def forward(self, x):
        return self.layers(x)


train_proportions, train_matrix_x, train_edge_list = generate_graph(train_data)
valid_proportions, valid_matrix_x, valid_edge_list = generate_graph(valid_data)
test_proportions, test_matrix_x, test_edge_list = generate_graph(test_data)


train_graph_data = create_data_object(None, train_proportions, train_matrix_x, train_edge_list)
valid_graph_data = create_data_object(None, valid_proportions, valid_matrix_x, valid_edge_list)
test_graph_data = create_data_object(None, test_proportions, test_matrix_x, test_edge_list)


train_dataset = GraphDataset(train_data, train_labels, train_proportions, train_matrix_x, train_edge_list, train_graph_data)
valid_dataset = GraphDataset(valid_data, valid_labels, valid_proportions, valid_matrix_x, valid_edge_list, valid_graph_data)
test_dataset  = GraphDataset(test_data,  test_labels,  test_proportions, test_matrix_x, test_edge_list, test_graph_data)
# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define GNN model
gnn_input_dim = 1
gnn_output_dim = 1

gnn_model = EGraphSage(gnn_input_dim, gnn_output_dim, edge_channels=1, activation='relu', edge_mode=1, normalize_emb=True, aggr='mean').to(device)
#(node_in_dim,node_out_dim,edge_dim,activation,edge_mode,normalize_emb, aggr)

# Define Binary MLP models for each task
mlp_input_dim = raw_data_matrix.shape[1]
label_dim = label_data.shape[1]
mlp_hidden_dim = 64
mlp_model = MultiTaskMLP(mlp_input_dim, mlp_hidden_dim, label_dim).to(device)

optimizer = optim.Adam(
    list(gnn_model.parameters()) +
    list(mlp_model.parameters()),
    lr=0.00001
)

# ... Other imports and definitions ...
num_epochs = 200
best_model_params = None
patience = 10  # Number of epochs to wait for improvement before stopping
epochs_without_improvement = 0

# Initialize the best validation loss to a large number
best_val_loss = float('inf')
model_dir = './models'  # Directory to save model parameters
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
# Define loss functions for each task
criterion = nn.BCELoss()
train_loss = []
val_loss =[]

for epoch in tqdm(range(num_epochs), desc='Epochs', unit='epoch'):
    # Training phase
    gnn_model.train()
    total_loss = 0
    with tqdm(total=len(train_loader), desc='Train', unit='batch') as pbar:
        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()

            x_train, edge_index_train,edge_attr_train, y_train = data.x, data.edge_index, data.edge_attr, data.y
            x_train = x_train.squeeze().unsqueeze(-1)
            
            gnn_output_train = gnn_model(x_train, edge_attr_train, edge_index_train)
            mlp_output = mlp_model(gnn_output_train.view(-1))
            
            flat_predictions = mlp_output.view(-1)
            flat_targets = y_train.view(-1)
            loss = criterion(flat_predictions, flat_targets)*label_dim
            
#             regular_lamda = 0.05
#             regular_regularization = torch.tensor(0.)
#             regular_regularization = regular_regularization.to("cuda")
#             for param in mlp_model.parameters():
#                 regular_regularization += torch.norm(param,p = 2)
            
#             loss = loss_1 + loss_2 + loss_3 + loss_4 + regular_lamda * regular_regularization
#             loss = loss_1 + loss_2 + loss_3 + loss_4 
            #print("Loss=",loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)
    
    pbar.close()

    # Validation phase
    gnn_model.eval()
    total_val_loss = 0
    best_val_loss = float('inf')
    with torch.no_grad():
        with tqdm(total=len(valid_loader), desc='Validation', unit='batch') as pbar:
            for data in valid_loader:
                data.to(device)

                x_valid, edge_index_valid, edge_attr_valid, y_valid = data.x, data.edge_index, data.edge_attr, data.y
                x_valid = x_valid.squeeze().unsqueeze(-1)
                gnn_output_valid = gnn_model(x_valid, edge_attr_valid, edge_index_valid)
                mlp_output = mlp_model(gnn_output_valid.view(-1))
                
                flat_predictions = mlp_output.view(-1)
                flat_targets = y_valid.view(-1)
                loss = criterion(flat_predictions, flat_targets)*label_dim
                
                total_val_loss += loss.item()
            
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
      
        pbar.close()
        

    # Print statistics
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}, Validation Loss: {total_val_loss/len(valid_loader)}')
    train_loss.append(total_loss/len(train_loader))
    val_loss.append(total_val_loss/len(valid_loader))
    
    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        torch.save(gnn_model.state_dict(), os.path.join(model_dir, 'best_gnn_model.pth'))
        torch.save(mlp_model.state_dict(), os.path.join(model_dir, 'best_mlp_model.pth'))

with open('val_loss.txt', 'w') as file1:
    for item in val_loss:
        file1.write(str(item) + '\n')
with open('train_loss.txt', 'w') as file2:
    for item in train_loss:
        file2.write(str(item) + '\n')

