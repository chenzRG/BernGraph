import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch_geometric.nn import SAGEConv
from egsage import EGraphSage
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset, DataLoader, Batch
from sklearn.metrics import roc_auc_score


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

label_data = pd.read_csv('label.csv')
label_data = label_data.values[:,1:]

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

    matrix_x = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            logic = np.logical_and(raw_data_matrix.T[i] == 1, raw_data_matrix.T[j] == 1)
            if np.any(logic):
                matrix_x[i, j] = 1

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
    
proportions, matrix_x, edge_list = generate_graph(raw_data_matrix)
# np.save('proportions.npy',proportions)
# np.save('matrix_x.npy',matrix_x)
#proportions = np.load('proportions.npy')
#matrix_x = np.load('matrix_x_complicated.npy')
edge_list = list(combinations(range(raw_data_matrix.shape[1]), 2))
graph_data = create_data_object(None, proportions, matrix_x, edge_list)

# Define the datasets
train_dataset = GraphDataset(train_data, train_labels, proportions, matrix_x, edge_list, graph_data)
valid_dataset = GraphDataset(valid_data, valid_labels, proportions, matrix_x, edge_list, graph_data)
test_dataset  = GraphDataset(test_data,  test_labels,  proportions, matrix_x, edge_list, graph_data)

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
mlp_input_dim = 692
mlp_hidden_dim = 64
mlp_model = MultiTaskMLP(mlp_input_dim, mlp_hidden_dim, 4).to(device)

# Define loss functions for each task
criterion_1 = nn.BCELoss()
criterion_2 = nn.BCELoss()
criterion_3 = nn.BCELoss()
criterion_4 = nn.BCELoss()

# Define optimizer for all models
optimizer = optim.Adam(
    list(gnn_model.parameters()) +
    list(mlp_model.parameters()),
    lr=0.00001
)

# ... Other imports and definitions ...
num_epochs = 100
best_model_params = None
patience = 10  # Number of epochs to wait for improvement before stopping
epochs_without_improvement = 0

# Initialize the best validation loss to a large number
best_val_loss = float('inf')
model_dir = './models'  # Directory to save model parameters
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

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
            

            mlp_output_1, mlp_output_2, mlp_output_3, mlp_output_4 = torch.split(mlp_output, 1, dim=-1)
            #print("mlp_output_1=",mlp_output_1.shape)
            y_train = y_train.unsqueeze(-1)
            loss_1 = criterion_1(mlp_output_1, y_train[0])
            loss_2 = criterion_2(mlp_output_2, y_train[1])
            loss_3 = criterion_3(mlp_output_3, y_train[2])
            loss_4 = criterion_4(mlp_output_4, y_train[3])

            
#             regular_lamda = 0.05
#             regular_regularization = torch.tensor(0.)
#             regular_regularization = regular_regularization.to("cuda")
#             for param in mlp_model.parameters():
#                 regular_regularization += torch.norm(param,p = 2)
            
#             loss = loss_1 + loss_2 + loss_3 + loss_4 + regular_lamda * regular_regularization
            loss = loss_1 + loss_2 + loss_3 + loss_4 
            #print("Loss=",loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # 更新进度条的状态
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)
    # 重置进度条
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

                mlp_output_1, mlp_output_2, mlp_output_3, mlp_output_4 = torch.split(mlp_output, 1, dim=-1)
                y_valid = y_valid.unsqueeze(-1)
                loss_1 = criterion_1(mlp_output_1, y_valid[0])
                loss_2 = criterion_2(mlp_output_2, y_valid[1])
                loss_3 = criterion_3(mlp_output_3, y_valid[2])
                loss_4 = criterion_4(mlp_output_4, y_valid[3])

                loss = loss_1 + loss_2 + loss_3 + loss_4
                total_val_loss += loss.item()
                # 更新进度条的状态
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
        # 重置进度条
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

# Load the best model parameters and evaluate on the test set
gnn_model.load_state_dict(torch.load(os.path.join(model_dir, 'best_gnn_model.pth')))
mlp_model.load_state_dict(torch.load(os.path.join(model_dir, 'best_mlp_model.pth')))

gnn_model.eval()
mlp_model.eval()

list_output = []
list_y = []

with torch.no_grad():
    with tqdm(total=len(test_loader), desc='Test', unit='batch') as pbar:
        for data in test_loader:
            data.to(device)

            x_test, edge_index_test, edge_attr_test, y_test = data.x, data.edge_index, data.edge_attr, data.y
            x_test = x_test.squeeze().unsqueeze(-1)
            gnn_output_test = gnn_model(x_test, edge_attr_test, edge_index_test)
            mlp_output = mlp_model(gnn_output_test.view(-1))
            
            list_output.append(mlp_output.cpu().detach())
            list_y.append(y_test.cpu().detach())

            pbar.update(1)
    # 重置进度条
    pbar.close()
    
    
true_labels_np = np.array([tensor.numpy() for tensor in list_y])
model_outputs_np = np.array([tensor.numpy() for tensor in list_output])
pred_labels = np.round(model_outputs_np) 

true_labels_np = true_labels_np.T
model_outputs_np = model_outputs_np.T
pred_labels = pred_labels.T

from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, auc

def metrics(true_labels,predicted_labels):
    TP = 0
    FP = 0
    FN = 0

    # 计算 TP、FP 和 FN
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == 1 and predicted_label == 1:
            TP += 1
        elif true_label == 0 and predicted_label == 1:
            FP += 1
        elif true_label == 1 and predicted_label == 0:
            FN += 1

    # 计算 Precision 和 Recall
    precision = TP / (TP + FP+1e-9)
    recall = TP / (TP + FN +1e-9)

    return precision, recall
    
all_f1 = []
total_prc = []
total_auc = []
total_recall = []
total_precision = []
drug_count = []

for index in range(pred_labels.shape[0]):
    drug_count.append(np.sum(pred_labels[index]==1))
for i in range(pred_labels.shape[0]):
#     if np.all(true_labels_np.T[i] !=0):
    precision,recall = metrics(true_labels_np[i],pred_labels[i])
    total_recall.append(recall)
    total_precision.append(precision)
    prc = average_precision_score(true_labels_np[i],model_outputs_np[i], average='macro')
    total_prc.append(prc)
    f1 = f1_score(true_labels_np[i], pred_labels[i], average='macro')
    all_f1.append(f1)
    auc = roc_auc_score(true_labels_np[i],model_outputs_np[i])
    total_auc.append(auc)
    

def jaccard_sim(a, b):
    unions = len(set(a).union(set(b)))
    intersections = len(set(a).intersection(set(b)))
    return intersections / unions

drug_count = []
total_jaccard=[]
for index in range(pred_labels.T.shape[0]):
    drug_count.append(np.sum(pred_labels.T[index]==1))
for i in range(pred_labels.T.shape[0]):
    if np.all(true_labels_np.T[i] !=0):
    #     if np.all(true_labels_np[i] != 0):
#         if np.all(true_labels_np[i] != 1):
#         total_auc.append(0)
#         continue
        total_jaccard.append(jaccard_sim(np.where(pred_labels.T[i] == 1)[0], np.where(true_labels_np.T[i] == 1)[0]))

print("total drug counts:",  np.mean(drug_count))
print("total jaccard:",  np.mean(total_jaccard))
print("total PRAUC:",  np.mean(total_prc))
print("total AUROC:",  np.mean(total_auc))
print("total F1:",  np.mean(all_f1))
print("total recall:",  np.mean(total_recall))
print("total precision:",  np.mean(total_precision))
