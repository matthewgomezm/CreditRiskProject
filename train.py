# importing libraries for data preprocessing and model training
import pandas as pd 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# defining the model
class Model(nn.Module):
    def __init__(self, inputs=9, h1=8, h2=9, outputs=1):
        super().__init__()  # init the nn module
        self.fc1 = nn.Linear(inputs, h1)  # first fully connected layer
        self.fc2 = nn.Linear(h1, h2)  # second fully connected layer
        self.out = nn.Linear(h2, outputs)  # output layer

    # the forward pass of the model
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

df = pd.read_csv('data/dataset.csv') # reading the dataset 
torch.manual_seed(32)   # manual seed for reproducible results
model = Model()  # instantiate ml model


#####  data preprocessing  #####

# turns text column for employment status to numbers
df = pd.get_dummies(df, columns=['employment_status'], dtype=float)

X = df.drop(columns=['default'])
y = df['default']

X = X.values.astype('float32')
y = y.values.astype('float32')

# splitting data into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=32, stratify=y)

# scaling features so large columns don't drown out small ones
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# converting to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.0]))       # loss function
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)    # optimizer

#### training ####
epochs = 100
losses = []
for i in range(epochs):
    y_prediction = model(X_train)
    loss = criterion(y_prediction, y_train)
    losses.append(loss.item())   # plain number, handy for plotting later

    if i%10==0:
        print(f'Epoch: {i} Loss: {loss.item():.4f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# eval model on test data set 
with torch.no_grad():   # turn off back propagation
    y_eval = model(X_test)        # X_test are features from test set, y_eval will be predictions
    loss = criterion(y_eval, y_test)      # find loss off y eval vs y test

print(f'Test Loss: {loss.item():.4f}')

preds = (torch.sigmoid(y_eval) > 0.5).float()   # logits -> 0/1 predictions
tp = ((preds == 1) & (y_test == 1)).sum().item()
fp = ((preds == 1) & (y_test == 0)).sum().item()
fn = ((preds == 0) & (y_test == 1)).sum().item()
print(f'Recall: {tp/(tp+fn):.2f}  Precision: {tp/(tp+fp):.2f}')

