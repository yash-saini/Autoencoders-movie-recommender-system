import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import pandas as pd

"""Data Preprocessing"""

movies=pd.read_csv("movies.dat",sep="::",header=None, engine="python", encoding="latin-1")
users=pd.read_csv("users.dat",sep="::",header=None, engine="python", encoding="latin-1")
ratings=pd.read_csv("ratings.dat",sep="::",header=None, engine="python", encoding="latin-1")
 
#Training and test set
training_set=pd.read_csv("u1.base",delimiter="\t")
training_set=np.array(training_set, dtype='int')
test_set=pd.read_csv("u1.test",delimiter="\t")
test_set=np.array(test_set, dtype='int')

#number of users and movies
nb_users=int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies=int(max(max(training_set[:,1]),max(test_set[:,1])))

#movies in columns, each user in row and rating to the corresponding movie in the table
def convert(data):
    new_dat=[]
    for i in range(1,nb_users+1):
        movies_id=data[:,1][data[:,0]==i]
        ratings_id=data[:,2][data[:,0]==i]
        ratings_list=np.zeros(nb_movies)
        ratings_list[movies_id-1]=ratings_id
        new_dat.append(list(ratings_list))
    return new_dat
training_set=convert(training_set)
test_set=convert(test_set)

# to torch tensors
training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_set)

""" Auto encoders"""

class SAE(nn.Module):
    def __init__(self):
        super(SAE,self).__init__()
        self.fc1=nn.Linear(nb_movies,20)#full connection layer(input,output)
        self.fc2=nn.Linear(20,10)
        self.fc3=nn.Linear(10,20)
        self.f4=nn.Linear(20,nb_movies)
        self.activation=nn.Sigmoid()
    def forward(self,x):
        x=self.activation(self.fc1(x))
        x=self.activation(self.fc2(x))
        x=self.activation(self.fc3(x))
        x=self.f4(x)
        return x
sae=SAE()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(sae.parameters(),lr=0.01,weight_decay=0.5)


#Training
nb_epoch=200
for epoch in range(1,nb_epoch+1):
    train_loss=0
    #users rating atleast one movie
    s=0.
    for id_user in range(nb_users):
        #addition of dimension
        inp=Variable(training_set[id_user]).unsqueeze(0)
        target=inp.clone()
        if torch.sum(target.data > 0 ) > 0:
            output=sae(inp)
            target.require_grad=False
            output[target==0]=0
            loss=criterion(output,target)
            mean_corrector=nb_movies/float(torch.sum(target.data > 0) +1e-10)
            loss.backward()
            train_loss +=np.sqrt( loss.data * mean_corrector)
            s+=1.
            optimizer.step()
    print ("epoch: "+str(epoch)+" loss: "+str(train_loss/s))
            
    
    
#Testing


test_loss=0
#users rating atleast one movie
s=0.
for id_user in range(nb_users):
    #addition of dimension
    inp=Variable(test_set[id_user]).unsqueeze(0)
    target=inp.clone()
    if torch.sum(target.data > 0 ) > 0:
        output=sae(inp)
        target.require_grad=False
        output[target==0]=0
        loss=criterion(output,target)
        mean_corrector=nb_movies/float(torch.sum(target.data > 0) +1e-10)
        loss.backward()
        test_loss +=np.sqrt( loss.data * mean_corrector)
        s+=1.
        optimizer.step()
print (" test loss: "+str(train_loss/s))
        





        
        
        
