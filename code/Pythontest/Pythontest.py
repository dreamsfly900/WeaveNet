import torch
import torch.nn as nn
from torch.autograd import Variable



if __name__ == '__main__':
   #np.random.seed(args.seed)
   torch.manual_seed(0)
   input=torch.ones(1,1,2,2)
   input2=torch.ones(1,1,3,3)
   outputa=torch.ones(1,1,2,2)
   m=nn.ConvTranspose2d(1,1,3,2,1)
   #m=nn.Conv2d(1,1,3,2,1)
   output=m(input)
   print(output)
   #output=output

    
   for p in m.parameters():
        print(p.data)
        
        
   #loss=nn.MSELoss()
   #loss=loss(input2,output)
   #print( loss)
   #loss.backward()
   #output.backward()

   
   #print(m.grad)
   print(m.weight.grad)
   #print( m.grad)