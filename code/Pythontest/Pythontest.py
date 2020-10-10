import torch
import torch.nn as nn
from torch.autograd import Variable



if __name__ == '__main__':
   #np.random.seed(args.seed)
   torch.manual_seed(0)
   input=torch.ones(1,1,3,3,requires_grad=True)
   input2=torch.ones(1,1,5,5)
   input3=torch.randn(1,1,2,2,requires_grad=True)
   outputa=torch.ones(1,1,2,2)
   
   m=nn.ConvTranspose2d(1,1,3,2,1)
   m2=nn.Conv2d(1,1,2,1,0)
    
   #output=m2(input)
   output=m(input)
  
  
   #output.requires_grad=true
   print(output)
   #output=output

    
   for p in m.parameters():
        print(p.data)
        
   for p in m2.parameters():
        print(p.data)
  
  
   loss=nn.MSELoss()
   loss=loss(input2,output)
   #print( loss)
   loss.backward()
   #output.backward()
   #input4.backward()
   
   #print(m.grad)
   #print(m2.weight.grad)
   #print(m2.bias.grad)

   print(m.weight.grad)
   print(m.bias.grad)
   print(input.grad)
   print(input3.grad)