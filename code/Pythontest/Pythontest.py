import torch
import torch.nn as nn
from torch.autograd import Variable
import json
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        #assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.wjson("wxi",{"weight":self.Wxi.weight.tolist(),"bias": self.Wxi.bias.tolist()})
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.wjson("Whi",{"weight":self.Wxi.weight.tolist(),"bias": self.Wxi.bias.tolist()})
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.wjson("Wxf",{"weight":self.Wxi.weight.tolist(),"bias": self.Wxi.bias.tolist()})
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.wjson("Whf",{"weight":self.Wxi.weight.tolist(),"bias": self.Wxi.bias.tolist()})
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.wjson("Wxc",{"weight":self.Wxi.weight.tolist(),"bias": self.Wxi.bias.tolist()})
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.wjson("Wxc",{"weight":self.Wxi.weight.tolist(),"bias": self.Wxi.bias.tolist()})
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.wjson("Wxo",{"weight":self.Wxi.weight.tolist(),"bias": self.Wxi.bias.tolist()})
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.wjson("Who",{"weight":self.Wxi.weight.tolist(),"bias": self.Wxi.bias.tolist()})

        self.Wci = None
        self.Wcf = None
        self.Wco = None

   
    def wjson(self,file,new_dict):
        with open(file+".json","w") as f:
             json.dump(new_dict,f)   
    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1]))
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))



if __name__ == '__main__':
   #np.random.seed(args.seed)
   torch.manual_seed(0)
    
   input2=torch.ones(1,1,5,5) 
   outputa=torch.ones(1,1,5,5)
   
   #m=nn.ConvTranspose2d(1,1,3,2,1)
   m2=nn.Conv2d(1,1,2,1,0)
    
   output=m2(input)
   #output=m(input)
  
  
   #output.requires_grad=true
   #print(output)
   #output=output

    
   #for p in m.parameters():
   #     print(p.data)
        
   #for p in m2.parameters():
   #     print(p.data)
  
  
   
   #output.backward()
   #input4.backward()
   
   #print(m.grad)
   #print(m2.weight.grad)
   #print(m2.bias.grad)

   print(m.weight.grad)
   print(m.bias.grad)
   print(input.grad)
   print(input3.grad)

