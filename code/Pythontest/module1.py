import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
import json
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)
        self.wjson("Gates",{"weight":self.Gates.weight.tolist(),"bias": self.Gates.bias.tolist()})
    def wjson(self,file,new_dict):
        with open(file+".json","w") as f:
             json.dump(new_dict,f)   
    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        return hidden, cell
if __name__ == '__main__':
    b, c, h, w = 1,1,5,5
    d = 1           # hidden state size
    lr = 1e-1       # learning rate
    T = 1           # sequence length
    max_epoch = 20  # number of epochs

    # set manual seed
    torch.manual_seed(0)

    print('Instantiate model')
    model = ConvLSTMCell(c, d)
    print(repr(model))

    print('Create input and target Variables')
    #x = Variable(torch.rand(T, b, c, h, w))
    #y = Variable(torch.randn(T, b, d, h, w))
    x = Variable(torch.ones(T, b, c, h, w))
    y = Variable(torch.ones(T, b, d, h, w))
    
    print('Create a MSE criterion')
    loss_fn = nn.MSELoss()

    print('Run for', max_epoch, 'iterations')
    for epoch in range(0, max_epoch):
        state = None
        loss = 0
        for t in range(0, T):
            state = model(x[t], state)
            loss += loss_fn(state[0], y[t])

        print(' > Epoch {:2d} loss: {:.3f}'.format((epoch+1), loss.data))

        # zero grad parameters
        model.zero_grad()

        # compute new grad parameters through time!
        loss.backward()

        # learning_rate step against the gradient
        for p in model.parameters():
            print(p.grad.data)
            #p.data.sub_(p.grad.data * lr)

    print('Input size:', list(x.data.size()))
    print('Target size:', list(y.data.size()))
    print('Last hidden state size:', list(state[0].size()))