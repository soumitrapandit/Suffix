import torch


class Sequential:

    def __init__(self,layers):
        # So we're basically going to initialize a sequential model
        # And every layer would be a funtion
        # When we call the model, it will take in x
        # And it should just return whatever the last function in the sequence returns
        # This is the magic of classes in action

        self.layers = layers
        self.parameters = []

    def __call__(self,x):
        #
        for layer in self.layers:
            x = layer(x)
            
        
        self.out = x
        return self.out
    
    def parameters(self):
        #
        for layer in self.layers:
            self.parameters += layer.parameters()
        
        return self.parameters
        

class Linear:
    
    def __init__(self, dim_in, dim_out, bias = True):
        self.bias = bias
        self.weight = torch.randn((dim_in,dim_out)) * 1/dim_in **2 # This is the kaiming normalization
        self.bias = torch.randn(dim_out) if bias else None

    def __call__(self,x):
        self.out =  (x@ self.weight + (self.bias if self.bias else 0))
        return self.out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias else [])


class BatchNorm1d:

    def __init__(self, dim, eps = 1e-5, momentum = 0.1):
        self.eps = eps # Don't know what this is
        self.momentum = momentum # Do know what it is
        
        # Squeeze Factors (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.ones(dim)

        # Training Time Tools
        self.training = True 

        # Accumulators:s
        self.global_mean = torch.zeros(dim)
        self.global_var = torch.ones(dim) 

    def __call__(self,x):
        if self.training:
            xmean = x.mean(dim=0,keepdim = True)
            xvar = x.var(dim=0,keepdim = True)
        else:
            xmean = self.global_mean
            xvar = self.global_var

        # So we "whiten our batch"
        x_hat = (x - xmean)/ torch.sqrt(xvar + self.eps)

        # And create another scaling + shifting linear function
        self.out = self.gamma * x_hat + self.beta 

        # sprinkle in batch means and variance to capture population mean and var
        if self.training:
            with torch.no_grad():
                self.global_mean = (1- self.momentum) * self.global_mean + self.momentum * xmean
                self.global_var = (1 - self.momentum) * self.global_var + self.momentum * xvar
        
        return self.out


    def parameters(self):
        return [self.gamma, self.beta]


class Embedding:

    def __init__(self,num_embeddings,embedding_dim):
        self.weight = torch.randn(num_embeddings,embedding_dim)
    
    def __call__(self,IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]


class Flatten:

    def __call__(self,x):
        self.out = x.view(x.shape[0],-1)
        return self.out
    
    def parameters(self):
        return []


class FlattenConsecutive:

    def __init__(self, n):
        self.n = n

    def __call__(self,x):
        B,T,C = x.shape
        self.out = x.view(B,T//self.n,C*self.n)
        if self.out.shape[1] == 1:
            self.out.squeeze(1)
        return self.out


class ReLU:

    def __call__(self,x):
        self.out = torch.ReLU(x)
        return self.out

    def parameters(self):
        return []


class Tanh:

    def __call__(self,x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []