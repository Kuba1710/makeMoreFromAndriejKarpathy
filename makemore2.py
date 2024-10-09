import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

block_size = 3

def build_dataset(words):  
  X, Y = [], []
  for w in words:

    #print(w)
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,2), generator=g)
#print(C[X][13,2]) # dlaczego to sie rowna C[1]? bo index w 13 wierszu i 2 kolumnie wynosi 1 --> i mapujemy to do wartosci C pod indexem 1 to jest nasza lookup table
W1 = torch.randn((6, 100), generator=g, requires_grad=True)
b1 = torch.randn(100, generator=g,requires_grad=True)
W2 = torch.randn((100, 27), generator=g, requires_grad=True)
b2 = torch.randn(27, generator=g, requires_grad=True)
parameters = [W1, b1, W2, b2]

lre = torch.linspace(-3, 0 ,1000)
lrs = 10**lre

lri = []
lossi = []

for i in range(10000):

  #minibatch
  ix = torch.randint(0, Xtr.shape[0], (32,))
  emb = C[Xtr[ix]] # (32,3,2)
  h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32,100)
  logits = h @ W2 + b2 # (32, 27)

  ### this is the same what F.cross_entropy()
  #counts = logits.exp()
  #probs = counts / counts.sum(1, keepdip=True)
  #loss = -probs[torch.arange(32, Y)].log().mean()
  loss = F.cross_entropy(logits, Ytr[ix])
  #print(loss.item())
  #backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  #lr =lrs[i]
  lr = 0.1
  for p in parameters:
    p.data += -lr * p.grad
  
  #lri.append(lre[i])
  #lossi.append(loss.item())

print(loss.item())
#plt.plot(lri, lossi)
#plt.show()

emb = C[Xdev] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ydev)
loss
print(loss.item())
