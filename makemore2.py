import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
block_size = 3 # context length: how many characters do we take to predict the next one?
X, Y = [], []
for w in words[:5]:
  
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

C = torch.randn((27,2))
print('this is C')
print(C)
print('this is C[5]')
print(C[5])
print('this is X')
print(X)
print('this is c[X]')
print(C[X])
print('this is C[X][13,2]')
print(C[X][13,2]) # dlaczego to sie rowna C[1]? bo index w 13 wierszu i 2 kolumnie wynosi 1 --> i mapujemy to do wartosci C pod indexem 1 to jest nasza lookup table
