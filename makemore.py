import torch
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

words = open('names.txt', 'r').read().splitlines()

N = torch.zeros((27,27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

itos = {i:s for s,i in stoi.items()}

#plt.figure(figsize=(16,16))
#plt.imshow(N, cmap='Blues')
#for i in range(27):
#    for j in range(27):
#        chstr = itos[i] + itos[j]
#        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
#        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
#plt.axis('off')
#plt.show()

g = torch.Generator().manual_seed(2147483647)

P = (N+1).float()
P /=  P.sum(1, keepdims=True)

for i in range(5):

    out = []
    ix = 0
    while True:

        p = P[ix]
        #p = N[ix].float()
        #p = p / p.sum()

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    
    #print(''.join(out))

log_likelihood = 0.0
n = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        #print (f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
    
#print(f'{log_likelihood =}')
nll = -log_likelihood
#print(f'{nll=}')
#print(f'{nll/n=}')

xs, ys = [], []

for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        print(ch1, ch2)
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

for k in range(200):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(xs, num_classes=27).float() # zmiana danych w wejsciowych w wektor 0 i 1 
        #plt.imshow(xenc)
        #plt.show()
        #print((xenc @ W)[3,13])
        #print(xenc[3])
        #print(W[:, 13])
        #print((xenc[3] * W[:, 13]).sum())
        #print((xenc[3] * W[:, 13]))

        logits = xenc @ W #log-counts
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        loss = -probs[torch.arange(5), ys].log().mean()
        print(loss.item())

        #backward pass
        W.grad = None #gradient to zero
        loss.backward()

        W.data += -50 * W.grad




