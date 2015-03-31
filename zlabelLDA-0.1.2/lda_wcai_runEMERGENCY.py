import pickle
import math
from numpy import *
from zlabelLDA import zlabelLDA

docs = pickle.load(open("toydocs.txt",'r'))
zlabels = pickle.load(open("zlabeltoy.txt",'r'))


T = 28 #319 #len(train)#240 #319 #Number of topics
W = 184 #len(dictionary.keys()) #vocab size
numsamp = 100 #Num samples from Gibbs sampler
randseed = 194582 #Random to initialize Gibbs sampler Random number Generator
eta = 1


alpha = .1 * ones((1,T))
beta = .1 * ones((T,W))
eta = .95 # confidence in the our label

(phi,theta,sample) = zlabelLDA(docs,zlabels,eta,alpha,beta,numsamp,randseed)


print ''
print 'Theta - P(z|d)'
print array_str(theta,precision=2)
print ''


print ''
print 'Phi - P(w|z)'
print array_str(phi,precision=2)
print ''

pickle.dump(theta,open('theta.txt','w'))
pickle.dump(phi,open('phi.txt','w'))
pickle.dump(sample,open('sample.txt','w'))
