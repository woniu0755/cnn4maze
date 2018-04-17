#!/usr/bin/python
# -*- coding: utf8 -*-

import torch 
import gm
import numpy as np
import matplotlib.pyplot as plt 


maze=gm.get_maze()
maze=np.array(maze)
maze=(maze==' ').astype(np.float)


test=torch.Tensor(maze)
pad=torch.nn.ConstantPad2d(1,1)
test=pad(test).view(1,1,23,23)

cnn=torch.nn.Conv2d(1,6,3,1,1,bias=False)
cnn.load_state_dict(torch.load('cnn.pkl'))

while True:
	tmp=cnn(test)
	tmp=tmp.eq(3)
	tmp=tmp.float()
	tmp=tmp.sum(dim=1,keepdim=True)
	tmp=tmp>0.5
	tmp=tmp.float()
	if tmp.equal(test):
		break
	test=tmp

out=test.data.numpy().reshape((23,23))
out=out[1:-1,1:-1]
f1=plt.subplot(2,1,1)
f1.matshow(maze)
f2=plt.subplot(2,1,2)
f2.matshow(out)
plt.savefig('out.jpg')
