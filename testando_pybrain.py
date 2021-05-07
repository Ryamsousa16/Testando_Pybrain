from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer


ds = SupervisedDataSet(2, 1)

ds.addSample((0.8, 0.4), (0.7))
ds.addSample((0.5, 0.7), (0.5))
ds.addSample((1.0, 0.8), (0.95))

#Utilizando "bias" para o algoritmo treinar mais rápido
nn = buildNetwork(2, 16, 1,bias =True)

#treinando o algoritmo
trainer = BackpropTrainer(nn, ds)

