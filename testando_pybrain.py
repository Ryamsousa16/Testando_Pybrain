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

#analisando a evolução do algoritmo
for i in range(2000):
    print(trainer.train())

#com o buildNetwork utilizando 4 neurônios, temos 0.013496550372475475 de margem de erro
#com o buildNetwork utilizando 16 neurônios, temos 0.0009338957668739359 de margem de erro
#teste 1 com 512 neurônios: 1.6434602192104412e-32 de margem de erro
#teste 2 com 512 neurônios: 1.3702349577667055e-30 de margem de erro
#irei utilizar neste algoritmo apenas 16 neurônios

#alguns testes por curiosidade
while True:
    sleep = float(input("How many hours did you sleep? "))
    study = float(input("How many hours did you study? "))

    z = nn.activate((sleep, study))[0] * 10.0

    if sleep >= 1.1 or study >= 1.1:
        break

    print("{:.2f}".format(z))
