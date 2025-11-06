#airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks
import keras
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import svm

#v1=keras.datasets.cifar10.load_data()
#print(v1)

(XTreino, yTreino), (XTeste, yTeste) = keras.datasets.cifar10.load_data()

#XTreino, XTeste, yTreino, yTeste = train_test_split(X, y, test_size = 0.2)

XTreino_reshaped = XTreino.reshape((XTreino.shape[0], -1))
XTeste_reshaped = XTeste.reshape((XTeste.shape[0], -1))

XTreino_normalized = XTreino_reshaped / 255.0
XTeste_normalized = XTeste_reshaped / 255.0

yTreino_flat = yTreino.flatten()
yTeste_flat = yTeste.flatten()
clf2 = DecisionTreeClassifier()
#clf2.fit(XTreino_normalized[:5000],yTreino_flat[:5000])
clf2.fit(XTreino_normalized,yTreino_flat)
class_names = ['avião', 'automóvel', 'pássaro', 'gato', 'cervo', 'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']
esperado2 = yTeste_flat
previsto2 = clf2.predict(XTeste_normalized)
print(classification_report(esperado2, previsto2, target_names=class_names))