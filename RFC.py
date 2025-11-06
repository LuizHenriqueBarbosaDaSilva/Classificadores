from keras.datasets import cifar10
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Carrega o dataset CIFAR-10
# (XTreino tem 50.000 imagens, XTeste tem 10.000)
(XTreino, yTreino), (XTeste, yTeste) = cifar10.load_data()

# Achata e normaliza as imagens (de 32x32x3 para 3072)
XTreino = XTreino.reshape((XTreino.shape[0], -1)) / 255.0
XTeste = XTeste.reshape((XTeste.shape[0], -1)) / 255.0

# Achata os rótulos (y)
yTreino = yTreino.flatten()
yTeste = yTeste.flatten()

# Cria o classificador sem limitar profundidade
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

# ESTA LINHA TREINA O MODELO COM TODAS AS 50.000 IMAGENS
print("Treinando o Random Forest com todas as 50.000 imagens de treino...")
clf.fit(XTreino, yTreino)
print("Treinamento concluído.")

# Avalia o modelo nas 10.000 imagens de teste
previsto = clf.predict(XTeste)
class_names = ['avião', 'automóvel', 'pássaro', 'gato', 'cervo', 'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']

print("\nRelatório de Classificação Completo:\n")
print(classification_report(yTeste, previsto, target_names=class_names))