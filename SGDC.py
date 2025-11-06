import keras
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

(XTreino, yTreino), (XTeste, yTeste) = keras.datasets.cifar10.load_data()
XTreino_reshaped = XTreino.reshape((XTreino.shape[0], -1))
XTeste_reshaped  = XTeste.reshape((XTeste.shape[0], -1))

# Normaliza e padroniza (necessário para modelos lineares)
XTreino_normalized = XTreino_reshaped / 255.0
XTeste_normalized  = XTeste_reshaped / 255.0
scaler = StandardScaler()
XTreino_normalized = scaler.fit_transform(XTreino_normalized)
XTeste_normalized  = scaler.transform(XTeste_normalized)

yTreino_flat = yTreino.flatten()
yTeste_flat  = yTeste.flatten()

# SGD com loss='log_loss' → similar à Regressão Logística
clf2 = SGDClassifier(loss='log_loss', max_iter=60000, tol=1e-3, random_state=42)
clf2.fit(XTreino_normalized, yTreino_flat)
previsto2 = clf2.predict(XTeste_normalized)

class_names = ['avião','automóvel','pássaro','gato','cervo','cachorro','sapo','cavalo','navio','caminhão']
print(classification_report(yTeste_flat, previsto2, target_names=class_names))