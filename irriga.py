import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Exemplo de dados de treinamento (apenas para fins ilustrativos)
# Neste exemplo, temos 4 características: temperatura do ar, umidade do ar, umidade do solo e nível de água no reservatório.
# Os rótulos (saídas desejadas) podem ser, por exemplo, valores para controlar o sistema de irrigação, como a quantidade de água a ser liberada.
# Certifique-se de substituir esses dados por dados reais coletados de sensores.

dados_treinamento = np.array([
    [25.5, 60, 30, 70, 50],   # Exemplo 1: [temperatura, umidade_ar, umidade_solo, nivel_agua_reservatorio, quantidade_agua]
    [28.0, 55, 40, 75, 45],   # Exemplo 2: [temperatura, umidade_ar, umidade_solo, nivel_agua_reservatorio, quantidade_agua]
    [23.8, 65, 35, 60, 55],   # Exemplo 3: [temperatura, umidade_ar, umidade_solo, nivel_agua_reservatorio, quantidade_agua]
    # Adicione mais exemplos de dados de treinamento aqui
])

# Separar as características (temperatura, umidade do ar e do solo, nível de água) dos rótulos (quantidade de água)
caracteristicas_treinamento = dados_treinamento[:, :-1]
rotulos_treinamento = dados_treinamento[:, -1]

# Preparar os dados (normalização, tratamento de valores ausentes, etc.)
# Implemente esta etapa com base nos requisitos específicos dos seus dados.

# Criar o modelo da rede neural
modelo = Sequential()
modelo.add(Dense(16, input_dim=4, activation='relu'))   # Camada oculta com 16 neurônios e função de ativação ReLU
modelo.add(Dense(8, activation='relu'))   # Outra camada oculta com 8 neurônios e função de ativação ReLU
modelo.add(Dense(1, activation='linear'))   # Camada de saída com 1 neurônio e função de ativação linear (pois é uma regressão)

# Compilar o modelo
modelo.compile(loss='mean_squared_error', optimizer='adam')

# Treinar o modelo
modelo.fit(caracteristicas_treinamento, rotulos_treinamento, epochs=100, batch_size=1, verbose=1)

# Exemplo de dados de teste (apenas para fins ilustrativos)
dados_teste = np.array([
    [24.5, 62, 35, 65],   # Exemplo 1: [temperatura, umidade_ar, umidade_solo, nivel_agua_reservatorio]
    [26.8, 58, 38, 72],   # Exemplo 2: [temperatura, umidade_ar, umidade_solo, nivel_agua_reservatorio]
    # Adicione mais exemplos de dados de teste aqui
])

# Preparar as características dos dados de teste
caracteristicas_teste = dados_teste

# Fazer previsões com o modelo treinado
previsoes = modelo.predict(caracteristicas_teste)

# Exibir as previsões (quantidade de água a ser liberada)
print("Previsões de quantidade de água:")
print(previsoes)
