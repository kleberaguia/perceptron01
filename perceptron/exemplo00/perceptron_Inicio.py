
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import MultilayerPerceptronClassifier # Podemos usar este para simplicidade
from pyspark.sql.functions import lit

#Inicializa a SparkSession
spark = SparkSession.builder.appName("SimplePerceptron").getOrCreate()

#Cria dados de treinamento de exemplo
data = [
(Vectors.dense([1.0, 1.0]), 0.0),  # Classe 0
(Vectors.dense([0.0, 0.0]), 0.0),  # Classe 0
(Vectors.dense([1.0, 0.0]), 1.0),  # Classe 1
(Vectors.dense([0.0, 1.0]), 1.0)   # Classe 1
]

df = spark.createDataFrame(data, ["features", "label"])

#Define as camadas do Perceptron (MultilayerPerceptronClassifier)
#Para um Perceptron simples, podemos pensar em uma camada de entrada,
#uma camada oculta com poucos neurônios (ex: 2), e uma camada de saída (2 classes).
layers = [2, 2, 2] # Camada de entrada (2 features), 1 camada oculta com 2 neurônios, camada de saída (2 classes)

#Cria o treinador do Perceptron
trainer = MultilayerPerceptronClassifier(layers=layers, seed=42)

#Treina o modelo
model = trainer.fit(df)

#Cria dados de teste
test_data = [(Vectors.dense([1.1, 0.9]),), (Vectors.dense([0.1, 0.1]),)]
test_df = spark.createDataFrame(test_data, ["features"])

#Faz as previsões
predictions = model.transform(test_df)
predictions.select("features", "prediction").show()

#Adiciona rótulos de exemplo para avaliação (apenas para ilustração)
labeled_test = spark.createDataFrame([(Vectors.dense([1.1, 0.9]), 1.0), (Vectors.dense([0.1, 0.1]), 0.0)], ["features", "label"])
results = model.transform(labeled_test)
results.select("prediction", "label").show()

#Para a SparkSession

spark.stop()