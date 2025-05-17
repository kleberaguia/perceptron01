from pyspark.sql import SparkSession

# Cria uma SparkSession
spark = SparkSession.builder.appName("MeuPrimeiroSpark").getOrCreate()

# Cria um DataFrame simples
data = [("Alice", 1), ("Bob", 2), ("Charlie", 3)]
columns = ["nome", "idade"]
df = spark.createDataFrame(data, columns)

# Mostra o DataFrame
df.show()

# Realiza uma transformação: filtra pessoas com idade > 1
df_filtrado = df.filter(df["idade"] > 1)

# Realiza uma ação: mostra o DataFrame filtrado
df_filtrado.show()

# Para a SparkSession
spark.stop()
