import numpy as np

class SimplePerceptron:
    #variaveis
    def __init__(self, taxa_aprendizado = 0.5):
        self.taxaaprendizado = taxa_aprendizado
        self.w = np.zeros(4)
        self.bias = 0

    def funcao_ativacao(self, w_sum):
        if w_sum > 0:
            return 1
        else:
            return -1
        
    #saida do perceptron
    def predict(self, inputs):
        inputs = np.array(inputs)
        if len(inputs) !=len(self.w):
            raise ValueError(f"O tamanho esperado é {len(self.w)}")
        w_sum =  np.dot(inputs, self.w) + self.bias
        output = self.funcao_ativacao(w_sum)
        return output, w_sum

    def update_w(self, inputs, esperado,calculo_atual):
        error = esperado-calculo_atual
        if error != 0:
            self.w += self.w + self.taxaaprendizado * error * np.array(inputs)
            self.bias = self.bias + self.taxaaprendizado * error
        print(f"Pesos atuatualizados: {self.w}")
        print(f"Bias atualizados:{self.bias}")
        print(f"Erro: {error}")
        print(f"saida atual: {calculo_atual}")
        print(f"saida esperada: {esperado}")





perceptron =  SimplePerceptron(taxa_aprendizado=0.5)
treinamento_inputs = [1,-1,1,1]

treinamento_esperado = 1


print(f"Entrada de treinamento:{treinamento_inputs}")
print(f"Entrada de treinamento:{treinamento_esperado}")

atual_saida_depois, sum_before = perceptron.predict(treinamento_inputs)
print(f"Predição antes de atualizar os pesos: {atual_saida_depois }(Soma ponderada + bias:{sum_before:.2f})")

perceptron.update_w(treinamento_inputs,treinamento_esperado,atual_saida_depois)


atual_saida_depois, sum_after = perceptron.predict(treinamento_inputs)
print(f"Predição depois de atualizar os pesos: {atual_saida_depois}(Soma ponderada + bias:{sum_after})")

if atual_saida_depois == treinamento_esperado:
    print(f"Saida Correta")
else:
    print("O perceptron ainda não classifica esta entrada corretamente após 1 pesso.")
    print("Pode ser necessario mais exemplos de treinamento ou épocas para convergir")

