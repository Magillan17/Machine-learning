import numpy as np
import pandas as pd
import time
from scipy import ndimage

df = pd.read_csv('df_rotate_120000.csv')
df_test = pd.read_csv('test.csv')


# df_test = df[1000:]
# df = df[1000:1390]

# df_test = df[40000:]
# df = df[:40000]



class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, lr):
        # кол-во узлов в входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # lr
        self.lr = lr
        #str 163 yly4shennii koef
        self.w_ih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        self.w_ho = (np.random.rand(self.onodes, self.hnodes) - 0.5)

        #использую сигмоиду в качестве функции активации
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))
        pass

    def train(self, inputs, targets):
        hidden_inputs = self.w_ih @ inputs
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = self.w_ho @ hidden_outputs
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = self.w_ho.T @ output_errors

        hidden_outputs = np.reshape(hidden_outputs, (1, hidden_outputs.shape[0]))
        a_1 = np.reshape(output_errors * final_outputs * (1 - final_outputs),
                       ((output_errors * final_outputs * (1 - final_outputs)).shape[0],1))
        self.w_ho += self.lr * (a_1 @ hidden_outputs)


        hidden_errors = np.reshape(hidden_errors, (hidden_errors.shape[0], 1))
        hidden_outputs = hidden_outputs.T
        a_2 = hidden_errors * hidden_outputs * (1.0 - hidden_outputs)
        inputs = np.reshape(inputs, (1, inputs.shape[0]))
        self.w_ih += self.lr * (a_2 @ inputs)

        pass

    # opros neironnoy seti
    def query(self, inputs):
        # assert inputs.shape[1] == 2
        """почему не наоборот? почему веса на инпуты"""
        hidden_inputs = self.w_ih @ inputs
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = self.w_ho @ hidden_outputs
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


        pass


i_n = 784
h_n = 500
o_n = 10
lr = 0.01
n = neuralNetwork(i_n, h_n, o_n, lr)

y = np.array(df['label'])
x = np.array(df.drop(['label'],axis=1))
x = x / 255 * 0.99  + 0.01

# x_test = np.array(df_test.drop(['label'],axis=1))
# x_test = x_test / 255 * 0.99  + 0.01
# y_test = np.array(df_test['label'])


start_time = time.time()
for epoch in range(10):
    for example in range(len(x)):
        targets = np.zeros(o_n) + 0.01
        targets[int(y[example])] = 0.99
        n.train(x[example], targets)

    print("--- %s seconds ---" % (time.time() - start_time))


    # acc = 0
    # for t in range(len(x_test)):
    #     outputs = n.query(x_test[t])
    #     label = np.argmax(outputs)
    #     if label == y_test[t]:
    #         acc += 1
    #     else:
    #         acc += 0
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print(f'epoch {epoch + 1}')
    # print(f'lr: {lr}, skritiy sloy {h_n} accuracy: {acc / len(y_test)}')

"""Kaggle"""

x_test = np.array(df_test)
x_test = x_test / 255 * 0.99  + 0.01
sub = pd.read_csv('sample_submission.csv')
answer = []
for t in range(len(x_test)):
    outputs = n.query(x_test[t])
    label = np.argmax(outputs)
    answer.append(label)


sub['Label'] = pd.Series(answer)
sub.to_csv('Answer11.csv', index=False)

print('Finish')