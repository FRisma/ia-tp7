import math
import random
import matplotlib.pyplot as plt
import numpy as np

class Perceptron:
    def __init__(self, inputs=None, weights=None, real_output=None):
        self.inputs = inputs
        self.weights = weights
        self.real_output = real_output

class Event:
    def __init__(self, inputs, output):
        self.inputs = inputs  # Inputs
        self.output = output  # Expected output given inputs

def xor_work():
    # 0 | 0 | 0
    # 0 | 1 | 0
    # 1 | 0 | 0
    # 1 | 1 | 1
    events = [
        Event([1, 0, 0], 0),
        Event([1, 0, 1], 1),
        Event([1, 1, 0], 1),
        Event([1, 1, 1], 0)
    ]

    multilayer_perceptron_work(events= events)


def multilayer_perceptron_work(events: Event):
    error_array = []
    for i in range(len(events)):
        error_array.append([])

    input_layer = []
    hidden_layer = []
    # Add input layers
    # input_layer.append(Perceptron(weights=[0.9, 0.7, 0.5]))
    # input_layer.append(Perceptron(weights=[0.3, -0.9, -1]))

    # Add hidden layers
    number_of_hidden_layers = 10
    for n in range(number_of_hidden_layers):
        hidden_layer.append(Perceptron(weights=[random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)]))

    # Add output layers
    out_lay_weights = []
    for n in range(number_of_hidden_layers+1):
        out_lay_weights.append(random.uniform(-1,1))
    output_layer = Perceptron(weights=out_lay_weights)

    # Work on input layer
    # input_layer_output = []
    # for p in enumerate(input_layer):
    #     calculate_weights(perceptron=p,entries=p.inputs)

    learning_rate = 0.1
    number_of_iterations = 0
    while True:
        number_of_iterations+= 1
        if number_of_iterations == 10000:
            show_plot(data=error_array, title='Multicapa TP4', x_label='Iteraciones', y_label='Errores', label='error ')
            break

        print("\nIteracion ", number_of_iterations)
        print("Learning Rate ", learning_rate)
        for single_row_index, single_row in enumerate(events):
            desired_output = single_row.output
            first_inputs = single_row.inputs

            # Work on hidden layer
            for p_index, p in enumerate(hidden_layer):
                p.inputs = first_inputs
                calculate_output(perceptron=p, inputs_row= single_row.inputs)

            # Work on output layer
            output_layer_inputs = [1]
            for p in hidden_layer:
                output_layer_inputs.append(p.real_output)

            calculate_output(perceptron=output_layer, inputs_row=output_layer_inputs)
            print("| Bias | Input 1 | Input 2 | Desired Output | Real output |")
            print("  ", single_row.inputs[0],"        ", single_row.inputs[1],"       ", single_row.inputs[2],"         ", desired_output,"      ", output_layer.real_output)

            # Modify weights on output layer
            error = calculate_error(desired_output, output_layer.real_output)
            error_array[single_row_index].append(error)
            print("Final real error ", error)
            delta_final = calculate_delta(output_layer.real_output, error)
            for index, single_input in enumerate(output_layer_inputs):
                delta_weight = calculate_delta_weight(learning_rate=learning_rate, input=single_input, delta_final=delta_final)
                new_weight = output_layer.weights[index] + delta_weight
                output_layer.weights[index] = new_weight
            # print("Final weights output layer ", output_layer.weights)

            # Modify weights on hidden layer
            # Soc1 = S1 * (1 - S1) * delta final
            for p in hidden_layer:
                hidden_layer_delta = calculate_delta(p.real_output, delta_final)
                for w_index, weight in enumerate(p.weights):
                    delta_weight_hidden_p = learning_rate * p.inputs[w_index] * hidden_layer_delta
                    new_weight = delta_weight_hidden_p + weight
                    p.weights[w_index] = new_weight

def calculate_output(perceptron=None, inputs_row=None):
    x = calculate_x(inputs = inputs_row, weights = perceptron.weights)
    real_output = calculate_activation(x)
    perceptron.real_output = real_output

def calculate_x(inputs, weights):
    x = 0
    for input_index, singe_input in enumerate(inputs):
        x += singe_input * weights[input_index]
    return x

def calculate_activation(input):
    return 1 / (1 + math.exp(-input))

def calculate_error(desired_output=None, real_output=None):
    return desired_output - real_output

def calculate_delta(real_output=None, error=None):
    return real_output * (1 - real_output) * error

def calculate_delta_weight(learning_rate=None, input=None, delta_final=None):
    return learning_rate * input * delta_final

def show_plot(data=None, title=None, x_label=None, y_label=None, label=None):
    values = np.arange(0, len(data[0]))

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i, element in enumerate(data):
        plot_label = label + str(i)
        plt.plot(values, element, label=plot_label)

    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    xor_work()
