import math
import random
import matplotlib.pyplot as plt
import numpy as np
from imageAnalizer import ImageAnalizer


class Perceptron:
    def __init__(self, inputs=None, weights=None, real_output=None):
        self.inputs = inputs
        self.weights = weights
        self.real_output = real_output


class Event:
    def __init__(self, inputs, output, should_backpropagate):
        self.inputs = inputs  # Inputs (array de pixeles)
        self.output = output  # Expected output given inputs (0 o 1)
        self.should_backpropagate = should_backpropagate


def image_work():
    image_analizer = ImageAnalizer()
    gestureA1 = image_analizer.get_grayscale_array(image_name="../ia-tp5/1A4353.jpg", width=80, height=96)
    gestureB1 = image_analizer.get_grayscale_array(image_name="../ia-tp5/1B4698.jpg", width=80, height=96)
    gestureA2 = image_analizer.get_grayscale_array(image_name="../ia-tp5/2A4353.jpg", width=80, height=96)
    gestureB2 = image_analizer.get_grayscale_array(image_name="../ia-tp5/2B4698.jpg", width=80, height=96)
    gestureA3 = image_analizer.get_grayscale_array(image_name="../ia-tp5/3A4353.jpg", width=80, height=96)
    gestureB3 = image_analizer.get_grayscale_array(image_name="../ia-tp5/3B4698.jpg", width=80, height=96)
    gestureA4 = image_analizer.get_grayscale_array(image_name="../ia-tp5/4A4353.jpg", width=80, height=96)
    gestureB4 = image_analizer.get_grayscale_array(image_name="../ia-tp5/4B4698.jpg", width=80, height=96)
    gestureA5 = image_analizer.get_grayscale_array(image_name="../ia-tp5/5A4353.jpg", width=80, height=96)
    gestureB5 = image_analizer.get_grayscale_array(image_name="../ia-tp5/5B4698.jpg", width=80, height=96)
    # Diego --> 0
    # Franco --> 1

    # Witness images, do not backpropagate
    gestureA6 = image_analizer.get_grayscale_array(image_name="../ia-tp5/6A4353.jpg", width=80, height=96)
    gestureB6 = image_analizer.get_grayscale_array(image_name="../ia-tp5/6B4698.jpg", width=80, height=96)
    gestureA7 = image_analizer.get_grayscale_array(image_name="../ia-tp5/7A4353.jpg", width=80, height=96)
    gestureB7 = image_analizer.get_grayscale_array(image_name="../ia-tp5/7B4698.jpg", width=80, height=96)
    gestureA8 = image_analizer.get_grayscale_array(image_name="../ia-tp5/8A4353.jpg", width=80, height=96)
    gestureB8 = image_analizer.get_grayscale_array(image_name="../ia-tp5/8B4698.jpg", width=80, height=96)

    # Add bias
    gestureA1.insert(0, 1)
    gestureB1.insert(0, 1)
    gestureA2.insert(0, 1)
    gestureB2.insert(0, 1)
    gestureA3.insert(0, 1)
    gestureB3.insert(0, 1)
    gestureA4.insert(0, 1)
    gestureB4.insert(0, 1)
    gestureA5.insert(0, 1)
    gestureB5.insert(0, 1)
    gestureA6.insert(0, 1)
    gestureB6.insert(0, 1)
    gestureA7.insert(0, 1)
    gestureB7.insert(0, 1)
    gestureA8.insert(0, 1)
    gestureB8.insert(0, 1)

    # Ceate matrix
    events = [
        Event(gestureA1, 0, should_backpropagate=True),
        Event(gestureB1, 1, should_backpropagate=True),
        Event(gestureA2, 0, should_backpropagate=True),
        Event(gestureB2, 1, should_backpropagate=True),
        Event(gestureA3, 0, should_backpropagate=True),
        Event(gestureB3, 1, should_backpropagate=True),
        Event(gestureA4, 0, should_backpropagate=True),
        Event(gestureB4, 1, should_backpropagate=True),
        Event(gestureA5, 0, should_backpropagate=True),
        Event(gestureB5, 1, should_backpropagate=True),
        Event(gestureA6, 0, should_backpropagate=False),
        Event(gestureB6, 1, should_backpropagate=False),
        Event(gestureA7, 0, should_backpropagate=False),
        Event(gestureB7, 1, should_backpropagate=False),
        Event(gestureA8, 0, should_backpropagate=False),
        Event(gestureB8, 1, should_backpropagate=False),
    ]

    # Configurations
    learning_rate = 0.5
    number_of_hidden_layers = 100
    max_number_of_iterations = 100

    multilayer_perceptron_work(events=events,
                               max_iterations=max_number_of_iterations,
                               learning_rate=learning_rate,
                               number_of_hidden_layers=number_of_hidden_layers)

def multilayer_perceptron_work(events: Event, max_iterations, learning_rate, number_of_hidden_layers):
    error_array = []
    witness_real_output_array = []
    for event_index, event in enumerate(events):
        if not event.should_backpropagate:
            witness_real_output_array.append({
                "name": event_index,
                "value": []
            })
        else:
            error_array.append([])

    # Add hidden layers
    hidden_layer = []
    for n in range(number_of_hidden_layers):
        perceptron_weights = []
        for x in range(len(events[0].inputs)):
            perceptron_weights.append(calculate_random_weight())
        hidden_layer.append(Perceptron(weights=perceptron_weights))

    # Add output layers
    out_lay_weights = []
    for n in range(number_of_hidden_layers + 1):
        out_lay_weights.append(calculate_random_weight())
    output_layer = Perceptron(weights=out_lay_weights)

    number_of_iterations = 0
    while True:
        number_of_iterations += 1
        if number_of_iterations == max_iterations:
            # Validamos las personas
            # validate_results(hidden_layer=hidden_layer, output_layer= output_layer)
            # show_plot(data=error_array, title='Imagenes TP6', x_label='Iteraciones', y_label='Errores', label='error ')
            show_witness_plot(data=witness_real_output_array, title='Witness images', x_label='Iteraciones', y_label='Salida real', label='salida ')
            break

        print("\nIteracion ", number_of_iterations)
        print("Learning Rate ", learning_rate)
        for single_row_index, single_row in enumerate(events):
            # print("Procesando imagen numero ", single_row_index)
            desired_output = single_row.output
            first_inputs = single_row.inputs

            # Work on hidden layer
            for p_index, p in enumerate(hidden_layer):
                p.inputs = first_inputs
                calculate_output(perceptron=p, inputs_row=single_row.inputs)

            # Work on output layer
            output_layer_inputs = [1]
            for p in hidden_layer:
                output_layer_inputs.append(p.real_output)

            calculate_output(perceptron=output_layer, inputs_row=output_layer_inputs)

            if not single_row.should_backpropagate:
                print("Witness image, skipping backpropagation")
                witness_element = [x for x in witness_real_output_array if x["name"]==single_row_index]
                witness_element[0]["value"].append(output_layer.real_output)
                continue

            # Modify weights on output layer
            error = calculate_error(desired_output, output_layer.real_output)
            error_array[single_row_index].append(error)

            delta_final = calculate_delta(output_layer.real_output, error)
            for index, single_input in enumerate(output_layer_inputs):
                delta_weight = calculate_delta_weight(learning_rate=learning_rate,
                                                      input=single_input,
                                                      delta_final=delta_final)
                new_weight = output_layer.weights[index] + delta_weight
                output_layer.weights[index] = new_weight

            # Modify weights on hidden layer
            # Soc1 = S1 * (1 - S1) * delta final
            for p in hidden_layer:
                hidden_layer_delta = calculate_delta(p.real_output, delta_final)
                for w_index, weight in enumerate(p.weights):
                    if len(p.weights) != len(p.inputs):
                        print("Error weights ", len(p.weights), " and inputs ", len(p.inputs))
                    delta_weight_hidden_p = learning_rate * p.inputs[w_index] * hidden_layer_delta
                    new_weight = delta_weight_hidden_p + weight
                    p.weights[w_index] = new_weight


def calculate_output(perceptron=None, inputs_row=None):
    x = calculate_x(inputs=inputs_row, weights=perceptron.weights)
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


def show_witness_plot(data=None, title=None, x_label=None, y_label=None, label=None):
    values = np.arange(0, len(data[0]["value"]))

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i, element in enumerate(data):
        plot_label = element["name"]
        plt.plot(values, element["value"], label=plot_label)

    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()


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

def calculate_random_weight():
    weight_lower_bound = -1
    weight_upper_bound = 1
    return random.uniform(weight_lower_bound, weight_upper_bound)


def validate_results(hidden_layer: [Perceptron], output_layer: Perceptron):
    print("VALIDAMOS Diego 0 -- Franco 1")
    image_analizer = ImageAnalizer()
    gestureA6 = image_analizer.get_grayscale_array(image_name="../ia-tp5/6A4353.jpg", width=80, height=96)
    gestureB6 = image_analizer.get_grayscale_array(image_name="../ia-tp5/6B4698.jpg", width=80, height=96)
    gestureA7 = image_analizer.get_grayscale_array(image_name="../ia-tp5/7A4353.jpg", width=80, height=96)
    gestureB7 = image_analizer.get_grayscale_array(image_name="../ia-tp5/7B4698.jpg", width=80, height=96)
    gestureA8 = image_analizer.get_grayscale_array(image_name="../ia-tp5/8A4353.jpg", width=80, height=96)
    gestureB8 = image_analizer.get_grayscale_array(image_name="../ia-tp5/8B4698.jpg", width=80, height=96)
    # Diego --> 0
    # Franco --> 1

    # Add bias
    gestureA6.insert(0, 1)
    gestureB6.insert(0, 1)
    gestureA7.insert(0, 1)
    gestureB7.insert(0, 1)
    gestureA8.insert(0, 1)
    gestureB8.insert(0, 1)

    # Ceate matrix
    validation_events = [
        Event(gestureA6, 0),
        Event(gestureB6, 1),
        Event(gestureA7, 0),
        Event(gestureB7, 1),
        Event(gestureA8, 0),
        Event(gestureB8, 1)
    ]

    for single_row_index, single_row in enumerate(validation_events):
        if single_row_index %2 == 0:
            print("Procesando foto de Diego")
        else:
            print("Procesando foto de Franco")
        desired_output = single_row.output
        first_inputs = single_row.inputs

        # Work on hidden layer
        for p_index, p in enumerate(hidden_layer):
            p.inputs = first_inputs
            calculate_output(perceptron=p, inputs_row=single_row.inputs)

        # Work on output layer
        output_layer_inputs = [1]
        for p in hidden_layer:
            output_layer_inputs.append(p.real_output)

        calculate_output(perceptron=output_layer, inputs_row=output_layer_inputs)
        print("Resultado final para la foto ", single_row_index, " es ", output_layer.real_output, "y lo esperado era ", desired_output)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image_work()