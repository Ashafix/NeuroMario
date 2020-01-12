import os
import sys
import numpy as np
import tensorflow as tf
import keras

from MachineLearning import MachineLearning, get_standard_model

PICKLE_INPUT = 'input_rgb.pickle.npy'
PICKLE_OUTPUT = 'output_rgb.pickle.npy'
MODEL_NAME = 'frozen_model.pb'


def main():
    ml = MachineLearning()

    if os.path.isfile(PICKLE_INPUT) and os.path.isfile(PICKLE_OUTPUT):
        ml.input = np.load(PICKLE_INPUT)
        ml.output = np.load(PICKLE_OUTPUT)
    else:
        for direc in os.listdir(os.path.join(os.getcwd(), 'movies_images')):
            dir_name = os.path.join(os.getcwd(), 'movies_images', direc)
            if direc.endswith('bk2') and os.path.isdir(dir_name):
                ml.add_dir(dir_name)

        ml.input_output(normalize=False, mirror=False,
                        bit_array=False, gray=False)

        np.save(PICKLE_INPUT, ml.input)
        np.save(PICKLE_OUTPUT, ml.output)

    inp = ml.input.astype('uint8')
    output = ml.output

    keep_prob = 0.8

    train_graph = tf.Graph()
    train_sess = tf.compat.v1.Session(graph=train_graph)

    keras.backend.set_session(train_sess)

    def build_keras_model():
        model = get_standard_model(inp.shape[1:], output.shape[1], 'sigmoid', keep_prob)
        return model

    with train_graph.as_default():
        train_model = build_keras_model()

        tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
        train_sess.run(tf.global_variables_initializer())

        train_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        train_model.fit(inp, output, epochs=5)

        return train_model

        # save graph and checkpoints
        saver = tf.train.Saver()
        saver.save(train_sess, 'checkpoints')

    eval_graph = tf.Graph()
    eval_sess = tf.Session(graph=eval_graph)

    keras.backend.set_session(eval_sess)

    with eval_graph.as_default():
        keras.backend.set_learning_phase(0)
        eval_model = build_keras_model()
        tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
        eval_graph_def = eval_graph.as_graph_def()
        saver = tf.train.Saver()
        saver.restore(eval_sess, 'checkpoints')

        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            eval_sess,
            eval_graph_def,
            [eval_model.output.op.name]
        )

        with open(MODEL_NAME, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

    print("""
    tflite_convert --output_file=model.tflite --graph_def_file='{} --inference_type=QUANTIZED_UINT8 --input_arrays=conv2d_1_input --output_arrays=dense_5/act_quant/FakeQuantWithMinMaxVars --mean_values=0 --std_dev_values=255
     """.format(MODEL_NAME))

    print('edgetpu_compiler --min_runtime_version 10 model.tflite')


if __name__ == '__main__':
    model = main()

