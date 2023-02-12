
# Imports
import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import imageio.v3 as iio
# from multiprocessing import Pool  # , Process
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search\
    import SimpleCerebrosRandomSearch
import pendulum
from cerebros.units.units import DenseUnit
from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component\
    import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid
from ast import literal_eval

### Global configurables:


# How many of the samples in the data set to actually use on this training run
number_of_samples_to_use = 200
INPUT_SHAPES = [(32, 32, 3)]  # resize from ]
RESIZE_TO = (224, 224, 3)

# Read in the data set and make it useable

ciphar10_metadata = pd.read_csv("cifar10-mini/file_metadata.csv")

ciphar10_train = ciphar10_metadata.query("data_set == 'train'")
ciphar10_test = ciphar10_metadata.query("data_set == 'test'")


def make_dataset(dataset):
    images = []
    labels = []
    for i in np.arange(ciphar10_metadata.shape[0]):
        imfile = ciphar10_metadata.loc[i]['file_name']

        # Debug delete
        # print(f"$$$$: attempting file: {imfile}")

        img = iio.imread(imfile)

        images.append(np.array(img))
        labels.append(int(ciphar10_metadata.loc[i]['label']))
    data_tensor = tf.constant(images)
    labels_tensor = tf.constant(labels)
    labels_tensor_ohe = tf.one_hot(indices=labels_tensor,
                                   depth=10)
    print(f"labels_tensor shape: {labels_tensor.shape}")
    print(f"data_tensor shape: {data_tensor.shape}")
    return data_tensor, labels_tensor_ohe


selected_x_train, selected_y_train_ohe =\
    make_dataset(ciphar10_train)

# Cerebros configurables:

activation = 'relu'
predecessor_level_connection_affinity_factor_first = 2.0
predecessor_level_connection_affinity_factor_main = 0.97
max_consecutive_lateral_connections = 5
p_lateral_connection = 0.97
num_lateral_connection_tries_per_unit = 2
learning_rate = 0.001
epochs = 10  # [1, 100]
batch_size = 20
maximum_levels = 4  # [3,7]
maximum_units_per_level = 7  # [2,10]
maximum_neurons_per_unit = 4  # [2,20]


# Build BERT base model
image_input_0 = tf.keras.layers.Input(shape=INPUT_SHAPES[0])
resized = tf.keras.layers.Resizing(
    height=RESIZE_TO[0],
    width=RESIZE_TO[1],
    interpolation='bilinear',
    crop_to_aspect_ratio=False)(image_input_0)

base_model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/"\
    + "classification/4"
preprocessor = hub.KerasLayer(base_model_url,
                              output_shape=[1001])
preprocessor_output = preprocessor(resized)
foundation_model = tf.keras.Model(image_input_0,
                                  preprocessor_output)

for layer in foundation_model.layers:
    layer.trainable = True

relevant_layers = foundation_model.layers[-2]
embedding_model = tf.keras.Model(inputs=foundation_model.layers[0].input,
                                 outputs=foundation_model.layers[-2].output)

##### Fix the data ingestion and the Cerebros config ...
### Load the Data set ... repalce with image ingestion
# raw_text = pd.read_csv(data_file, dtype='object')
# raw_text = raw_text.iloc[:number_of_samples_to_use, :]
# One hot encode the label
# raw_text[prediction_target_column] =\
#   raw_text[prediction_target_column]\
#   .apply(lambda x: 1 if x == positive_class_label else 0)

TIME = pendulum.now(tz='America/New_York').__str__()[:16]\
    .replace('T', '_')\
    .replace(':', '_')\
    .replace('-', '_')
PROJECT_NAME = f'{TIME}_cerebros_auto_ml_test'
INPUT_SHAPES = [()]

# Cerebros parameters:

training_x = [selected_x_train]
train_labels = [selected_y_train_ohe]

OUTPUT_SHAPES = [3]
meta_trial_number = str(int(np.random.random() * 10 ** 12))

cerebros_automl = SimpleCerebrosRandomSearch(
    unit_type=DenseUnit,
    input_shapes=INPUT_SHAPES,
    output_shapes=OUTPUT_SHAPES,
    training_data=training_x,
    labels=train_labels,
    validation_split=0.35,
    direction='maximize',
    metric_to_rank_by="val_top_1_categorical_accuracy",
    minimum_levels=2,
    maximum_levels=maximum_levels,
    minimum_units_per_level=1,
    maximum_units_per_level=maximum_units_per_level,
    minimum_neurons_per_unit=1,
    maximum_neurons_per_unit=maximum_neurons_per_unit,
    activation=activation,
    final_activation='softmax',
    number_of_architecture_moities_to_try=2,
    number_of_tries_per_architecture_moity=1,
    minimum_skip_connection_depth=1,
    maximum_skip_connection_depth=7,
    predecessor_level_connection_affinity_factor_first=predecessor_level_connection_affinity_factor_first,
    predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
    predecessor_level_connection_affinity_factor_main=predecessor_level_connection_affinity_factor_main,
    predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
    predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
    seed=8675309,
    max_consecutive_lateral_connections=max_consecutive_lateral_connections,
    gate_after_n_lateral_connections=3,
    gate_activation_function=simple_sigmoid,
    p_lateral_connection=p_lateral_connection,
    p_lateral_connection_decay=zero_95_exp_decay,
    num_lateral_connection_tries_per_unit=num_lateral_connection_tries_per_unit,
    learning_rate=learning_rate,
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.TopKCategoricalAccuracy(
                k=1, name='top_1_categorical_accuracy')
             ],
    epochs=epochs,
    project_name=f"{PROJECT_NAME}_meta_{meta_trial_number}",
    # use_multiprocessing_for_multiple_neural_networks=False,  # pull this param
    model_graphs='model_graphs',
    batch_size=batch_size,
    meta_trial_number=meta_trial_number,
    base_models=[embedding_model])
val_binary_accuracy =\
    cerebros_automl.run_random_search()
print(val_binary_accuracy)
