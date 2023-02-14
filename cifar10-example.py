
# Imports
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
    print(f"labels_tensor_ohe shape: {labels_tensor_ohe.shape}")
    print(f"data_tensor shape: {data_tensor.shape}")
    return data_tensor, labels_tensor_ohe


selected_x_train, selected_y_train_ohe =\
    make_dataset(ciphar10_train)

# Cerebros configurables:

activation = 'gelu'
predecessor_level_connection_affinity_factor_first = 39.6439
predecessor_level_connection_affinity_factor_main = 0.22216
max_consecutive_lateral_connections = 28
p_lateral_connection = 0.21658
num_lateral_connection_tries_per_unit = 22
learning_rate = 0.0005292
epochs = 5  # [1, 100]
batch_size = 39
maximum_levels = 7  # [3,7]
maximum_units_per_level = 7  # [2,10]
maximum_neurons_per_unit = 3  # [2,20]


## ### replace with this
base_new = tf.keras.applications.MobileNetV3Large(
    input_shape=None,
    alpha=1.0,
    minimalistic=False,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_preprocessing=True,
)

for layer in base_new.layers:
    layer.trainable = True

last_relevant_layer = base_new.layers[-2]
# last_relevant_layer_extracted = last_relevant_layer #.output[0][0][0]
base_embedding = tf.keras.Model(inputs=base_new.layers[0].input,
                                outputs=last_relevant_layer.output)


image_input_0 = tf.keras.layers.Input(shape=INPUT_SHAPES[0])
resizing = tf.keras.layers.Resizing(
    height=RESIZE_TO[0],
    width=RESIZE_TO[1],
    interpolation='bilinear',
    crop_to_aspect_ratio=False)
resized = resizing(image_input_0)
embedded = base_embedding(resized)

embedding_model = tf.keras.Model(image_input_0,
                                 embedded)

# Final training task

TIME = pendulum.now(tz='America/New_York').__str__()[:16]\
    .replace('T', '_')\
    .replace(':', '_')\
    .replace('-', '_')
PROJECT_NAME = f'{TIME}_cerebros_auto_ml_test'

# Cerebros parameters:

training_x = [selected_x_train]
train_labels = [selected_y_train_ohe]

OUTPUT_SHAPES = [10]
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
val_top_1_categorical_accuracy =\
    cerebros_automl.run_random_search()
print(val_top_1_categorical_accuracy)
