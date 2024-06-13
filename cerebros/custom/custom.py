
import tensorflow as tf
from keras_nlp.models import GPT2Tokenizer, GPT2Preprocessor, GPT2Backbone

# Custom keras layer, analogue of Dense that does 
# ternary mathematical operations 
class TernaryDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, input_dim, **kwargs):
        super(TernaryDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self.ternary_weights = self.add_weight(name='ternary_weights', 
                                                shape=(input_dim, units),
                                                initializer='glorot_uniform',
                                                trainable=True)

    def build(self, input_shape):
        # Create a trainable weight variable for the bias
        self.bias = self.add_weight(name='bias', 
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs):
        # Apply ternary weights to the input vector
        ternary_inputs = tf.cast(tf.sign(inputs), tf.float32) * tf.abs(inputs)
        output = tf.matmul(ternary_inputs, self.ternary_weights)
        # Add bias and apply activation function
        output = tf.nn.bias_add(output, self.bias)
        output = tf.nn.relu(output)
        return output

# Utility layer for encoding for GPT
class GPT2Layer(tf.keras.layers.Layer):
    """### A custom GPT2 encoder layer for text embedding"""
    def __init__(self, max_seq_length, **kwargs):
        #
        super(GPT2Layer, self).__init__(**kwargs)
        #
        # Load the GPT2 tokenizer, preprocessor and model
        self.tokenizer = GPT2Tokenizer.from_preset("gpt2_base_en")
        self.preprocessor = GPT2Preprocessor(self.tokenizer,
                                             sequence_length=max_seq_length)
        self.encoder   = GPT2Backbone.from_preset("gpt2_base_en")
        #
        # Set whether the GPT2 model's layers are trainable
        #self.encoder.trainable = False
        for layer in self.encoder.layers:
            layer.trainable = False
        #
        self.encoder.layers[-2].trainable = True
        #
        # Set the maximum sequence length for tokenization
        self.max_seq_length = max_seq_length

    def call(self, inputs):
        #
        # Output the GPT2 embedding
        prep = self.preprocessor([inputs])
        embedding  = self.encoder(prep)
        avg_pool = tf.reduce_mean(embedding, axis=1)
        #
        return avg_pool

    def get_config(self):
        #
        config = super(GPT2Layer, self).get_config()
        config.update({'max_seq_length': self.max_seq_length})
        #
        return config

    @classmethod
    def from_config(cls, config):
        #
        return cls(max_seq_length=config['max_seq_length'])

