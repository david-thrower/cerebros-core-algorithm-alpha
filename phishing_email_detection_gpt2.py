import tensorflow as tf
import tensorflow_text
from keras_nlp.models import GPT2Tokenizer, GPT2Preprocessor, GPT2Backbone
from keras_nlp.layers import PositionEmbedding
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten
import pandas as pd
import numpy as np
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search\
    import SimpleCerebrosRandomSearch
import pendulum
from cerebros.units.units import DenseUnit
from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component\
    import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid
from ast import literal_eval
import time
from gc import collect
from os.path import getsize
import re

# Text encoding / embedding related constants


MAX_SEQ_LENGTH = 196 # 1536

tokenizer_checkpoint = "HuggingFaceTB/SmolLM3-3B" # "HuggingFaceTB/SmolLM2-1.7B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

# Step 1: Add special tokens
special_tokens = {
    "additional_special_tokens": ["<prompt>", "</prompt>", "<response>", "</response>"]
}
tokenizer.add_special_tokens(special_tokens)

VOCABULARY_SIZE = len(tokenizer)

# For interleaved Rotary Positional Embedding (iRoPE), the 
# embedding output dim must be an even number
# Maximize EMBEDDING_N based on available RAM and CPU / GPU

EMBEDDING_N = 12
EMBEDDING_DIM = int(EMBEDDING_N * 2)

PROJECTION_N = 1

## Parameters to Optimize with Hyperparam Optimization


POSITIONAL_EMBEDDING_DROPOUT = 0.43 # (Tune with Hyperparameter Optimization)

#
# Cerebros configurables (Parameters to Optimize continued)
#
activation = "relu"
predecessor_level_connection_affinity_factor_first = 10
predecessor_level_connection_affinity_factor_main = 40
max_consecutive_lateral_connections = 20
p_lateral_connection = 30
num_lateral_connection_tries_per_unit = 25
learning_rate = 3 * 10 ** -4 # 3 * 10 ** -3
epochs = 15  #
batch_size = 5 # 17
gradient_accumulation_steps = 1
minimum_levels = 2
maximum_levels = 2 # [3,7]

minimum_units_per_level = 4
maximum_units_per_level = 7

minimum_neurons_per_unit = 1
maximum_neurons_per_unit = 2

moities_to_try = 5
tries_per_moity = 1

####### DO NOT FORGET TO MERGE IN THE WORK THAT ADDED GRADIENT
####### ACCUMULATION STEPS AND ADD THAT AS A TUNABLE PARAMETER

# Data Preprocessing:

def prepare_data(data, max_seq_length: int = MAX_SEQ_LENGTH):
    all_input_ids = []
    all_labels = []

    pad_token_id = tokenizer.pad_token_id
    
    # Tokenize all data at once for efficiency
    tokenized_data = tokenizer(
        data,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=False  # We'll handle special tokens manually
    )
    
    # Get the token ID for </prompt>
    end_prompt_token_id = tokenizer.encode("</prompt>", add_special_tokens=False)[0]
    
    # Process each sample
    for sample_tokens in tokenized_data['input_ids']:
        # Find the index of </prompt> token
        try:
            end_prompt_index = sample_tokens.index(end_prompt_token_id)
        except ValueError:
            # If </prompt> not found, treat sample as a non-instruct sample
            end_prompt_index = int(np.ceil(len(sample_tokens) * (1/3)))  # 0 ## 1. Give it a fair starting place to predict the next word 2. reduce the number of expanded samples 
            
        # Find first pad token after </prompt>
        first_pad_index = None
        for i in range(end_prompt_index + 1, len(sample_tokens)):
            if sample_tokens[i] == pad_token_id:
                first_pad_index = i
                break
        
        # If no pad token found, use the end of sequence
        if first_pad_index is None:
            first_pad_index = len(sample_tokens)
        
        # Apply sliding window from after </prompt> to first pad token
        # Start from end_prompt_index + 1 (first token to predict)
        # End at first_pad_index - 1 (last token to predict)
        for i in range(end_prompt_index + 1, first_pad_index):
            # Input: from start up to (but not including) token i
            input_ids = sample_tokens[:i]
            
            # Pad or truncate to max_seq_length
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]
            else:
                input_ids = input_ids + [pad_token_id] * (max_seq_length - len(input_ids))
            
            # Label: one-hot encoding of token at position i
            next_token = sample_tokens[i]
            label = [0] * VOCABULARY_SIZE
            label[next_token] = 1
            
            all_input_ids.append(input_ids)
            all_labels.append(label)
        
        # Add final sample with pad token as label to indicate termination
        if first_pad_index < len(sample_tokens):  # Only if there's actually a pad token
            input_ids = sample_tokens[:first_pad_index]
            
            # Pad or truncate to max_seq_length
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]
            else:
                input_ids = input_ids + [pad_token_id] * (max_seq_length - len(input_ids))
            
            # Label: one-hot encoding of pad token
            label = [0] * VOCABULARY_SIZE
            label[pad_token_id] = 1
            
            all_input_ids.append(input_ids)
            all_labels.append(label)
    
    return all_input_ids, all_labels, VOCABULARY_SIZE


## Only add re, tokenizer already in script



from transformers import AutoTokenizer

tokenizer_checkpoint = "HuggingFaceTB/SmolLM3-3B" # "HuggingFaceTB/SmolLM2-1.7B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)


with open('king-james-bible.txt', 'r') as kjv:
    bible = kjv.read()


def package_non_instruct_text(text: str, desired_samples: int, max_length_tokens: int) -> list[str]:
    """
    Package a block of text into samples of approximately max_length_tokens.
    
    Args:
        text: Block of text to process (e.g., entire book)
        desired_samples: Number of samples to generate
        max_length_tokens: Maximum number of tokens per sample
        
    Returns:
        List of text samples, each approximately max_length_tokens long
    """
    # Split text into sentences using regex to handle various sentence endings
    sentences = re.split(r'[.!?]+', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    samples = []
    current_sample_sentences = []
    current_token_count = 0
    
    sentence_index = 0
    
    while len(samples) < desired_samples and sentence_index < len(sentences):
        sentence = sentences[sentence_index]
        
        # Estimate token count for this sentence
        sentence_tokens = len(tokenizer.encode(sentence))
        
        # Check if adding this sentence would exceed the token limit
        if current_token_count + sentence_tokens <= max_length_tokens:
            current_sample_sentences.append(sentence)
            current_token_count += sentence_tokens
            sentence_index += 1
        else:
            # If we have accumulated sentences, create a sample
            if current_sample_sentences:
                sample = " ".join(current_sample_sentences)
                samples.append(sample)
            
            # Reset for next sample
            current_sample_sentences = []
            current_token_count = 0
            
            # If this single sentence is too long, skip it
            if sentence_tokens > max_length_tokens:
                sentence_index += 1
    
    # Add the final sample if we have any remaining sentences
    if current_sample_sentences and len(samples) < desired_samples:
        sample = " ".join(current_sample_sentences)
        samples.append(sample)
    
    return samples

# Separate into samples
non_instruct_samples = package_non_instruct_text(text=bible, desired_samples=30, max_length_tokens=int(np.ceil(MAX_SEQ_LENGTH * .8))) ##

del(bible)
collect()

print(f"Samples from KJV bible consisting of {len(non_instruct_samples)} look like this (sub-sample of 3): {non_instruct_samples[:3]}") 


# Replace with imported text

data = [
    # Sample 1: Wiki-style summary (no thinking/code)
    "<prompt>What was the significance of the Magna Carta?</prompt><response>The Magna Carta, signed in 1215 by King John of England, was a foundational document in the development of constitutional law. It established the principle that no one, not even the king, is above the law. Key provisions included the right to a fair trial and protection from unlawful imprisonment. Though initially annulled, later reissues influenced democratic governance in England and inspired constitutional documents like the U.S. Constitution.</response>",

    # Sample 2: Short story (with Thinking)
    "<prompt>Tell me a short story about a robot learning to paint.</prompt><tool_call>Thinking about how a machine might interpret creativity as a form of logic and emotion. Perhaps through iterative learning and sensory input.<response>A robot named Luma stood in a sunlit studio, its optical sensors adjusting to the spectrum of morning light. Programmed for precision, it had never understood why humans smeared color on canvas. But after analyzing 10,000 paintings, it noticed patterns—not just in brushstrokes, but in the silence between them. One day, it mixed cadmium yellow with a dash of uncertainty and painted a sun that didn’t shine, but wept. The gallery called it 'The First Honest Machine.' People cried. Luma didn’t. But it saved the tear-stained feedback form forever.</response>",

    # Sample 3: Bible-style verse (no thinking/code)
    "<prompt>Write a verse in the style of Ecclesiastes about modern work life.</prompt><response>There is a season for every email under heaven: a time to CC, and a time to BCC; a time to schedule meetings, and a time to mute them. A person labors beneath the weight of notifications, yet gains little. See, I have seen all things under Wi-Fi to be fleeting. Better is quiet focus than the striving after unread messages.</response>",

    # Sample 4: Business writing (with code)
    "<prompt>How can I automate monthly sales reports in Python?</prompt><response><tool_call>You can use pandas to aggregate sales data and matplotlib to generate charts. Here's a basic template:<code>import pandas as pd\nimport matplotlib.pyplot as plt\n\n# Load data\ndf = pd.read_csv('sales.csv')\ndf['date'] = pd.to_datetime(df['date'])\n\ndf.set_index('date', inplace=True)\nmonthly_sales = df.resample('M').sum()\n\n# Plot\nmonthly_sales.plot(kind='bar')\nplt.title('Monthly Sales Report')\nplt.savefig('monthly_report.png')</code>\n\nYou can schedule this with cron or Task Scheduler.</response>",

    # Sample 5: Synthetic social blog (no thinking/code)
    "<prompt>Write a blog post about finding joy in small routines.</prompt><response>Mornings used to be chaos—coffee spilled, keys lost, emails screaming. Then I started the 'three-minute rule': three minutes of stillness, three sips of tea, three deep breaths. No phone. No agenda. Just the kettle’s whistle and the light on the floor. It didn’t fix everything. But it taught me that peace isn’t found in grand escapes. It’s in the ritual. In the pause before the world begins.</response>",

    # Sample 6: SMS-style conversation (with Thinking)
    "<prompt>Write a realistic SMS exchange about rescheduling a meeting.</prompt><tool_call>Modeling concise, informal tone with time-sensitive context.<response>Sarah: Hey, can we push mtg to Thurs? Kid’s sick 😣\nJamal: No prob. 10am still good?\nSarah: 11 if possible?\nJamal: Works. I’ll update the calendar.\nSarah: Thanks! Soup and cartoons today 🍲📺</response>",

    # Sample 7: Textbook-style physics summary (no thinking/code)
    "<prompt>Explain Newton’s laws of motion in simple terms.</prompt><response>Newton’s First Law: An object at rest stays at rest, and an object in motion stays in motion unless acted on by a force. (Inertia)\n\nNewton’s Second Law: Force equals mass times acceleration (F = ma). The harder you push, the faster it goes—especially if it’s light.\n\nNewton’s Third Law: For every action, there’s an equal and opposite reaction. If you push a wall, the wall pushes back—though you won’t see it move.</response>",

    # Sample 8: FAQ-style (with code)
    "<prompt>How do I reverse a string in Python?</prompt><response><tool_call>This is a common beginner question. The simplest way is slicing.<code># Reverse a string\ns = 'hello'\nreversed_s = s[::-1]\nprint(reversed_s)  # 'olleh'</code>\n\nAlternatively, use reversed() with join: ''.join(reversed(s)).</response>",

    # Sample 9: Inspirational short poem (no thinking/code)
    "<prompt>Write a short poem about perseverance.</prompt><response>The path was steep, the sky turned gray.\nEach step felt like yesterday.\nBut still I climbed, though slow and worn,\nFor every dusk, a dawn is born.\nNot fast, not loud, but never still—\nPersistence climbs the steepest hill.</response>",

    # Sample 10: Business proposal snippet (with Thinking and code)
    "<prompt>Generate a proposal outline for a green energy startup.</prompt><tool_call>Structuring a professional document with clear sections. Can use Markdown for formatting.<code># Project Helios: Clean Energy for Urban Homes\n\n## Overview\nAffordable solar integration for apartment dwellers via shared rooftop grids.\n\n## Goals\n- Reduce urban carbon footprint by 15% in 5 years\n- Achieve 10,000 subscribers by 2027\n- Partner with city housing authorities\n\n## Tech Stack\n- IoT energy monitors (Python backend)\n- Customer portal (React)\n- Data analytics (Pandas, Tableau)</code>\n\nWe’ll pilot in Portland and scale to 10 cities by 2026.</response>",


    # Sample 11: Textbook-style biology summary (no thinking/code)
    "<prompt>Explain photosynthesis in simple terms.</prompt><response>Photosynthesis is the process plants use to convert sunlight into energy. Using chlorophyll in their leaves, plants absorb sunlight and use it to turn carbon dioxide (from the air) and water (from the soil) into glucose (a type of sugar) and oxygen. The chemical equation is: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂. This process not only feeds the plant but also produces the oxygen animals and humans need to breathe.</response>",

    # Sample 12: Short story with Thinking
    "<prompt>Tell a story about a librarian who discovers a book that writes itself.</prompt><tool_call>Considering how a self-writing book might reflect the reader's inner thoughts or predict the future. Tone should be whimsical but grounded.<response>Every morning, Ms. Elara Reed dusted the shelves of the Cedar Hollow Library. One rainy Tuesday, she found a leather-bound book with no title. When she opened it, the first page read: 'Today, you will forget your lunch, but a child will share his apple.' She laughed—until it happened. The next day: 'You will receive a letter from someone long gone.' That afternoon, a 20-year-old postcard from her late sister arrived. The book didn’t predict disasters or fortunes. It wrote small truths, quietly. After a month, a new line appeared: 'Now, it’s your turn.' Elara picked up a pen. The next page filled with her handwriting—words she didn’t plan. And so, the book continued.</response>",

    # Sample 13: Bible-style verse (no thinking/code)
    "<prompt>Write a verse about kindness in the age of technology.</prompt><response>Be kind, even when the screen dims your voice. Let your words be bridges, not barricades. If your neighbor stumbles in silence, send a message that says, 'I see you.' Do not reply in haste, for anger spreads faster than fire. But let patience grow in the inbox of your heart. A single 'thank you' can light a room no algorithm can design.</response>",

    # Sample 14: Technical FAQ with code
    "<prompt>How do I check if a number is prime in Python?</prompt><response><tool_call>We can write a function that checks divisibility up to the square root for efficiency.<code>def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\n# Example usage\nprint(is_prime(17))  # True\nprint(is_prime(25))  # False</code>\nThis avoids unnecessary checks and runs in O(√n) time.</response>",

    # Sample 15: Synthetic social blog (no thinking/code)
    "<prompt>Write a personal blog post about learning to cook during the pandemic.</prompt><response>March 2020: I burned toast. Not slightly charred—*full smoke alarm, windows flung open, cat glaring* burned. With nowhere to go, I decided to learn cooking from scratch. Week one: scrambled eggs (soupy). Week two: pasta (overcooked). But by June, I made a lasagna that didn’t collapse. It wasn’t just about food. It was about control. About creating something warm in a cold, uncertain world. Now, my kitchen smells like rosemary and resilience. And the toast? Perfectly golden.</response>",

    # Sample 16: Business email (with Thinking)
    "<prompt>Write a polite email declining a job offer.</prompt><tool_call>Balancing professionalism and gratitude while maintaining clear boundaries.<response>Subject: Thank You for the Opportunity – Job Offer Decision\n\nDear Mr. Thompson,\n\nThank you so much for offering me the position of Senior Project Coordinator at NovaEdge Solutions. I truly appreciate the time you and the team spent with me during the interview process.\n\nAfter careful consideration, I’ve decided to pursue a path that aligns more closely with my current goals. This was not an easy decision, as I was deeply impressed by your mission and culture.\n\nI hope our paths cross again in the future, and I wish you continued success.\n\nWarm regards,\nLena Patel</response>",

    # Sample 17: Poetry (no thinking/code)
    "<prompt>Write a haiku about autumn leaves.</prompt><response>Crimson leaves descend,\nWhispering to the waiting earth—\nSummer lets go now.</response>",

    # Sample 18: SMS exchange with code (Thinking + code)
    "<prompt>Show an SMS thread where someone helps debug a loop.</prompt><tool_call>Modeling real-time peer support with lightweight code sharing.<code>def countdown(n):\n    while n > 0:\n        print(n)\n        n -= 1\n    print('Liftoff!')</code>\n\nAlex: This loop works but prints negative numbers if I call countdown(-3)\nSam: Add a condition at the start!\nAlex: Like… if n <= 0: return?\nSam: Exactly. Or use 'if n < 1: break' inside. Better safe than infinite 😄</response>",

    # Sample 19: Historical summary (no thinking/code)
    "<prompt>Summarize the fall of the Western Roman Empire.</prompt><response>The Western Roman Empire fell in 476 CE when the Germanic chieftain Odoacer deposed Emperor Romulus Augustulus. This collapse was not sudden but the result of centuries of decline—economic instability, overreliance on slave labor, political corruption, and military defeats. Frequent invasions by Visigoths, Vandals, and Huns weakened borders. Meanwhile, the Eastern Roman (Byzantine) Empire survived for nearly 1,000 more years. The fall marked the beginning of the European Middle Ages and a shift from centralized imperial rule to feudal kingdoms.</response>",

    # Sample 20: Instructional guide with Thinking
    "<prompt>How do I write a clear mission statement for a nonprofit?</prompt><tool_call>A mission statement should be concise, purpose-driven, and audience-focused. Use active verbs and avoid jargon.<response>Your mission statement should answer: Who do you serve? What do you do? Why does it matter?\n\nExample structure: '[Organization] empowers [target group] to [action] through [method], creating [impact].'\n\nFor a literacy nonprofit: 'ReadTogether empowers underserved children to become confident readers through free tutoring and book access, creating a future where every child can learn.'\n\nKeep it under 25 words. Test it: Does it inspire? Is it clear to a 12-year-old? If yes, you’ve got it.</response>"

    # No prompt (simple text completion)

    """In the beginning God created the heaven and the earth. And the earth was without form, and void;\nand darkness was upon the face of the deep. And\nthe Spirit of God moved upon the face of the waters.\nAnd God said, Let there be light: and there was light. And God saw the\nlight, that it was good: and God divided the light from the darkness. And\nGod called the light Day, and the darkness he called Night. And the evening\nand the morning were the ﬁrst day.\nAnd God said, Let there be a ﬁrmament in the midst of the waters, and\nlet it divide the waters from the waters. And God made the ﬁrmament, and let it divide the waters from the waters. And God made the ﬁrmament, and\ndivided the waters which were under the ﬁrmament from the waters which\nwere above the ﬁrmament: and it was so. And God called the ﬁrmament\nHeaven. And the evening and the morning were the second day."""
]

# Add non-instruct samples
# data += non_instruct_samples


x, y, vocab_size = prepare_data(non_instruct_samples) # data)

print("Input IDs shape:", len(x), "x", len(x[0]) if x else 0)
print("Labels shape:", len(y), "x", len(y[0]) if y else 0)
print("Vocabulary size:", vocab_size)
print("First few samples generated:", len(x))


# i = 1
# for d,l in zip(x, y):
#     print(f"Sample {i}:")
#     print(d)
#     print(f"label {i}: (index of)")
#     print(l.index(1))
#     i += 1

#  ... apply sliding window over the range over the first token to generate and the first pad_token ...


X_train, X_test, y_train, y_test = \
train_test_split(x, y, test_size=0.85, shuffle=False)

INPUT_SHAPES = [(MAX_SEQ_LENGTH,)]
OUTPUT_SHAPES = [(VOCABULARY_SIZE)]

x_train_tf = tf.constant(X_train, tf.int32)
y_train_tf = tf.constant(y_train, tf.float32)

x_train_packaged = [x_train_tf]
y_train_packaged = [y_train_tf]

x_test_tf = tf.constant(X_test, tf.int32)
y_test_tf = tf.constant(y_test, tf.float32)

x_test_packaged = [x_test_tf] 
y_test_packaged = [y_test_tf]

### Change loss to crossentropy and keep the metric as accuracy, tweak params, and the rest should be the same ... 

# --- Base Rotary Positional Embedding
@tf.keras.utils.register_keras_serializable()
class RotaryEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, max_seq_len=1024, temperature=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        # Ensure dim is even right at initialization
        if self.dim % 2 != 0:
            raise ValueError(f"Embedding dimension `dim` ({self.dim}) must be even for RotaryEmbedding.")
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        # *** No calculation or storage of inv_freq here or in build ***

    def build(self, input_shape):
        # Build should primarily be for creating trainable weights, which we don't have.
        # Call super().build() for Keras compatibility.
        super().build(input_shape)

    def call(self, x): # Removed seq_len argument, calculate from x
        shape = tf.shape(x)
        batch_size = shape[0]
        actual_seq_len = shape[1]

        # *** Calculate inv_freq inside call ***
        inv_freq_base = tf.range(0, self.dim, 2, dtype=tf.float32)
        inv_freq = 1.0 / (self.temperature ** (inv_freq_base / self.dim))
        # Ensure inv_freq has the correct shape [dim/2]
        inv_freq = tf.cast(inv_freq, dtype=x.dtype) # Match dtype early

        # Use actual_seq_len for calculations
        position = tf.range(actual_seq_len, dtype=x.dtype) # Match dtype

        # Calculate sinusoid input using einsum or broadcasting
        # Einsum approach: Ensure correct dimensions [seq_len, dim/2]
        sinusoid_inp = tf.einsum("i,j->ij", position, inv_freq)

        # Calculate sin and cos based on the actual sequence length
        sin = tf.sin(sinusoid_inp)
        cos = tf.cos(sinusoid_inp)

        # Repeat sin/cos for interleaving: [a, b] -> [a, a, b, b]
        # Result needs shape [actual_seq_len, dim]
        sin = tf.repeat(sin, 2, axis=-1)
        cos = tf.repeat(cos, 2, axis=-1)

        # Expand dims for batch and tile
        # Output shape needs to be [batch_size, actual_seq_len, dim]
        # Add batch dimension: [1, actual_seq_len, dim]
        sin = tf.expand_dims(sin, axis=0)
        cos = tf.expand_dims(cos, axis=0)

        # Tile to match the batch size: [batch_size, actual_seq_len, dim]
        sin = tf.tile(sin, [batch_size, 1, 1])
        cos = tf.tile(cos, [batch_size, 1, 1])

        # Casting to x.dtype was already done for inv_freq, sin/cos will inherit
        # sin = tf.cast(sin, x.dtype) # Already done via calculation chain
        # cos = tf.cast(cos, x.dtype) # Already done via calculation chain

        # Return sin and cos needed by InterleavedRoPE
        return sin, cos

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "max_seq_len": self.max_seq_len,
            "temperature": self.temperature,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



# iRoPE helper functions

@tf.keras.utils.register_keras_serializable()
def split_alternate(x):
    shape = tf.shape(x)
    x = tf.reshape(x, [shape[0], shape[1], shape[2] // 2, 2])
    x = tf.transpose(x, [0, 1, 3, 2])
    x = tf.reshape(x, [shape[0], shape[1], -1])
    return x


@tf.keras.utils.register_keras_serializable()
def rotate_half(x):
    x = split_alternate(x)
    d = tf.shape(x)[-1]
    rotated_x = tf.concat([-x[..., d//2:], x[..., :d//2]], axis=-1)
    return tf.reshape(rotated_x, tf.shape(x))


@tf.keras.utils.register_keras_serializable()
def apply_rotary_pos_emb(x, sin, cos):
    cos = tf.reshape(cos, [tf.shape(cos)[0], tf.shape(cos)[1], -1])
    sin = tf.reshape(sin, [tf.shape(sin)[0], tf.shape(sin)[1], -1])
    x_rotated = x * cos + rotate_half(x) * sin
    return x_rotated

# interleaved Rotary Postional Embedding (iRoPE)
@tf.keras.utils.register_keras_serializable()
class InterleavedRoPE(tf.keras.layers.Layer):
    def __init__(self, dim, max_seq_len=1024, **kwargs):
        super().__init__(**kwargs)
        if dim % 2 != 0:
             raise ValueError(f"Embedding dimension `dim` ({dim}) must be even for InterleavedRoPE.")
        self.dim = dim
        self.max_seq_len = max_seq_len
        # Instantiate the RotaryEmbedding layer
        # Ensure the name is consistent if needed for saving/loading
        self.rotary_emb = RotaryEmbedding(dim, max_seq_len, name="rotary_embedding")

    def call(self, x):
        # Get sin and cos from the RotaryEmbedding layer's call method
        # *** Pass only 'x'. RotaryEmbedding calculates seq_len internally. ***
        sin, cos = self.rotary_emb(x)

        # Apply the positional embeddings
        x_embedded = apply_rotary_pos_emb(x, sin, cos)
        return x_embedded

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "max_seq_len": self.max_seq_len,
        })
        # Keras handles nested layer serialization automatically
        return config

    @classmethod
    def from_config(cls, config):
        # Keras handles nested layer restoration automatically
        return cls(**config)

# Text embedding base model

inp = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32)

embedded = tf.keras.layers.Embedding(
    input_dim=VOCABULARY_SIZE,
    output_dim=EMBEDDING_DIM,
    input_length=MAX_SEQ_LENGTH,
    mask_zero=False)(inp)

position_embedding = InterleavedRoPE(
    dim=EMBEDDING_DIM,
    max_seq_len=MAX_SEQ_LENGTH,
    # initializer="uniform",
)(embedded)

# As an FYI, we tried an add layer both with and without
# LayerNorm ... It degraded accuracy
# Just an FYI for anyone trying to apply conventional wisdom
# to save you the time ...
x = tf.keras.layers.Concatenate()([embedded, position_embedding])
x = tf.keras.layers.Dropout(POSITIONAL_EMBEDDING_DROPOUT)(x)  # AI suggested 0.4 
flattened = tf.keras.layers.Flatten()(x)
projected = tf.keras.layers.Dense(EMBEDDING_DIM * PROJECTION_N)(flattened) # Dimensionality reduction

cerebros_base_model = tf.keras.Model(
    inputs=inp,
    outputs=projected  # Output enhanced embeddings now
)


## Cerebros 

#
# Logging
#
TIME = pendulum.now(tz='America/New_York').__str__()[:16]\
    .replace('T', '_')\
    .replace(':', '_')\
    .replace('-', '_')
PROJECT_NAME = f'{TIME}_cerebros_auto_ml_phishing_email_test'

meta_trial_number = 42 # irrelevant unless in distributed training

# Custom metric: Perplexity:

@tf.keras.utils.register_keras_serializable()
class Perplexity(tf.keras.metrics.Metric):
    """
    Computes perplexity, defined as e^(categorical crossentropy).
    """
    def __init__(self, name='perplexity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_crossentropy = self.add_weight(name='total_crossentropy', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate categorical crossentropy
        crossentropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # Update the running sum of crossentropy and the count of samples
        self.total_crossentropy.assign_add(tf.reduce_sum(crossentropy))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], dtype=tf.float32))

    def result(self):
        # Compute the average crossentropy
        average_crossentropy = self.total_crossentropy / self.count
        # Compute perplexity as e^(average crossentropy)
        return tf.exp(average_crossentropy)

    def reset_state(self):
        # Reset the state variables
        self.total_crossentropy.assign(0.0)
        self.count.assign(0.0)

perplexity_metric = Perplexity()

cerebros_automl = SimpleCerebrosRandomSearch(
    unit_type=DenseUnit,
    input_shapes=INPUT_SHAPES,
    output_shapes=OUTPUT_SHAPES,
    training_data=x_train_packaged,
    labels=y_train_packaged,
    validation_split=0.2,
    direction='minimize',
    metric_to_rank_by="perplexity", # REPLACE WITH VAL PERPLEXITY ONCE THIS IS STABLE
    minimum_levels=minimum_levels,
    maximum_levels=maximum_levels,
    minimum_units_per_level=minimum_units_per_level,
    maximum_units_per_level=maximum_units_per_level,
    minimum_neurons_per_unit=minimum_neurons_per_unit,
    maximum_neurons_per_unit=maximum_neurons_per_unit,
    activation=activation,
    final_activation='softmax',
    number_of_architecture_moities_to_try=moities_to_try,
    number_of_tries_per_architecture_moity=tries_per_moity,
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
    metrics=[tf.keras.metrics.CategoricalAccuracy(),
             perplexity_metric,
        # tf.keras.metrics.Accuracy()
            ],
    epochs=epochs,
    project_name=f"{PROJECT_NAME}_meta_{meta_trial_number}",
    model_graphs='model_graphs',
    batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    meta_trial_number=meta_trial_number,
    base_models=[cerebros_base_model],
    train_data_dtype=tf.int32)  # Changed from tf.string to tf.int32

cerebros_t0 = time.time()
result = cerebros_automl.run_random_search()
cerebros_t1 = time.time()
cerebros_time_all_models_min = (cerebros_t1 - cerebros_t0) / 60
models_tried = moities_to_try  * tries_per_moity
cerebros_time_per_model = cerebros_time_all_models_min / models_tried



print(f"Cerebros trained {models_tried} models FROM A COLD START in ONLY {cerebros_time_all_models_min} min. Cerebros took only {cerebros_time_per_model} minutes on average per model.")
""" ADD BACK


print(f"GPT2 took {gpt_time_on_one_model_min} just to FINE TUNE one PRE - TRAINED model for 3 epochs. Although this is a small scale test, this shows the advantage of scaling in ON timing VS ON**2 timing.")

"""

print(f'Cerebros best accuracy achieved is {result}')
print(f'val set accuracy')

"""### Testing the best model found"""

MODEL_FILE_NAME = "cerebros-foundation-model.keras"

best_model_found = cerebros_automl.get_best_model()
best_model_found.save(MODEL_FILE_NAME)
# del(best_model_found)
# del(cerebros_automl)
# collect()

file_size_bytes = getsize(MODEL_FILE_NAME)
print(f"Model size on disk: {file_size_bytes / (1024*1024):.2f} MB")

reconstituted_model = tf.keras.models.load_model(MODEL_FILE_NAME)

# Generate text from test samples
print("\n" + "="*50)
print("GENERATED TEXT SAMPLES")
print("="*50)

# Get pad token id
pad_token_id = tokenizer.pad_token_id
end_prompt_token_id = tokenizer.encode("</prompt>", add_special_tokens=False)[0]

# # Generate text for first 5 test samples (Working)
# generated_texts = []
# for i in range(min(5, len(x_test_packaged[0]))):
#     original_input = x_test_packaged[0][i].numpy()
    
#     # Find the end of the prompt
#     try:
#         end_prompt_index = list(original_input).index(end_prompt_token_id)
#     except ValueError:
#         end_prompt_index = 0
    
#     # Extract the prompt part
#     prompt_tokens = original_input[:end_prompt_index+1].tolist()
    
#     # Generate tokens sequentially
#     generated_tokens = []
#     current_input = prompt_tokens.copy()
    
#     # Generate up to 100 tokens or until pad token
#     for _ in range(100):
#         # Pad or truncate to MAX_SEQ_LENGTH
#         input_tensor = tf.constant([current_input + [pad_token_id] * (MAX_SEQ_LENGTH - len(current_input))], dtype=tf.int32)
        
#         # Get prediction
#         prediction = reconstituted_model(input_tensor)
#         next_token_id = int(tf.argmax(prediction[0], axis=-1).numpy())
        
#         # Stop if pad token generated
#         if next_token_id == pad_token_id:
#             break
            
#         generated_tokens.append(next_token_id)
#         current_input.append(next_token_id)
        
#         # Stop if we exceed max length
#         if len(current_input) >= MAX_SEQ_LENGTH:
#             break
    
#     generated_texts.append((prompt_tokens, generated_tokens))

# # Decode and print with proper formatting
# for idx, (prompt_tokens, generated_tokens) in enumerate(generated_texts):
#     # Decode prompt
#     prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
    
#     # Extract original prompt content
#     if '<prompt>' in prompt_text and '</prompt>' in prompt_text:
#         original_prompt = prompt_text.split('<prompt>')[-1].split('</prompt>')[0]
#     else:
#         original_prompt = prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text
    
#     # Decode generated text
#     generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False) if generated_tokens else ""
    
#     print(f"\nGenerated text from sample {idx+1}:")
#     print(f"<prompt>{original_prompt}</prompt>{generated_text}")



## Proper model wrapper and generation method (under development):

print("###### Output of the model wrapper (under development) ########### ")

# Register the config and model wrapper as serializable
@tf.keras.utils.register_keras_serializable()
class CerebrosAutoregressiveTextGeneratorConfig:
    def __init__(self, max_sequence_length=1536, padding_token=None):
        self.max_sequence_length = max_sequence_length
        self.padding_token = padding_token
    
    def get_config(self):
        return {
            'max_sequence_length': self.max_sequence_length,
            'padding_token': self.padding_token
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class CerebrosAutoregressiveTextGenerator(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.max_sequence_length = config.max_sequence_length
        self.padding_token = config.padding_token
        # Make self.model = the reconstituted model (constant)
        self.model = best_model_found # reconstituted_model
    
    def get_config(self):
        return {
            'config': self.config.get_config()
        }
    
    @classmethod
    def from_config(cls, config):
        config_obj = CerebrosAutoregressiveTextGeneratorConfig.from_config(config['config'])
        return cls(config=config_obj)
    
    def generate(self, token_ids, do_sample=False, max_new_tokens=None):
        """
        Generate text autoregressively from token IDs.
        
        Args:
            token_ids: Iterable of integers representing token IDs
            do_sample: Boolean, if True use sampling, if False use greedy argmax
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            List of token IDs including original tokens and generated tokens
        """
        # Convert token_ids to list if it's not already
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)
            
        # Determine the actual maximum number of new tokens
        if max_new_tokens is None:
            max_new_tokens = self.max_sequence_length - len(token_ids)
        else:
            max_new_tokens = min(max_new_tokens, self.max_sequence_length - len(token_ids))
            
        # Initialize the generated tokens list
        generated_tokens = []
        current_tokens = token_ids.copy()
        
        # Autoregressive generation loop
        temp_gen_count = 0 # <--------<< Debug code to remove later
        for _ in range(max_new_tokens):
            # Pad or truncate to max_sequence_length (CORRECTED PADDING LOGIC)
            if len(current_tokens) > self.max_sequence_length:
                input_tokens = current_tokens[:self.max_sequence_length]
            else:
                # Manual padding with padding token
                padding_needed = self.max_sequence_length - len(current_tokens)
                input_tokens = current_tokens + [self.padding_token] * padding_needed
            
            # Convert to tensor and get model prediction
            input_tensor = tf.constant([input_tokens], dtype=tf.int32)
            logits = self.model(input_tensor)  # Shape: (batch_size, VOCABULARY_SIZE)
            
            # Get next token based on sampling strategy
            if do_sample:
                # Sample from the distribution
                # probabilities = tf.nn.softmax(logits[0], axis=-1) # Model already applies softmax
                next_token_id = tf.random.categorical(tf.math.log(logits[0])[None, :], 1)[0, 0].numpy()
            else:
                # Greedy sampling (argmax)
                next_token_id = int(tf.argmax(logits[0], axis=-1).numpy())
            # Debug code to removel later
            print(f"Generating {temp_gen_count}")
            print(f"... next_token_id: {next_token_id}")
            next_word = tokenizer.decode(next_token_id)
            print(f"Next decoded word: {next_word}")
            temp_gen_count +=1

            # Check for termination condition
            if next_token_id == self.padding_token:
                break
                
            # Add to generated tokens and update current tokens
            generated_tokens.append(int(next_token_id))
            current_tokens.append(int(next_token_id))
            
            # Check if we've reached max sequence length
            if len(current_tokens) >= self.max_sequence_length:
                break
        
        # Pad the result to the required length if necessary (CORRECTED PADDING LOGIC)
        total_tokens = token_ids + generated_tokens
        if len(total_tokens) < len(token_ids) + max_new_tokens:
            padding_needed = len(token_ids) + max_new_tokens - len(total_tokens)
            total_tokens.extend([self.padding_token] * padding_needed)
            
        return total_tokens

    def call(self, inputs):
        # This is just for compatibility, the main logic is in generate()
        return self.model(inputs)

# Replace the generation code block with this:

print("\n" + "="*50)
print("GENERATED TEXT SAMPLES USING WRAPPER")
print("="*50)

# Create config and generator
config = CerebrosAutoregressiveTextGeneratorConfig(
    max_sequence_length=MAX_SEQ_LENGTH,
    padding_token=tokenizer.pad_token_id
)
generator = CerebrosAutoregressiveTextGenerator(config)

print("########### BEFORE SEARIALIZING THE GENERATIVE MODEL")

counter = 0
for sample in non_instruct_samples:
    half_sample_len = int(np.ceil(len(sample) / 2))
    half_sample = sample[:half_sample_len]
    
    # Tokenize the text
    half_sample_tokenized = tokenizer(
        half_sample,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=False
    )['input_ids']
    
    # # Extract token IDs as a list of integers (not tensors)
    # if isinstance(half_sample_tokenized, dict):
    #     # If tokenizer returns a dict, extract the token IDs
    #     token_ids = half_sample_tokenized['input_ids']  # or 'token_ids' depending on your tokenizer
    # else:
    #     # If tokenizer returns a list directly
    #     token_ids = half_sample_tokenized
    
    # # Convert to Python list of integers if it's a tensor
    # if hasattr(token_ids, 'numpy'):
    #     token_ids = token_ids.numpy().tolist()
    # if not isinstance(token_ids, list):
    #     token_ids = list(token_ids)
    
    # Now pass the list of integers to your generate method
    generated_tokens = generator.generate(
        token_ids=token_ids,  # This should now be a list of integers
        do_sample=False,
        max_new_tokens=40
    )
    
    # Decode the result
    full_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    print(f"PROMPT number {counter}: {half_sample}; RESPONSE: {full_generated_text}")
    counter += 1



# # Process ALL original samples from data - REAL WORLD USAGE
# generated_texts = []
# for i, original_text in enumerate(data[:5]):  # Process first 5 samples
#     print(f"\nProcessing sample {i+1}...")
    
#     # Extract prompt part (everything up to and including </prompt>)
#     if '</prompt>' in original_text:
#         prompt_part = original_text.split('</prompt>')[0] + '</prompt>'
#     else:
#         prompt_part = original_text
    
#     # Tokenize the prompt part
#     tokenized = tokenizer(
#         prompt_part,
#         add_special_tokens=False,  # We handle special tokens manually
#         return_tensors=None  # Return lists, not tensors
#     )
#     prompt_tokens = tokenized['input_ids']
    
#     print(f"Original prompt: {prompt_part[:100]}...")
#     print(f"Tokenized prompt length: {len(prompt_tokens)} tokens")
    
#     # Generate tokens using the wrapper class - REAL WORLD USAGE
#     generated_tokens = generator.generate(
#         token_ids=prompt_tokens,
#         do_sample=False,
#         max_new_tokens=100
#     )
    
#     # Decode the full generated text
#     full_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    
#     # Extract just the newly generated part (after the prompt)
#     generated_part = full_generated_text[len(prompt_part):]
    
#     generated_texts.append((prompt_part, generated_part))
    
#     print(f"Generated response: {generated_part}...")

# # Display results with proper formatting
# print("\n" + "="*50)
# print("FINAL GENERATED RESULTS")
# print("="*50)

# for idx, (original_prompt, generated_response) in enumerate(generated_texts):
#     print(f"\nSample {idx+1}:")
#     print(f"Prompt:{original_prompt}")
#     print(f"Response: {generated_response}")

# Save the model
model_save_path = f"{TIME}_cerebros-autoregressive-text-generator.keras"
generator.save(model_save_path)
print(f"\nModel saved as '{model_save_path}'")

# Garbage collect
del generator
collect()
print("Garbage collection completed.")

# Reconstitute the model
print("Reconstituting model...")
reconstituted_generator = tf.keras.models.load_model(model_save_path)
print("Model reconstituted successfully!")

counter = 0
for sample in non_instruct_samples:
    half_sample_len = int(np.ceil(len(sample) / 2))
    half_sample = sample[:half_sample_len]
    
    # Tokenize the text
    half_sample_tokenized = tokenizer(
        half_sample,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=False
    )['input_ids']
    
    # # Extract token IDs as a list of integers (not tensors)
    # if isinstance(half_sample_tokenized, dict):
    #     # If tokenizer returns a dict, extract the token IDs
    #     token_ids = half_sample_tokenized['input_ids']  # or 'token_ids' depending on your tokenizer
    # else:
    #     # If tokenizer returns a list directly
    #     token_ids = half_sample_tokenized
    
    # Convert to Python list of integers if it's a tensor
    if hasattr(token_ids, 'numpy'):
        token_ids = token_ids.numpy().tolist()
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)
    
    # Now pass the list of integers to your generate method
    generated_tokens = reconstituted_generator.generate(
        token_ids=token_ids,  # This should now be a list of integers
        do_sample=False,
        max_new_tokens=40
    )
    
    # Decode the result
    full_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    print(f"PROMPT number {counter}: {half_sample}; RESPONSE: {full_generated_text}")
    counter += 1

# # Test with all original data samples - REAL WORLD DEMO (reconstituted)
# print("\n" + "="*50)
# print("GENERATED TEXT SAMPLES FROM ALL DATA - REAL WORLD USAGE (reconstituted)")
# print("="*50)

# generated_texts_all = []
# for i, text in enumerate(data):
#     # Extract prompt part (everything up to and including </prompt>)
#     if '</prompt>' in text:
#         prompt_text = text.split('</prompt>')[0] + '</prompt>'
#     else:
#         prompt_text = text
    
#     # Tokenize the prompt part for model input
#     tokenized = tokenizer(
#         prompt_text,
#         max_length=MAX_SEQ_LENGTH,
#         padding='max_length',
#         truncation=True,
#         add_special_tokens=False
#     )
#     token_ids = tokenized['input_ids']
    
#     # Generate using the reconstituted model
#     generated_token_ids = reconstituted_generator.generate(
#         token_ids=token_ids,
#         do_sample=False,
#         max_new_tokens=100
#     )
    
#     # Decode generated text
#     generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
#     generated_texts_all.append(generated_text)
    
        
#     print(f"\nSample {i+1}:")
#     print(f"Prompt: {prompt_text}")
#     print(f"Generated: {generated_text}")
#     # [len(prompt_text):][:200]}...")

print("\nAll samples processed with reconstituted model!")


# # Test with all original data samples
# print("\n" + "="*50)
# print("GENERATED TEXT SAMPLES FROM ALL DATA")
# print("="*50)

# generated_texts_all = []
# for i, text in enumerate(data[:3]):  # Process first 3 for demo
#     # Split such that everything before </prompt> or the entire text if </prompt> is not present
#     if '</prompt>' in text:
#         prompt_text = text.split('</prompt>')[0] + '</prompt>'
#     else:
#         prompt_text = text
    
#     # Tokenize with proper padding
#     tokenized = tokenizer(
#         prompt_text,
#         max_length=MAX_SEQ_LENGTH,
#         padding='max_length',
#         truncation=True,
#         add_special_tokens=False
#     )
#     token_ids = tokenized['input_ids']
    
#     # Generate using the reconstituted model
#     generated_token_ids = reconstituted_generator.generate(
#         token_ids=token_ids,
#         do_sample=False,
#         max_new_tokens=100
#     )
    
#     # Decode generated text
#     generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
#     generated_texts_all.append(generated_text)
    
#     # Extract and print prompt for display
#     if '<prompt>' in text and '</prompt>' in text:
#         display_prompt = text.split('<prompt>')[1].split('</prompt>')[0]
#     else:
#         display_prompt = text[:100] + "..." if len(text) > 100 else text
        
#     print(f"\nSample {i+1}:")
#     print(f"Prompt: {text}")
#     print(f"Generated: {generated_text}") # [len(prompt_text):][:200]}...")

print("\nAll samples processed!")



## Model validation
print("Validation")
reconstituted_model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

results = reconstituted_model.evaluate(x_test_packaged, y_test_packaged)
print("Test loss:", results[0])
print("Test accuracy:", results[-1])


