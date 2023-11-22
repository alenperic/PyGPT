import tensorflow as tf
from transformers import GPT2Tokenizer

# Custom Learning Rate Scheduler
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# Function to create and save a tokenizer
def create_and_save_tokenizer(dataset_path, save_path):
    """
    Create a tokenizer based on the dataset and save it for later use.
    This is an example and may vary based on your dataset and model requirements.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Additional tokenizer training code would go here if you're not using a pre-trained tokenizer

    tokenizer.save_pretrained(save_path)

# Function to load a saved tokenizer
def load_tokenizer(path):
    return GPT2Tokenizer.from_pretrained(path)

# Additional utility functions can be added here
