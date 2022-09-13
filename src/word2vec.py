#heavely inspired by https://www.tensorflow.org/tutorials/text/word2vec
import re
import string

import tensorflow as tf
import numpy as np
import joblib

#import tqdm
#import io

SEED=42
BATCH_SIZE = 1024
BUFFER_SIZE = 10000

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  #for sequence in tqdm.tqdm(sequences):
  for sequence in sequences:
    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels

class Word2VecEmbedding(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim,window_size):
    super(Word2VecEmbedding, self).__init__()
    self.target_embedding = tf.layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = tf.layers.Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=2*window_size+1)

  def call(self, pair):
    target, context = pair
    # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
    # context: (batch, context)
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    # target: (batch,)
    word_emb = self.target_embedding(target)
    # word_emb: (batch, embed)
    context_emb = self.context_embedding(context)
    # context_emb: (batch, context, embed)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    # dots: (batch, context)
    return dots

class Word2Vec:
    def __init__(self,embedding_dim=128,vocab_size=4096,sequence_length=10,window_size=2,num_ns=4,epochs=20):
        self.embedding_dim   = embedding_dim
        self.vocab_size      = vocab_size
        self.vocab           = None
        self.embeddings      = None

        self.window_size     = window_size
        self.sequence_length = sequence_length
        self.num_ns          = num_ns                
        self.epochs          = epochs

    def __vectorize__(self,input_text):
        input_text_ds = tf.data.Dataset.from_tensor_slices(input_text)
        #vectorize layer
        vectorize_layer = tf.layers.TextVectorization(max_tokens=self.vocab_size,output_mode='int',output_sequence_length=self.sequence_length)
        #fit/adapt layer to input text
        vectorize_layer.adapt(input_text_ds.batch(BATCH_SIZE))
        #set vocab
        self.vocab = vectorize_layer.get_vocabulary()
        # vectorize input text
        text_vector_ds = input_text_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()
        #convert to numpy array
        sequences = list(text_vector_ds.as_numpy_iterator())
        return sequences
    
    def __gen_training_data__(self,input_text):
        sequences = self.__vectorize__(input_text)
        
        targets, contexts, labels = generate_training_data(sequences=sequences,window_size=self.window_size,num_ns=self.num_ns,vocab_size=self.vocab_size,seed=SEED)
        
        targets = np.array(targets)
        contexts = np.array(contexts)[:,:,0]
        labels = np.array(labels)

        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def fit(self,input_text):

        # training dataset
        dataset = self.__gen_training_data__(input_text)

        #create embedding model
        word2vec = Word2VecEmbedding(self.vocab_size, self.embedding_dim)
        word2vec.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

        # create fit embedding model
        word2vec.fit(dataset, self.epochs)

        #get weights from embedding model
        weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
        self.embeddings = { word: weights[index] for index, word in enumerate(self.vocab) if index != 0 }
    
    def serialize(self,title):
        joblib.dump(value=self.embeddings,filename=f"word2vec_{title}_{self.embedding_dim}_{self.vocab_size}_{self.sequence_length}_{self.window_size}_{self.num_ns}.joblib")


#path_to_file = ""
#text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))

# Now, create a custom standardization function to lowercase the text and
# remove punctuation.
# def custom_standardization(input_data):
#   lowercase = tf.strings.lower(input_data)
#   return tf.strings.regex_replace(lowercase,
#                                   '[%s]' % re.escape(string.punctuation), '')

# Define the vocabulary size and the number of words in a sequence.
# vocab_size = 4096
# sequence_length = 10

# Use the `TextVectorization` layer to normalize, split, and map strings to
# integers. Set the `output_sequence_length` length to pad all samples to the
# same length.
# vectorize_layer = tf.layers.TextVectorization(
#     standardize=custom_standardization,
#     max_tokens=vocab_size,
#     output_mode='int',
#     output_sequence_length=sequence_length)

#vectorize_layer.adapt(text_ds.batch(1024))

# Vectorize the data in text_ds.
#text_vector_ds = text_ds.batch(1024).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()

#sequences = list(text_vector_ds.as_numpy_iterator())
#print(len(sequences))

# targets, contexts, labels = generate_training_data(
#     sequences=sequences,
#     window_size=2,
#     num_ns=4,
#     vocab_size=vocab_size,
#     seed=SEED)

# targets = np.array(targets)
# contexts = np.array(contexts)[:,:,0]
# labels = np.array(labels)

# print('\n')
# print(f"targets.shape: {targets.shape}")
# print(f"contexts.shape: {contexts.shape}")
# print(f"labels.shape: {labels.shape}")

# BATCH_SIZE = 1024
# BUFFER_SIZE = 10000
# dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
# dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
# print(dataset)

# dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
# print(dataset)
#embedding_dim = 128
# word2vec = Word2VecEmbedding(vocab_size, embedding_dim)
# word2vec.compile(optimizer='adam',
#                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#                  metrics=['accuracy'])

# word2vec.fit(dataset, epochs=20)

# weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
# vocab = vectorize_layer.get_vocabulary()

# out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
# out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

# for index, word in enumerate(vocab):
#   if index == 0:
#     continue  # skip 0, it's padding.
#   vec = weights[index]
#   out_v.write('\t'.join([str(x) for x in vec]) + "\n")
#   out_m.write(word + "\n")
# out_v.close()
# out_m.close()