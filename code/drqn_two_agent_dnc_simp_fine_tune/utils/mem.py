import collections
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

import utils.addressing

AccessState = collections.namedtuple('AccessState', (
    'memory', 'read_weights', 'write_weights', 'linkage', 'usage'))

def _erase_and_write(memory, address, reset_weights, values):
  """Module to erase and write in the external memory.
  Erase operation:
    M_t'(i) = M_{t-1}(i) * (1 - w_t(i) * e_t)
  Add operation:
    M_t(i) = M_t'(i) + w_t(i) * a_t
  where e are the reset_weights, w the write weights and a the values.
  Args:
    memory: 3-D tensor of shape `[batch_size, memory_size, word_size]`.
    address: 3-D tensor `[batch_size, num_writes, memory_size]`.
    reset_weights: 3-D tensor `[batch_size, num_writes, word_size]`.
    values: 3-D tensor `[batch_size, num_writes, word_size]`.
  Returns:
    3-D tensor of shape `[batch_size, memory_size, word_size]`.
  """
  with tf.name_scope('erase_memory', values=[memory, address, reset_weights]):
    expand_address = tf.expand_dims(address, 3)
    reset_weights = tf.expand_dims(reset_weights, 2)
    weighted_resets = expand_address * reset_weights
    reset_gate = tf.reduce_prod(1 - weighted_resets, [1])
    memory *= reset_gate

  with tf.name_scope('additive_write', values=[memory, address, values]):
    add_matrix = tf.matmul(address, values, adjoint_a=True)
    memory += add_matrix

  return memory

class dotdict(dict):
   __getattr__ = dict.get
   __setattr__ = dict.__setitem__
   __delattr__ = dict.__delitem__

class MemoryAccess(tf.nn.rnn_cell.RNNCell):
  """Access module of the Differentiable Neural Computer.
  This memory module supports multiple read and write heads. It makes use of:
  *   `addressing.TemporalLinkage` to track the temporal ordering of reads in
      memory for each read head.
  *   `addressing.FreenessAllocator` for keeping track of memory usage, where
      usage increase when a memory location is written to, and decreases when
      memory is read from that the controller says can be freed.
  Write-address selection is done by an interpolation between content-based
  lookup and using unused memory.
  Read-address selection is done by an interpolation of content-based lookup
  and following the link graph in the forward or backwards read direction.
  """

  def __init__(self,
               memory_size=128,
               word_size=20,
               num_reads=1,
               num_writes=1,
               name='memory_access'):
    """Creates a MemoryAccess module.
    Args:
      memory_size: The number of memory slots (N in the DNC paper).
      word_size: The width of each memory slot (W in the DNC paper)
      num_reads: The number of read heads (R in the DNC paper).
      num_writes: The number of write heads (fixed at 1 in the paper).
      name: The name of the module.
    """
    super(MemoryAccess, self).__init__(name=name)
    self._memory_size = memory_size
    self._word_size = word_size
    self._num_reads = num_reads
    self._num_writes = num_writes

    self._write_content_weights_mod = utils.addressing.CosineWeights(
        num_writes, word_size)
    self._read_content_weights_mod = utils.addressing.CosineWeights(
        num_reads, word_size)

    self._linkage = utils.addressing.TemporalLinkage(memory_size, num_writes)
    self._freeness = utils.addressing.Freeness(memory_size)

  def __call__(self, inputs, prev_state):
    """Connects the MemoryAccess module into the graph.
    Args:
      inputs: tensor of shape `[batch_size, input_size]`. This is used to
          control this access module.
      prev_state: Instance of `AccessState` containing the previous state.
    Returns:
      A tuple `(output, next_state)`, where `output` is a tensor of shape
      `[batch_size, num_reads, word_size]`, and `next_state` is the new
      `AccessState` named tuple at the current time t.
    """
    inputs = self._read_inputs(inputs)

    # Update usage using inputs['free_gate'] and previous read & write weights.
    usage = self._freeness(
        write_weights=prev_state.write_weights,
        free_gate=inputs['free_gate'],
        read_weights=prev_state.read_weights,
        prev_usage=prev_state.usage)
    #usage = tf.Print(usage, [usage], message="usage: ", summarize=6)

    # Write to memory.
    write_weights = self._write_weights(inputs, prev_state.memory, usage)
    #write_weights = tf.Print(write_weights, [write_weights], message="write weights: ", summarize=6)
    memory = _erase_and_write(
        prev_state.memory,
        address=write_weights,
        reset_weights=inputs['erase_vectors'],
        values=inputs['write_vectors'])

    linkage_state = self._linkage(write_weights, prev_state.linkage)

    # Read from memory.
    read_weights = self._read_weights(
        inputs,
        memory=memory,
        prev_read_weights=prev_state.read_weights,
        link=linkage_state.link)
    #read_weights = tf.Print(read_weights, [read_weights], message="read weights: ", summarize=6)
    read_words = tf.matmul(read_weights, memory)

    return (read_words, AccessState(
        memory=memory,
        read_weights=read_weights,
        write_weights=write_weights,
        linkage=linkage_state,
        usage=usage))

  def _read_inputs(self, inputs):
    """Applies transformations to `inputs` to get control for this module."""

    def _linear(first_dim, second_dim, name, activation=None):
      """Returns a linear transformation of `inputs`, followed by a reshape."""
      linear = layers.fully_connected(inputs, first_dim*second_dim, activation_fn = activation, scope=name, \
        weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
      return tf.reshape(linear, [-1, first_dim, second_dim])

    # v_t^i - The vectors to write to memory, for each write head `i`.
    write_vectors = _linear(self._num_writes, self._word_size, 'write_vectors')

    # e_t^i - Amount to erase the memory by before writing, for each write head.
    erase_vectors = _linear(self._num_writes, self._word_size, 'erase_vectors', tf.sigmoid)

    # f_t^j - Amount that the memory at the locations read from at the previous
    # time step can be declared unused, for each read head `j`.
    free_gate = layers.fully_connected(inputs, self._num_reads, activation_fn = tf.nn.sigmoid, scope='free_gate', \
        weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())

    # g_t^{a, i} - Interpolation between writing to unallocated memory and
    # content-based lookup, for each write head `i`. Note: `a` is simply used to
    # identify this gate with allocation vs writing (as defined below).
    allocation_gate = layers.fully_connected(inputs, self._num_writes, activation_fn = tf.nn.sigmoid, scope='allocation_gate', \
        weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())

    # g_t^{w, i} - Overall gating of write amount for each write head.
    write_gate = layers.fully_connected(inputs, self._num_writes, activation_fn = tf.nn.sigmoid, scope='write_gate', \
        weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())

    # \pi_t^j - Mixing between "backwards" and "forwards" positions (for
    # each write head), and content-based lookup, for each read head.
    num_read_modes = 1 + 2 * self._num_writes
    read_mode = tf.nn.softmax(_linear(self._num_reads, num_read_modes, 'read_mode'))

    # Parameters for the (read / write) "weights by content matching" modules.
    write_keys = _linear(self._num_writes, self._word_size, 'write_keys')
    write_strengths = layers.fully_connected(inputs, self._num_writes, activation_fn = None, scope='write_strengths', \
        weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())

    read_keys = _linear(self._num_reads, self._word_size, 'read_keys')
    read_strengths = layers.fully_connected(inputs, self._num_reads, activation_fn = None, scope='read_strengths', \
        weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())

    result = {
        'read_content_keys': read_keys,
        'read_content_strengths': read_strengths,
        'write_content_keys': write_keys,
        'write_content_strengths': write_strengths,
        'write_vectors': write_vectors,
        'erase_vectors': erase_vectors,
        'free_gate': free_gate,
        'allocation_gate': allocation_gate,
        'write_gate': write_gate,
        'read_mode': read_mode,
    }
    return result

  def _write_weights(self, inputs, memory, usage):
    """Calculates the memory locations to write to.
    This uses a combination of content-based lookup and finding an unused
    location in memory, for each write head.
    Args:
      inputs: Collection of inputs to the access module, including controls for
          how to chose memory writing, such as the content to look-up and the
          weighting between content-based and allocation-based addressing.
      memory: A tensor of shape  `[batch_size, memory_size, word_size]`
          containing the current memory contents.
      usage: Current memory usage, which is a tensor of shape `[batch_size,
          memory_size]`, used for allocation-based addressing.
    Returns:
      tensor of shape `[batch_size, num_writes, memory_size]` indicating where
          to write to (if anywhere) for each write head.
    """
    with tf.name_scope('write_weights', values=[inputs, memory, usage]):
      # c_t^{w, i} - The content-based weights for each write head.
      write_content_weights = self._write_content_weights_mod(
          memory, inputs['write_content_keys'],
          inputs['write_content_strengths'])

      # a_t^i - The allocation weights for each write head.
      write_allocation_weights = self._freeness.write_allocation_weights(
          usage=usage,
          write_gates=(inputs['allocation_gate'] * inputs['write_gate']),
          num_writes=self._num_writes)

      # Expands gates over memory locations.
      allocation_gate = tf.expand_dims(inputs['allocation_gate'], -1)
      write_gate = tf.expand_dims(inputs['write_gate'], -1)

      # w_t^{w, i} - The write weightings for each write head.
      return write_gate * (allocation_gate * write_allocation_weights +
                           (1 - allocation_gate) * write_content_weights)

  def _read_weights(self, inputs, memory, prev_read_weights, link):
    """Calculates read weights for each read head.
    The read weights are a combination of following the link graphs in the
    forward or backward directions from the previous read position, and doing
    content-based lookup. The interpolation between these different modes is
    done by `inputs['read_mode']`.
    Args:
      inputs: Controls for this access module. This contains the content-based
          keys to lookup, and the weightings for the different read modes.
      memory: A tensor of shape `[batch_size, memory_size, word_size]`
          containing the current memory contents to do content-based lookup.
      prev_read_weights: A tensor of shape `[batch_size, num_reads,
          memory_size]` containing the previous read locations.
      link: A tensor of shape `[batch_size, num_writes, memory_size,
          memory_size]` containing the temporal write transition graphs.
    Returns:
      A tensor of shape `[batch_size, num_reads, memory_size]` containing the
      read weights for each read head.
    """
    with tf.name_scope(
        'read_weights', values=[inputs, memory, prev_read_weights, link]):
      # c_t^{r, i} - The content weightings for each read head.
      content_weights = self._read_content_weights_mod(
          memory, inputs['read_content_keys'], inputs['read_content_strengths'])

      # Calculates f_t^i and b_t^i.
      forward_weights = self._linkage.directional_read_weights(
          link, prev_read_weights, forward=True)
      backward_weights = self._linkage.directional_read_weights(
          link, prev_read_weights, forward=False)

      backward_mode = inputs['read_mode'][:, :, :self._num_writes]
      forward_mode = (
          inputs['read_mode'][:, :, self._num_writes:2 * self._num_writes])
      content_mode = inputs['read_mode'][:, :, 2 * self._num_writes]

      read_weights = (
          tf.expand_dims(content_mode, 2) * content_weights + tf.reduce_sum(
              tf.expand_dims(forward_mode, 3) * forward_weights, 2) +
          tf.reduce_sum(tf.expand_dims(backward_mode, 3) * backward_weights, 2))

      return read_weights

  @staticmethod
  def test():
    return np.zeros([1])

  @staticmethod
  def zero_state(config, batch_size, dtype=np.float32):
    return AccessState(
        memory=np.zeros([batch_size, config.memory_size, config.word_size], dtype=dtype),
        read_weights=np.zeros([batch_size, config.num_reads, config.memory_size], dtype=dtype),
        write_weights=np.zeros([batch_size, config.num_writes, config.memory_size], dtype=dtype),
        linkage=utils.addressing.TemporalLinkage.zero_state(config, batch_size, dtype),
        usage=np.zeros([batch_size, config.memory_size], dtype=dtype))

  @staticmethod
  def state_placeholder(config, dtype=tf.float32):
    return AccessState(
        memory=tf.placeholder(dtype, shape=(None, config.memory_size, config.word_size)),
        read_weights=tf.placeholder(dtype, shape=(None, config.num_reads, config.memory_size)),
        write_weights=tf.placeholder(dtype, shape=(None, config.num_writes, config.memory_size)),
        linkage=utils.addressing.TemporalLinkage.state_placeholder(config, dtype),
        usage=tf.placeholder(dtype, shape=(None, config.memory_size)))

  @property
  def state_size(self):
    """Returns a tuple of the shape of the state tensors."""
    return AccessState(
        memory=tf.TensorShape([self._memory_size, self._word_size]),
        read_weights=tf.TensorShape([self._num_reads, self._memory_size]),
        write_weights=tf.TensorShape([self._num_writes, self._memory_size]),
        linkage=self._linkage.state_size,
        usage=self._freeness.state_size)

  @property
  def output_size(self):
    """Returns the output shape."""
    return tf.TensorShape([self._num_reads, self._word_size])

def np_softmax(x):
  e_x = np.exp(x-np.max(x))
  return e_x / e_x.sum(axis=-1, keepdims=True)

if __name__ == '__main__':
    config = {"memory_size": 3, "word_size": 2, "num_reads": 1, "num_writes": 1}
    config = dotdict(config)
    mem = MemoryAccess(memory_size=config.memory_size,
               word_size=config.word_size,
               num_reads=config.num_reads,
               num_writes=config.num_writes,
               name='memory_access')
    ini_state = MemoryAccess.state_placeholder(config)
    inputs = tf.placeholder(tf.float32, shape=(None, 4, 3))
    out, h_state_out = tf.nn.dynamic_rnn(inputs=inputs, cell=mem, dtype=tf.float32, initial_state=ini_state)

    batch_size = 2
    sess = tf.Session()
    fd = {
      ini_state: MemoryAccess.zero_state(config, batch_size),
      inputs: np.random.random([batch_size, 4, 3]),
    }
    init = tf.global_variables_initializer()
    sess.run(init)
    out_eva, h_state_out_eva = sess.run([out, h_state_out], feed_dict=fd)
    print(out_eva.shape, out_eva, h_state_out_eva)