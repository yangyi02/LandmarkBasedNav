import collections
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

import utils.addressing_simp2
#import addressing_simp2

AccessState = collections.namedtuple('AccessState', (
    'memory_key', 'memory', 'write_weights', 'linkage', 'usage'))

def _erase_and_write(memory, address, values):
  # alway write unit vector modulated by writing weights
  """Module to erase and write in the external memory.
  Erase operation:
    M_t'(i) = M_{t-1}(i) * (1 - w_t(i))
  Add operation:
    M_t(i) = M_t'(i) + w_t(i) * a_t
  w the write weights and a the values.
  Args:
    memory: 3-D tensor of shape `[batch_size, memory_size, word_size]`.
    address: 3-D tensor `[batch_size, num_writes, memory_size]`.
    values: 3-D tensor `[batch_size, num_writes, word_size]`.
  Returns:
    3-D tensor of shape `[batch_size, memory_size, word_size]`.
  """
  with tf.name_scope('erase_memory', values=[memory, address]):
    expand_address = tf.tile(tf.expand_dims(address, 3),[1,1,1,tf.shape(memory)[2]])
    reset_gate = tf.reduce_prod(1 - expand_address, [1])
    memory *= reset_gate

  with tf.name_scope('additive_write', values=[memory, address, values]):
    squared_norms = tf.reduce_sum(values * values, axis=2, keep_dims=True)
    #values = values/tf.sqrt(squared_norms + 1e-6)
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
               key_size=20,
               mask_size=1,
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
    self._key_size = key_size
    self._mask_size = mask_size
    self._num_reads = num_reads
    self._num_writes = num_writes

    self._write_content_weights_mod = utils.addressing_simp2.MaskedRotInvariantCosineWeights(mask_size=self._mask_size)

    self._linkage = utils.addressing_simp2.TemporalLinkage(memory_size, num_writes)
    self._freeness = utils.addressing_simp2.Freeness(memory_size)

  def __call__(self, raw_inputs, cur_mask, prev_state, write_gate, flag):
    """Connects the MemoryAccess module into the graph.
    Args:
      raw_inputs: tensor of shape `[batch_size, raw_input_size, 4]`. This is the content to be
          stored in the memory with four different orientations
      cur_mask: tensor of shape `[batch_size, mask_size]`
      prev_state: Instance of `AccessState` containing the previous state.
      write_gate: tensor of shape `[batch_size]`, indicating whether write or not
      flag: a float tensor of shape `[batch_size]`, indicating whether the sequence has ended
    Returns:
      `next_state` is the new `AccessState` named tuple at the current time t.
    """
    inputs = self._read_inputs(raw_inputs, prev_state.memory_key, write_gate)

    # Update usage using inputs['free_gate'] and previous read & write weights.
    usage = self._freeness(
        write_weights=prev_state.write_weights,
        prev_usage=prev_state.usage)
    #usage = tf.Print(usage, [usage], message="usage: ", summarize=6)

    # Write to memory.
    write_weights_a, write_weights_c = self._write_weights(inputs, prev_state.memory_key, usage) # only write for allocation
    write_weights = write_weights_a+write_weights_c
    #write_weights = self._write_weights(inputs, prev_state.memory_key, usage)
    #write_weights = tf.Print(write_weights, [write_weights], message="write weights: ", summarize=6)
    memory = _erase_and_write(
        prev_state.memory,
        address=write_weights_a,
        values=tf.tile(tf.expand_dims(raw_inputs[:,:,0],1), [1, self._num_writes, 1]))
    memory_key = _erase_and_write(
        prev_state.memory_key,
        address=write_weights_a,
        values=tf.tile(tf.expand_dims(tf.concat([raw_inputs[:,:,0],cur_mask], axis=1),1), [1, self._num_writes, 1]))

    linkage_state = self._linkage(write_weights, prev_state.linkage, flag)

    return AccessState(
        memory_key=memory_key,
        memory=memory,
        write_weights=write_weights,
        linkage=linkage_state,
        usage=usage)

  def _read_inputs(self, raw_inputs, memory_key, write_gate):
    """
    Applies transformations to `inputs` to get control for this module.
    input: (batch_size, input_size, 4)
    memory_key: (batch_size, memory_size, key_size=(word_size+mask_size))
    write_gate: (batch_size, )
    """
    write_keys = tf.reshape(raw_inputs, [-1, self._num_writes, self._word_size, 4])
    similarity = self._write_content_weights_mod.get_sim(
          memory_key[:,:,:-self._mask_size], memory_key[:,:,-self._mask_size:], write_keys)

    allocation_gate = 1-tf.nn.sigmoid(500*(tf.reduce_max(similarity,axis=2)-tf.Variable(0.98)))
    
    #memory_norms = tf.reduce_max(tf.one_hot(tf.argmax(similarity, axis=2), depth=tf.shape(similarity)[2])*
    #  tf.expand_dims(tf.squeeze(memory_norms, axis=2),axis=1), axis=2, keep_dims=True)
    #similarity = tf.reduce_max(similarity, axis=2, keep_dims=True)
    #similarity /= (memory_norms + 1e-6) # eric: either learn gate or specify

    # g_t^{a, i} - Interpolation between writing to unallocated memory and
    # content-based lookup, for each write head `i`. Note: `a` is simply used to
    # identify this gate with allocation vs writing (as defined below).
    #allocation_gate = layers.fully_connected(layers.fully_connected(tf.reduce_max(similarity,axis=2,keep_dims=True),16), 1, activation_fn = tf.nn.sigmoid, scope='allocation_gate', \
    #    weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
    #allocation_gate = tf.reshape(allocation_gate, [-1, self._num_writes])

     # Parameters for the (read / write) "weights by content matching" modules.
    #write_strengths = layers.fully_connected(controller_inputs, self._num_writes, activation_fn = None, scope='write_strengths', \
    #    weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
    write_strengths = 500*tf.ones_like(allocation_gate)

    # g_t^{w, i} - Overall gating of write amount for each write head.
    #write_gate = layers.fully_connected(controller_inputs, self._num_writes, activation_fn = tf.nn.sigmoid, scope='write_gate', \
    #    weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())

    result = {
        'write_content_keys': write_keys,
        'write_content_strengths': write_strengths,
        'allocation_gate': allocation_gate,
        'write_gate': tf.reshape(write_gate,[-1, self._num_writes]),
    }
    return result

  def _write_weights(self, inputs, memory_key, usage):
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
    with tf.name_scope('write_weights', values=[inputs, memory_key, usage]):
      # c_t^{w, i} - The content-based weights for each write head.
      # eric: whether to use cos weight or unormalized cos weight
      write_content_weights = self._write_content_weights_mod(
          memory_key[:,:,:-self._mask_size], memory_key[:,:,-self._mask_size:], inputs['write_content_keys'],
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
      #write_weights = (allocation_gate * write_allocation_weights +
      #                     (1 - allocation_gate) * write_content_weights)
      #return write_gate * write_weights

      write_weights_a = allocation_gate * write_allocation_weights * write_gate
      write_weights_c = (1 - allocation_gate) * write_content_weights * write_gate
      return write_weights_a, write_weights_c

  @staticmethod
  def test():
    return np.zeros([1])

  @staticmethod
  def zero_state(config, batch_size, dtype=np.float32):
    return AccessState(
        memory_key=np.zeros([batch_size, config.memory_size, config.key_size], dtype=dtype),
        memory=np.zeros([batch_size, config.memory_size, config.word_size], dtype=dtype),
        write_weights=np.zeros([batch_size, config.num_writes, config.memory_size], dtype=dtype),
        linkage=utils.addressing_simp2.TemporalLinkage.zero_state(config, batch_size, dtype),
        usage=np.zeros([batch_size, config.memory_size], dtype=dtype))

  @staticmethod
  def state_placeholder(config, dtype=tf.float32):
    return AccessState(
        memory_key=tf.placeholder(dtype, shape=(None, config.memory_size, config.key_size)),
        memory=tf.placeholder(dtype, shape=(None, config.memory_size, config.word_size)),
        write_weights=tf.placeholder(dtype, shape=(None, config.num_writes, config.memory_size)),
        linkage=utils.addressing_simp2.TemporalLinkage.state_placeholder(config, dtype),
        usage=tf.placeholder(dtype, shape=(None, config.memory_size)))

  @property
  def state_size(self):
    """Returns a tuple of the shape of the state tensors."""
    return AccessState(
        memory_key=tf.TensorShape([self._memory_size, self._key_size]),
        memory=tf.TensorShape([self._memory_size, self._word_size]),
        write_weights=tf.TensorShape([self._num_writes, self._memory_size]),
        linkage=self._linkage.state_size,
        usage=self._freeness.state_size)

def np_softmax(x):
  e_x = np.exp(x-np.max(x))
  return e_x / e_x.sum(axis=-1, keepdims=True)

if __name__ == '__main__':
    config = {"memory_size": 3, "word_size": 2, "key_size": 2, "num_reads": 1, "num_writes": 1}
    config = dotdict(config)
    mem = MemoryAccess(memory_size=config.memory_size,
               word_size=config.word_size,
               key_size=config.key_size,
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