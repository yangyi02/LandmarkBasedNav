import collections
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from tensorflow.python import debug as tf_debug

from utils.mem_simp import MemoryAccess
#from mem_simp import MemoryAccess

DNCState = collections.namedtuple('DNCState', ('access_state', 'controller_state'))
DNCState.name = 'DNCState'

class dotdict(dict):
   __getattr__ = dict.get
   __setattr__ = dict.__setitem__
   __delattr__ = dict.__delitem__

class DNC(tf.nn.rnn_cell.RNNCell):
  """DNC core module.
  Contains controller and memory access module.
  """

  def __init__(self,
               config,
               output_size,
               clip_value=None,
               name='dnc'):
    """Initializes the DNC core.
    Args:
      access_config: dictionary of access module configurations.
      controller_config: dictionary of controller (LSTM) module configurations.
      output_size: output dimension size of core.
      clip_value: clips controller and core output values to between
          `[-clip_value, clip_value]` if specified.
      name: module name (default 'dnc').
    Raises:
      TypeError: if direct_input_size is not None for any access module other
        than KeyValueMemory.
    """
    super(DNC, self).__init__(name=name)
    self._config = config
    self._controller = tf.nn.rnn_cell.LSTMCell(num_units=self._config.controller_h_size)
    self._access = MemoryAccess(memory_size=self._config.memory_size, word_size=self._config.word_size,\
      key_size=self._config.key_size, num_reads=self._config.num_reads, num_writes=self._config.num_writes)

    self._output_size = output_size
    self._clip_value = clip_value or 0

    self._output_size = tf.TensorShape([output_size])
    self._state_size = DNCState(
        access_state=self._access.state_size,
        controller_state=self._controller.state_size)

  def _clip_if_enabled(self, x):
    if self._clip_value > 0:
      return tf.clip_by_value(x, -self._clip_value, self._clip_value)
    else:
      return x

  def __call__(self, inputs, prev_state):
    """Connects the DNC core into the graph.
    Args:
      inputs: Tensor input of shape (batch_size, input_size+1), last dimension is a flag
      prev_state: A `DNCState` tuple containing the fields `access_output`,
          `access_state` and `controller_state`. `access_output` is a 3-D Tensor
          of shape `[batch_size, num_reads, word_size]` containing read words.
          `access_state` is a tuple of the access module's state, and
          `controller_state` is a tuple of controller module's state.
    Returns:
      A tuple `(output, next_state)` where `output` is a tensor and `next_state`
      is a `DNCState` tuple containing the fields `access_output`,
      `access_state`, and `controller_state`.
    """

    prev_access_state = prev_state.access_state
    prev_controller_state = prev_state.controller_state

    controller_input = inputs[:,:-1]

    # controller output of shape (batch_size, controller_h_size)
    controller_output, controller_state = self._controller(
        controller_input, prev_controller_state)

    controller_output = self._clip_if_enabled(controller_output)
    controller_state = tf.nn.rnn_cell.LSTMStateTuple(self._clip_if_enabled(controller_state[0]),self._clip_if_enabled(controller_state[1]))

    access_state = self._access(inputs[:,:-1], controller_output, prev_access_state, inputs[:,-1])

    output = inputs

    return output, DNCState(
        access_state=access_state,
        controller_state=controller_state)

  def value_iteration(self, access_state, src_inputs, tgt_inputs, max_len, iter_len=10):
    """Conduct value iteration on top of the memory for planning.
    assume inputs are keys for now
    Args:
      state: A `DNCState` tuple containing the fields `access_output`,
          `access_state` and `controller_state`. `access_output` is a 3-D Tensor
          of shape `[batch_size, num_reads, word_size]` containing read words.
          `access_state` is a tuple of the access module's state, and
          `controller_state` is a tuple of controller module's state.
      src_inputs: Tensor input of shape (batch_size, input_size)
      tgt_inputs: Tensor input of shape (batch_size, input_size)
      iter_len: the length of value iteration
    Returns:
      path: A tensor with shape (batch_size, max_len, input_size)
    """
    def _vector_norms(m):
      squared_norms = tf.reduce_sum(m * m, axis=2, keep_dims=True)
      return tf.sqrt(squared_norms + 1e-6)

    def _cos_sim(memory_key, keys):
      # memory: [batch_size, memory_size, key_size]
      # keys: [batch_size, key_size]
      # return similarity: [batch_size, memory_size]
      keys = tf.expand_dims(keys, 1)
      dot = tf.matmul(keys, memory_key, adjoint_b=True)
      memory_norms = _vector_norms(memory_key)
      key_norms = _vector_norms(keys)
      norm = tf.matmul(key_norms, memory_norms, adjoint_b=True)
      similarity = dot / (norm + 1e-6)
      similarity = tf.squeeze(similarity, axis=1)
      ## soft to hard
      #similarity = tf.one_hot(tf.argmax(similarity, axis=1),depth=tf.shape(similarity)[1])
      return similarity

    def _read_mem(memory, read_weights):
      # memory: [batch_size, memory_size, word_size]
      # read_weights: [batch_size, memory_size]
      # return read_out: [batch_size, word_size]
      return tf.reduce_sum(tf.expand_dims(read_weights, axis=2)*memory, axis=1)

    # eric: how to set sim here
    L = tf.squeeze(access_state.linkage.link, axis=1) 
    L = tf.maximum(L, tf.transpose(L, perm=[0,2,1])) # (batch_size, memory_size, memory_size)
    src_sim = _cos_sim(access_state.memory_key, src_inputs) # [batch_size, memory_size]
    tgt_sim = _cos_sim(access_state.memory_key, tgt_inputs) # [batch_size, memory_size]

    src_sim = tf.nn.softmax(src_sim*100)
    tgt_sim = tf.nn.softmax(tgt_sim*100)

    R = L*tf.expand_dims(tgt_sim, axis=1) # (batch_size, memory_size, memory_size)
    gamma = 0.8
    V = tf.expand_dims(tf.zeros_like(tgt_sim),1) # (batch_size, 1, memory_size)
    for i in range(iter_len):
      self.Q = (R+gamma*V)*L # (batch_size, memory_size, memory_size)
      V = tf.expand_dims(tf.reduce_max(self.Q, axis=2),1)
    cur_sim = src_sim
    #cur_sim = tf.nn.softmax(cur_sim*100)
    #cur_sim = cur_sim/(tf.reduce_sum(cur_sim, axis=1, keep_dims=True)+1e-6)
    i = tf.constant(1)
    while_condition = lambda i, cur_sim, out: tf.less(i, max_len)
    out = tf.expand_dims(_read_mem(access_state.memory, cur_sim),1)
    def body(i, cur_sim, out):
        cur_sim = tf.reduce_sum(self.Q*tf.expand_dims(cur_sim,2),1)
        cur_sim = cur_sim/(tf.reduce_sum(cur_sim, axis=1, keep_dims=True)+1e-6)
        cur_sim = tf.nn.softmax(cur_sim*100)
        #cur_sim = tf.one_hot(tf.argmax(cur_sim,1), depth=tf.shape(cur_sim)[1])
        out = tf.concat([out, tf.expand_dims(_read_mem(access_state.memory, cur_sim),1)], axis=1)
        return [tf.add(i, 1), cur_sim, out]
    r, cur_sim, out = tf.while_loop(while_condition, body, loop_vars=[i, cur_sim, out],
      shape_invariants=[i.get_shape(), tf.TensorShape([None, None]), tf.TensorShape([out.get_shape()[0], None, out.get_shape()[2]])])
    return out

  @staticmethod
  def zero_state(config, batch_size, dtype=np.float32):
    return DNCState(
        access_state=MemoryAccess.zero_state(config, batch_size, dtype),
        controller_state=tf.nn.rnn_cell.LSTMStateTuple(np.zeros([batch_size, config.controller_h_size], dtype=dtype), \
          np.zeros([batch_size, config.controller_h_size], dtype=dtype)))

  @staticmethod
  def state_placeholder(config, dtype=tf.float32):
    return DNCState(
        access_state=MemoryAccess.state_placeholder(config, dtype),
        controller_state=tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(dtype, shape=(None, config.controller_h_size)), \
          tf.placeholder(dtype, shape=(None, config.controller_h_size))))

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

if __name__ == '__main__':
    #### Caution: word_size must be equal to input_size-1 (last dim is flag) ####
    config = {"memory_size": 3, "word_size": 3, "key_size": 3, "num_reads": 1, "num_writes": 1, "controller_h_size": 6}
    config = dotdict(config)
    dnc = DNC(config=config, output_size=4, clip_value=20, name='dnc')
    ini_state = dnc.state_placeholder(config)
    
    inputs = tf.placeholder(tf.float32, shape=(None, 4))
    out, h_state_out = dnc(inputs, ini_state)

    #inputs = tf.placeholder(tf.float32, shape=(None, 4, 3))
    #out, h_state_out = tf.nn.dynamic_rnn(inputs=inputs, cell=dnc, dtype=tf.float32, initial_state=ini_state)

    sess = tf.Session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    batch_size = 1
    fd = {
      ini_state: dnc.zero_state(config, batch_size),
      inputs: np.concatenate([np.random.random([batch_size, 3]), np.array([[0]])], 1),
    }
    init = tf.global_variables_initializer()
    sess.run(init)
    out_eva, h_state_out_eva = sess.run([out, h_state_out], feed_dict=fd)

    print(out_eva.shape, out_eva, h_state_out_eva)

    fd = {
      ini_state: h_state_out_eva,
      inputs: np.concatenate([np.random.random([batch_size, 3]), np.array([[1]])], 1),
    }
    out_eva, h_state_out_eva = sess.run([out, h_state_out], feed_dict=fd)

    print(out_eva.shape, out_eva, h_state_out_eva)