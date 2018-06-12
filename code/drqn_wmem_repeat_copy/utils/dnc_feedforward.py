import collections
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from tensorflow.python import debug as tf_debug

from utils.mem import MemoryAccess

DNCState = collections.namedtuple('DNCState', ('access_output', 'access_state'))
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
    self._access = MemoryAccess(memory_size=self._config.memory_size, word_size=self._config.word_size,\
      num_reads=self._config.num_reads, num_writes=self._config.num_writes)

    self._access_output_size = np.prod(self._access.output_size.as_list())
    self._output_size = output_size
    self._clip_value = clip_value or 0

    self._output_size = tf.TensorShape([output_size])
    self._state_size = DNCState(
        access_output=self._access_output_size,
        access_state=self._access.state_size)

  def _clip_if_enabled(self, x):
    if self._clip_value > 0:
      return tf.clip_by_value(x, -self._clip_value, self._clip_value)
    else:
      return x

  def __call__(self, inputs, prev_state):
    """Connects the DNC core into the graph.
    Args:
      inputs: Tensor input of shape (batch_size, input_size)
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

    prev_access_output = prev_state.access_output
    prev_access_state = prev_state.access_state

    controller_input = tf.concat(
        [inputs, tf.reshape(prev_access_output, shape=[-1, self._config.num_reads*self._config.word_size])], 1)

    # controller output of shape (batch_size, controller_h_size)
    controller_output = layers.fully_connected(controller_input, 64, activation_fn = tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
    controller_output = layers.fully_connected(controller_output, self._config.controller_h_size, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())

    controller_output = self._clip_if_enabled(controller_output)

    access_output, access_state = self._access(controller_output, prev_access_state)

    output = tf.concat([controller_output, tf.reshape(access_output, shape=[-1, self._config.num_reads*self._config.word_size])], 1)

    output = layers.fully_connected(output, self._output_size.as_list()[0], activation_fn = None, scope='output_linear', \
        weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
    output = self._clip_if_enabled(output)

    return output, DNCState(
        access_output=access_output,
        access_state=access_state)

  @staticmethod
  def zero_state(config, batch_size, dtype=np.float32):
    return DNCState(
        access_output=np.zeros([batch_size, config.num_reads, config.word_size], dtype=dtype),
        access_state=MemoryAccess.zero_state(config, batch_size, dtype))

  @staticmethod
  def state_placeholder(config, dtype=tf.float32):
    return DNCState(
        access_output=tf.placeholder(dtype, shape=(None, config.num_reads, config.word_size)),
        access_state=MemoryAccess.state_placeholder(config, dtype))

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

if __name__ == '__main__':
    config = {"memory_size": 3, "word_size": 2, "num_reads": 1, "num_writes": 1}
    config = dotdict(config)
    dnc = DNC(config=config, output_size=4, clip_value=20, name='dnc')
    ini_state = dnc.state_placeholder(config)
    
    inputs = tf.placeholder(tf.float32, shape=(None, 3))
    out, h_state_out = dnc(inputs, ini_state)

    #inputs = tf.placeholder(tf.float32, shape=(None, 4, 3))
    #out, h_state_out = tf.nn.dynamic_rnn(inputs=inputs, cell=dnc, dtype=tf.float32, initial_state=ini_state)

    sess = tf.Session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    batch_size = 2
    fd = {
      ini_state: dnc.zero_state(config, batch_size),
      inputs: np.random.random([batch_size, 3]),
    }
    init = tf.global_variables_initializer()
    sess.run(init)
    out_eva, h_state_out_eva = sess.run([out, h_state_out], feed_dict=fd)

    fd = {
      ini_state: h_state_out_eva,
      inputs: np.random.random([batch_size, 3]),
    }
    out_eva, h_state_out_eva = sess.run([out, h_state_out], feed_dict=fd)

    print(out_eva.shape, out_eva, h_state_out_eva)