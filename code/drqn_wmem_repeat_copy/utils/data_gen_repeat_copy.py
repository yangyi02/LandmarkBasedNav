"""A repeat copy task."""
import collections
import numpy as np
import tensorflow as tf

DatasetTensors = collections.namedtuple('DatasetTensors', ('observations',
                                                           'target', 'mask','seqlen'))

def masked_sigmoid_cross_entropy(logits,
                                 target,
                                 mask,
                                 time_average=False,
                                 log_prob_in_bits=False):
  """Adds ops to graph which compute the (scalar) NLL of the target sequence.
  The logits parametrize independent bernoulli distributions per time-step and
  per batch element, and irrelevant time/batch elements are masked out by the
  mask tensor.
  Args:
    logits: `Tensor` of activations for which sigmoid(`logits`) gives the
        bernoulli parameter.
    target: time-major `Tensor` of target.
    mask: time-major `Tensor` to be multiplied elementwise with cost T x B cost
        masking out irrelevant time-steps.
    time_average: optionally average over the time dimension (sum by default).
    log_prob_in_bits: iff True express log-probabilities in bits (default nats).
  Returns:
    A `Tensor` representing the log-probability of the target.
  """
  xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
  loss_time_batch = tf.reduce_sum(xent, axis=2)
  loss_batch = tf.reduce_sum(loss_time_batch * mask, axis=1)

  batch_size = tf.cast(tf.shape(logits)[0], dtype=loss_time_batch.dtype)

  if time_average:
    mask_count = tf.reduce_sum(mask, axis=1)
    loss_batch /= (mask_count + np.finfo(np.float32).eps)

  loss = tf.reduce_sum(loss_batch) / batch_size
  if log_prob_in_bits:
    loss /= tf.log(2.)

  return loss


def bitstring_readable(data, batch_size, model_output=None, whole_batch=False):
  """Produce a human readable representation of the sequences in data.
  Args:
    data: data to be visualised
    batch_size: size of batch
    model_output: optional model output tensor to visualize alongside data.
    whole_batch: whether to visualise the whole batch. Only the first sample
        will be visualized if False
  Returns:
    A string used to visualise the data batch
  """

  def _readable(datum):
    return '+' + ' '.join(['-' if x == 0 else '%d' % x for x in datum]) + '+'

  obs_batch = data.observations
  targ_batch = data.target

  iterate_over = range(batch_size) if whole_batch else range(1)

  batch_strings = []
  for batch_index in iterate_over:
    obs = obs_batch[:, batch_index, :]
    targ = targ_batch[:, batch_index, :]

    obs_channels = range(obs.shape[1])
    targ_channels = range(targ.shape[1])
    obs_channel_strings = [_readable(obs[:, i]) for i in obs_channels]
    targ_channel_strings = [_readable(targ[:, i]) for i in targ_channels]

    readable_obs = 'Observations:\n' + '\n'.join(obs_channel_strings)
    readable_targ = 'Targets:\n' + '\n'.join(targ_channel_strings)
    strings = [readable_obs, readable_targ]

    if model_output is not None:
      output = model_output[:, batch_index, :]
      output_strings = [_readable(output[:, i]) for i in targ_channels]
      strings.append('Model Output:\n' + '\n'.join(output_strings))

    batch_strings.append('\n\n'.join(strings))

  return '\n' + '\n\n\n\n'.join(batch_strings)


class RepeatCopy(object):
  """Sequence data generator for the task of repeating a random binary pattern.
  When called, an instance of this class will return a tuple of tensorflow ops
  (obs, targ, mask), representing an input sequence, target sequence, and
  binary mask. Each of these ops produces tensors whose first two dimensions
  represent sequence position and batch index respectively. The value in
  mask[t, b] is equal to 1 iff a prediction about targ[t, b, :] should be
  penalized and 0 otherwise.
  For each realisation from this generator, the observation sequence is
  comprised of I.I.D. uniform-random binary vectors (and some flags).
  The target sequence is comprised of this binary pattern repeated
  some number of times (and some flags). Before explaining in more detail,
  let's examine the setup pictorially for a single batch element:
  ```none
  Note: blank space represents 0.
  time ------------------------------------------>
                +-------------------------------+
  mask:         |0000000001111111111111111111111|
                +-------------------------------+
                +-------------------------------+
  target:       |                              1| 'end-marker' channel.
                |         101100110110011011001 |
                |         010101001010100101010 |
                +-------------------------------+
                +-------------------------------+
  observation:  | 1011001                       |
                | 0101010                       |
                |1                              | 'start-marker' channel
                |        3                      | 'num-repeats' channel.
                +-------------------------------+
  ```
  The length of the random pattern and the number of times it is repeated
  in the target are both discrete random variables distributed according to
  uniform distributions whose parameters are configured at construction time.
  The obs sequence has two extra channels (components in the trailing dimension)
  which are used for flags. One channel is marked with a 1 at the first time
  step and is otherwise equal to 0. The other extra channel is zero until the
  binary pattern to be repeated ends. At this point, it contains an encoding of
  the number of times the observation pattern should be repeated. Rather than
  simply providing this integer number directly, it is normalised so that
  a neural network may have an easier time representing the number of
  repetitions internally. To allow a network to be readily evaluated on
  instances of this task with greater numbers of repetitions, the range with
  respect to which this encoding is normalised is also configurable by the user.
  As in the diagram, the target sequence is offset to begin directly after the
  observation sequence; both sequences are padded with zeros to accomplish this,
  resulting in their lengths being equal. Additional padding is done at the end
  so that all sequences in a minibatch represent tensors with the same shape.
  """

  def __init__(
      self,
      num_bits=6,
      batch_size=1,
      min_length=1,
      max_length=1,
      min_repeats=1,
      max_repeats=2,
      norm_max=10,
      log_prob_in_bits=False,
      time_average_cost=False):
    """Creates an instance of RepeatCopy task.
    Args:
      name: A name for the generator instance (for name scope purposes).
      num_bits: The dimensionality of each random binary vector.
      batch_size: Minibatch size per realization.
      min_length: Lower limit on number of random binary vectors in the
          observation pattern.
      max_length: Upper limit on number of random binary vectors in the
          observation pattern.
      min_repeats: Lower limit on number of times the obervation pattern
          is repeated in targ.
      max_repeats: Upper limit on number of times the observation pattern
          is repeated in targ.
      norm_max: Upper limit on uniform distribution w.r.t which the encoding
          of the number of repetitions presented in the observation sequence
          is normalised.
      log_prob_in_bits: By default, log probabilities are expressed in units of
          nats. If true, express log probabilities in bits.
      time_average_cost: If true, the cost at each time step will be
          divided by the `true`, sequence length, the number of non-masked time
          steps, in each sequence before any subsequent reduction over the time
          and batch dimensions.
    """
    self._batch_size = batch_size
    self._num_bits = num_bits
    self._min_length = min_length
    self._max_length = max_length
    self._min_repeats = min_repeats
    self._max_repeats = max_repeats
    self._norm_max = norm_max
    self._log_prob_in_bits = log_prob_in_bits
    self._time_average_cost = time_average_cost

  def _normalise(self, val):
    return val / self._norm_max

  def _unnormalise(self, val):
    return val * self._norm_max

  @property
  def time_average_cost(self):
    return self._time_average_cost

  @property
  def log_prob_in_bits(self):
    return self._log_prob_in_bits

  @property
  def num_bits(self):
    """The dimensionality of each random binary vector in a pattern."""
    return self._num_bits

  @property
  def target_size(self):
    """The dimensionality of the target tensor."""
    return self._num_bits + 1

  @property
  def batch_size(self):
    return self._batch_size

  def __call__(self):
    """
    Implements build method which adds ops to graph.
    Output:
    max_seq_len = max(seq_len+2+seq_len*n_rep+1)
    obs: [batch_size, max_seq_len, num_bits+2]
    targ_obs: [batch_size, max_seq_len, num_bits+1]
    mask: [batch_size, max_seq_len]
    seq_len: [batch_size]
    """

    # short-hand for private fields.
    min_length, max_length = self._min_length, self._max_length
    min_reps, max_reps = self._min_repeats, self._max_repeats
    num_bits = self.num_bits
    batch_size = self.batch_size

    # We reserve one dimension for the num-repeats and one for the start-marker.
    full_obs_size = num_bits + 2
    # We reserve one target dimension for the end-marker.
    full_targ_size = num_bits + 1
    start_end_flag_idx = full_obs_size - 2
    num_repeats_channel_idx = full_obs_size - 1

    # Samples each batch index's sequence length and the number of repeats.
    sub_seq_length_batch = np.random.randint(min_length, max_length+1, batch_size)
    num_repeats_batch = np.random.randint(min_reps, max_reps+1, batch_size)

    # Pads all the batches to have the same total sequence length.
    total_length_batch = sub_seq_length_batch * (num_repeats_batch + 1) + 3
    max_length_batch = np.max(total_length_batch)
    residual_length_batch = max_length_batch - total_length_batch

    obs_batch_shape = [max_length_batch, batch_size, full_obs_size]
    targ_batch_shape = [max_length_batch, batch_size, full_targ_size]
    mask_batch_trans_shape = [batch_size, max_length_batch]

    obs_tensors = []
    targ_tensors = []
    mask_tensors = []
    seq_len = []

    # Generates patterns for each batch element independently.
    for batch_index in range(batch_size):
      sub_seq_len = sub_seq_length_batch[batch_index]
      num_reps = num_repeats_batch[batch_index]

      # The observation pattern is a sequence of random binary vectors.
      obs_pattern_shape = [sub_seq_len, num_bits]
      obs_pattern = np.random.randint(0,2,obs_pattern_shape).astype('float32')

      # The target pattern is the observation pattern repeated n times.
      # Some reshaping is required to accomplish the tiling.
      targ_pattern_shape = [sub_seq_len * num_reps, num_bits]
      flat_obs_pattern = np.reshape(obs_pattern, [-1])
      flat_targ_pattern = np.tile(flat_obs_pattern, [num_reps])
      targ_pattern = np.reshape(flat_targ_pattern, targ_pattern_shape)

      # Expand the obs_pattern to have two extra channels for flags.
      # Concatenate start flag and num_reps flag to the sequence.
      obs_flag_channel_pad = np.zeros([sub_seq_len, 2])
      obs_start_flag = np.zeros([1, full_obs_size])
      obs_start_flag[0, start_end_flag_idx] = 1
      num_reps_flag = np.zeros([1, full_obs_size])
      num_reps_flag[0, num_repeats_channel_idx] = self._normalise(float(num_reps))

      # note the concatenation dimensions.
      obs = np.concatenate([obs_pattern, obs_flag_channel_pad], 1)
      obs = np.concatenate([obs_start_flag, obs], 0)
      obs = np.concatenate([obs, num_reps_flag], 0)

      # Now do the same for the targ_pattern (it only has one extra channel).
      targ_flag_channel_pad = np.zeros([sub_seq_len * num_reps, 1])
      targ_end_flag = np.zeros([1, full_targ_size])
      targ_end_flag[0, start_end_flag_idx] = 1
      targ = np.concatenate([targ_pattern, targ_flag_channel_pad], 1)
      targ = np.concatenate([targ, targ_end_flag], 0)

      # Concatenate zeros at end of obs and begining of targ.
      # This aligns them s.t. the target begins as soon as the obs ends.
      obs_end_pad = np.zeros([sub_seq_len * num_reps + 1, full_obs_size])
      targ_start_pad = np.zeros([sub_seq_len + 2, full_targ_size])

      # The mask is zero during the obs and one during the targ.
      mask_off = np.zeros([sub_seq_len + 2])
      mask_on = np.ones([sub_seq_len * num_reps + 1])

      obs = np.concatenate([obs, obs_end_pad], 0)
      targ = np.concatenate([targ_start_pad, targ], 0)
      mask = np.concatenate([mask_off, mask_on], 0)

      obs_tensors.append(obs)
      targ_tensors.append(targ)
      mask_tensors.append(mask)
      seq_len.append(mask.shape[0])

    # End the loop over batch index.
    # Compute how much zero padding is needed to make tensors sequences
    # the same length for all batch elements.
    residual_obs_pad = [
        np.zeros([residual_length_batch[i], full_obs_size])
        for i in range(batch_size)
    ]
    residual_targ_pad = [
        np.zeros([residual_length_batch[i], full_targ_size])
        for i in range(batch_size)
    ]
    residual_mask_pad = [
        np.zeros([residual_length_batch[i]]) for i in range(batch_size)
    ]

    # Concatenate the pad to each batch element.
    obs_tensors = [
        np.concatenate([o, p], 0) for o, p in zip(obs_tensors, residual_obs_pad)
    ]
    targ_tensors = [
        np.concatenate([t, p], 0) for t, p in zip(targ_tensors, residual_targ_pad)
    ]
    mask_tensors = [
        np.concatenate([m, p], 0) for m, p in zip(mask_tensors, residual_mask_pad)
    ]

    # Concatenate each batch element into a single tensor.
    obs = np.transpose(np.reshape(np.concatenate(obs_tensors, 1), obs_batch_shape), (1,0,2))
    targ = np.transpose(np.reshape(np.concatenate(targ_tensors, 1), targ_batch_shape), (1,0,2))
    mask = np.reshape(np.concatenate(mask_tensors, 0), mask_batch_trans_shape)
    seq_len = np.array(seq_len)
    return DatasetTensors(obs, targ, mask, seq_len)

  def cost(self, logits, targ, mask):
    return masked_sigmoid_cross_entropy(
        logits,
        targ,
        mask,
        time_average=self.time_average_cost,
        log_prob_in_bits=self.log_prob_in_bits)

  def to_human_readable(self, data, model_output=None, whole_batch=False):
    obs = data.observations
    unnormalised_num_reps_flag = self._unnormalise(obs[:,:,-1:]).round()
    obs = np.concatenate([obs[:,:,:-1], unnormalised_num_reps_flag], axis=2)
    data = data._replace(observations=obs)
    return bitstring_readable(data, self.batch_size, model_output, whole_batch)


if __name__ == '__main__':
  rcopy = RepeatCopy(num_bits=6,
      batch_size=2,
      min_length=1,
      max_length=2,
      min_repeats=1,
      max_repeats=2,
      norm_max=10,
      log_prob_in_bits=False,
      time_average_cost=False)
  tmp = rcopy()
  print(tmp.observations.shape, tmp.target.shape, tmp.mask.shape, tmp)