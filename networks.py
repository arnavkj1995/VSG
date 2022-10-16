import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import tools

class StochasticRSSM(tools.BaseClass):

    def __init__(
        self,
        stoch=30,
        hidden=200,
        layers_input=1,
        layers_output=1,
        rec_depth=1,
        shared=False,
        discrete=False,
        act=tf.nn.elu,
        mean_act="none",
        std_act="softplus",
        temp_post=True,
        min_std=0.1,
        cell="keras",
    ):
        super().__init__()
        self._stoch = stoch
        self._hidden = hidden
        self._min_std = min_std
        self._layers_input = layers_input
        self._layers_output = layers_output
        self._rec_depth = rec_depth
        self._shared = shared
        self._discrete = discrete
        self._act = act
        self._mean_act = mean_act
        self._std_act = std_act
        self._temp_post = temp_post
        self._embed = None

        if cell == "sgru":
            self._cell = StochasticGRUCell(self._stoch, return_output=False)
            self._is_cell_stochastic = True
        elif cell == 'sgru_layernorm':
            self._cell = StochasticGRUCell(self._stoch, norm=True, return_output=False)
            self._is_cell_stochastic = True
        else:
            raise NotImplementedError(cell)

    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        # print (' in inital ', dtype)
        if self._discrete:
            state = dict(
                logit=tf.ones([batch_size, self._stoch, self._discrete], dtype)
                / self._discrete,
                stoch=tf.ones([batch_size, self._stoch, self._discrete], dtype)
                / self._discrete,
                cand=self._cell.get_initial_state(None, batch_size, dtype),
            )
            # state['stoch'][:,:,0] = 1.0
        else:
            state = dict(
                mean=tf.zeros([batch_size, self._stoch], dtype),
                std=tf.zeros([batch_size, self._stoch], dtype),
                stoch=tf.zeros([batch_size, self._stoch], dtype),
                cand=self._cell.get_initial_state(None, batch_size, dtype)
            )

        if self._is_cell_stochastic:
            state["u_sample"] = tf.zeros([batch_size, self._stoch], dtype)
            state["u_logit"] = tf.zeros([batch_size, self._stoch], dtype)

        return state

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = stoch.shape[:-2] + [self._stoch * self._discrete]
            stoch = tf.reshape(stoch, shape)
        return stoch

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            logit = tf.cast(logit, tf.float32)
            dist = tools.OneHotDist(probs=logit)
            if dtype != tf.float32:
                dist = tools.DtypeDist(dist, dtype or state["logit"].dtype)
        else:
            mean, std = state["mean"], state["std"]
            if dtype:
                mean = tf.cast(mean, dtype)
                std = tf.cast(std, dtype)
            dist = tfd.Normal(mean, std)
        return dist

    def sample_dist(self, stats, sample=True):
        if sample:
            return self.get_dist(stats).sample()
        else:
            return self.get_dist(stats).mode()

    @tf.function
    def obs_step(self, prev_state, prev_action, embed, sample=True):
        if not self._embed:
            self._embed = embed.shape[-1]
        prior = self.img_step(prev_state, prev_action, None, sample)
        if self._shared:
            post = self.img_step(prev_state, prev_action, embed, sample)
        else:
            if self._temp_post:
                x = tf.concat([prior["cand"], embed], -1)
            else:
                x = embed
            for i in range(self._layers_output):
                x = self.get(f"obi{i}", tfkl.Dense, self._hidden, self._act)(x)

            stats = self._suff_stats_layer("obs", x)

            if self._is_cell_stochastic:
                if self._discrete:
                    stats["logit"] = (
                        tf.expand_dims(prior["u_logit"], axis=-1)
                        * tf.nn.softmax(stats["logit"], axis=-1)
                        + (1.0 - tf.expand_dims(prior["u_logit"], axis=-1))
                        * prev_state["stoch"]
                    )
                    output = self.sample_dist(stats, sample)
                else:
                    stoch = self.sample_dist(stats, sample)
                    output = (
                        prior["u_sample"] * stoch
                        + (1.0 - prior["u_sample"]) * prev_state["stoch"]
                    )

                post = {"stoch": output, "cand": prior["cand"], **stats}
                post["u_sample"] = prior["u_sample"]
                post["u_logit"] = prior["u_logit"]
            else:
                stoch = self.sample_dist(stats, sample)
                post = {"stoch": stoch, "cand": prior["cand"], **stats}

        return post, prior

    @tf.function
    def img_step(self, prev_state, prev_action, embed=None, sample=True):
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
            prev_stoch = tf.reshape(prev_stoch, shape)
        if self._shared:
            if embed is None:
                shape = prev_action.shape[:-1] + [self._embed]
                embed = tf.zeros(shape, prev_action.dtype)
            x = tf.concat([prev_stoch, prev_action, embed], -1)
        else:
            x = tf.concat([prev_stoch, prev_action], -1)
        for i in range(self._layers_input):
            x = self.get(f"ini{i}", tfkl.Dense, self._hidden, self._act)(x)

        for _ in range(self._rec_depth):
            if self._is_cell_stochastic:
                deter = [prev_stoch, prev_state["u_sample"], prev_state["u_logit"]]
            else:
                deter = [prev_state["stoch"]]
            x, deter = self._cell(x, deter)

        for i in range(self._layers_output):
            x = self.get(f"imo{i}", tfkl.Dense, self._hidden, self._act)(x)

        stats = self._suff_stats_layer("ims", x)

        if self._is_cell_stochastic:
            if self._discrete:
                stats["logit"] = (
                    tf.expand_dims(deter[2], axis=-1)
                    * tf.nn.softmax(stats["logit"], axis=-1)
                    + (1.0 - tf.expand_dims(deter[2], axis=-1)) * prev_state["stoch"]
                )
                output = self.sample_dist(stats, sample)
            else:
                stoch = self.sample_dist(stats, sample)
                output = deter[1] * stoch + (1.0 - deter[1]) * prev_stoch

            prior = {"stoch": output, "cand": deter[0], **stats}
            prior["u_sample"] = deter[1]
            prior["u_logit"] = deter[2]
        else:
            stoch = self.sample_dist(stats, sample)
            prior = {"stoch": stoch, "cand": deter[0], **stats}

        return prior

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
            logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
            mean, std = tf.split(x, 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * tf.math.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: tf.nn.softplus(std),
                "abs": lambda: tf.math.abs(std + 1),
                "sigmoid": lambda: tf.nn.sigmoid(std),
                "sigmoid2": lambda: 2 * tf.nn.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

class RSSM(tools.BaseClass):

  def __init__(
      self, stoch=30, deter=200, hidden=200, layers_input=1, layers_output=1,
      rec_depth=1, shared=False, discrete=False, act=tf.nn.elu,
      mean_act='none', std_act='softplus', temp_post=True, min_std=0.1,
      cell='keras'):
    super().__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._min_std = min_std
    self._layers_input = layers_input
    self._layers_output = layers_output
    self._rec_depth = rec_depth
    self._shared = shared
    self._discrete = discrete
    self._act = act
    self._mean_act = mean_act
    self._std_act = std_act
    self._temp_post = temp_post
    self._embed = None
    if cell == 'gru':
      self._cell = GRUCell(self._deter)
      self._is_cell_stochastic = False
    elif cell == 'gru_layer_norm':
      self._cell = GRUCell(self._deter, norm=True)
      self._is_cell_stochastic = False
    elif cell == 'sgru':
      self._cell = StochasticGRUCell(self._deter)
      self._is_cell_stochastic = True
    elif cell == 'sgru_layernorm':
      self._cell = StochasticGRUCell(self._deter, norm=True)
      self._is_cell_stochastic = True
    else:
      raise NotImplementedError(cell)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))

    if self._is_cell_stochastic:
      state["u_sample"] = tf.zeros([batch_size, self._deter], dtype)
      state["u_logit"] = tf.zeros([batch_size, self._deter], dtype)

    return state

  def get_feat(self, state):
    stoch = state['stoch']
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return tf.concat([stoch, state['deter']], -1)

  def get_dist(self, state, dtype=None):
    if self._discrete:
      logit = state['logit']
      logit = tf.cast(logit, tf.float32)
      dist = tools.OneHotDist(logit)
      if dtype != tf.float32:
        dist = tools.DtypeDist(dist, dtype or state['logit'].dtype)
    else:
      mean, std = state['mean'], state['std']
      if dtype:
        mean = tf.cast(mean, dtype)
        std = tf.cast(std, dtype)
      dist = tfd.Normal(mean, std)
    return dist

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      mean = {
          'none': lambda: mean,
          'tanh5': lambda: 5.0 * tf.math.tanh(mean / 5.0),
      }[self._mean_act]()
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'abs': lambda: tf.math.abs(std + 1),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, sample=True):
    if not self._embed:
      self._embed = embed.shape[-1]
    prior = self.img_step(prev_state, prev_action, None, sample)
    if self._shared:
      post = self.img_step(prev_state, prev_action, embed, sample)
    else:
      if self._temp_post:
        x = tf.concat([prior['deter'], embed], -1)
      else:
        x = embed
      for i in range(self._layers_output):
        x = self.get(f'obi{i}', tfkl.Dense, self._hidden, self._act)(x)
      stats = self._suff_stats_layer('obs', x)
      if sample:
        stoch = self.get_dist(stats).sample()
      else:
        stoch = self.get_dist(stats).mode()
      post = {'stoch': stoch, 'deter': prior['deter'], **stats}

      if self._is_cell_stochastic:
        post["u_sample"] = prior["u_sample"]
        post["u_logit"] = prior["u_logit"]
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, embed=None, sample=True):
    prev_stoch = prev_state['stoch']
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
    if self._shared:
      if embed is None:
        shape = prev_action.shape[:-1] + [self._embed]
        embed = tf.zeros(shape, prev_action.dtype)
      x = tf.concat([prev_stoch, prev_action, embed], -1)
    else:
      x = tf.concat([prev_stoch, prev_action], -1)
    for i in range(self._layers_input):
      x = self.get(f'ini{i}', tfkl.Dense, self._hidden, self._act)(x)
    for _ in range(self._rec_depth):
      deter = prev_state['deter']
      x, deter = self._cell(x, [deter])
    for i in range(self._layers_output):
      x = self.get(f'imo{i}', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('ims', x)
    if sample:
      stoch = self.get_dist(stats).sample()
    else:
      stoch = self.get_dist(stats).mode()

    prior = {'stoch': stoch, 'deter': deter[0], **stats}
    if self._is_cell_stochastic:
        prior["u_sample"] = deter[1]
        prior["u_logit"]  = deter[2]

    return prior

class ConvEncoder(tools.Module):

  def __init__(
      self, depth=32, act=tf.nn.relu, kernels=(4, 4, 4, 4)):
    self._act = act
    self._depth = depth
    self._kernels = kernels

  def __call__(self, obs):
    x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** i * self._depth
      x = self._act(self.get(f'h{i}', tfkl.Conv2D, depth, kernel, 2)(x))
    x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
    shape = tf.concat([tf.shape(obs['image'])[:-3], [x.shape[-1]]], 0)
    return tf.reshape(x, shape)

class ConvDecoder(tools.Module):

  def __init__(
      self, depth=32, act=tf.nn.relu, shape=(64, 64, 3), kernels=(5, 5, 6, 6),
      thin=True):
    self._act = act
    self._depth = depth
    self._shape = shape
    self._kernels = kernels
    self._thin = thin

  def __call__(self, features, dtype=None):
    ConvT = tfkl.Conv2DTranspose
    if self._thin:
      x = self.get('hin', tfkl.Dense, 32 * self._depth, None)(features)
      x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
    else:
      x = self.get('hin', tfkl.Dense, 128 * self._depth, None)(features)
      x = tf.reshape(x, [-1, 2, 2, 32 * self._depth])
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** (len(self._kernels) - i - 2) * self._depth
      act = self._act
      if i == len(self._kernels) - 1:
        depth = self._shape[-1]
        act = None
      x = self.get(f'h{i}', ConvT, depth, kernel, 2, activation=act)(x)
    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    if dtype:
      mean = tf.cast(mean, dtype)
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))

class DenseHead(tools.Module):

  def __init__(
      self, shape, layers, units, act=tf.nn.elu, dist='normal', std=1.0):
    self._shape = (shape,) if isinstance(shape, int) else shape
    self._layers = layers
    self._units = units
    self._act = act
    self._dist = dist
    self._std = std

  def __call__(self, features, dtype=None):
    x = features
    for index in range(self._layers):
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    mean = self.get(f'hmean', tfkl.Dense, np.prod(self._shape))(x)
    mean = tf.reshape(mean, tf.concat(
        [tf.shape(features)[:-1], self._shape], 0))
    if self._std == 'learned':
      std = self.get(f'hstd', tfkl.Dense, np.prod(self._shape))(x)
      std = tf.nn.softplus(std) + 0.01
      std = tf.reshape(std, tf.concat(
          [tf.shape(features)[:-1], self._shape], 0))
    else:
      std = self._std
    if dtype:
      mean, std = tf.cast(mean, dtype), tf.cast(std, dtype)
    if self._dist == 'normal':
      return tfd.Independent(tfd.Normal(mean, std), len(self._shape))
    if self._dist == 'huber':
      return tfd.Independent(
          tools.UnnormalizedHuber(mean, std, 1.0), len(self._shape))
    if self._dist == 'binary':
      return tfd.Independent(tfd.Bernoulli(mean), len(self._shape))
    raise NotImplementedError(self._dist)

class ActionHead(tools.Module):

  def __init__(
      self, size, layers, units, act=tf.nn.elu, dist='trunc_normal',
      init_std=0.0, min_std=0.1, action_disc=5, temp=0.1, outscale=0):
    # assert min_std <= 2
    self._size = size
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act
    self._min_std = min_std
    self._init_std = init_std
    self._action_disc = action_disc
    self._temp = temp() if callable(temp) else temp
    self._outscale = outscale

  def __call__(self, features, dtype=None):
    x = features
    for index in range(self._layers):
      kw = {}
      if index == self._layers - 1 and self._outscale:
        kw['kernel_initializer'] = tf.keras.initializers.VarianceScaling(
            self._outscale)
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act, **kw)(x)
    if self._dist == 'tanh_normal':
      # https://www.desmos.com/calculator/rcmcf5jwe7
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      mean, std = tf.split(x, 2, -1)
      mean = tf.tanh(mean)
      std = tf.nn.softplus(std + self._init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
      dist = tools.DtypeDist(dist, dtype)
      dist = tfd.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'tanh_normal_5':
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      mean, std = tf.split(x, 2, -1)
      mean = 5 * tf.tanh(mean / 5)
      std = tf.nn.softplus(std + 5) + 5
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
      dist = tools.DtypeDist(dist, dtype)
      dist = tfd.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'normal':
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      mean, std = tf.split(x, 2, -1)
      std = tf.nn.softplus(std + self._init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tools.DtypeDist(dist, dtype)
      dist = tfd.Independent(dist, 1)
    elif self._dist == 'normal_1':
      mean = self.get(f'hout', tfkl.Dense, self._size)(x)
      if dtype:
        mean = tf.cast(mean, dtype)
      dist = tfd.Normal(mean, 1)
      dist = tools.DtypeDist(dist, dtype)
      dist = tfd.Independent(dist, 1)
    elif self._dist == 'trunc_normal':
      # https://www.desmos.com/calculator/mmuvuhnyxo
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      x = tf.cast(x, tf.float32)
      mean, std = tf.split(x, 2, -1)
      mean = tf.tanh(mean)
      std = 2 * tf.nn.sigmoid(std / 2) + self._min_std
      dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
      dist = tools.DtypeDist(dist, dtype)
      dist = tfd.Independent(dist, 1)
    elif self._dist == 'onehot':
      x = self.get(f'hout', tfkl.Dense, self._size)(x)
      x = tf.cast(x, tf.float32)
      dist = tools.OneHotDist(x, dtype=dtype)
      dist = tools.DtypeDist(dist, dtype)
    elif self._dist == 'onehot_gumble':
      x = self.get(f'hout', tfkl.Dense, self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      temp = self._temp
      dist = tools.GumbleDist(temp, x, dtype=dtype)
    else:
      raise NotImplementedError(self._dist)
    return dist

class GRUCell(tf.keras.layers.AbstractRNNCell):
  def __init__(self, size, norm=False, act=tf.tanh, update_bias=-1, **kwargs):
    super().__init__()
    self._size = size
    self._act = act
    self._norm = norm
    self._update_bias = update_bias
    self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
    if norm:
      self._norm = tfkl.LayerNormalization(dtype=tf.float32)

  @property
  def state_size(self):
    return self._size

  def call(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(tf.concat([inputs, state], -1))
    if self._norm:
      dtype = parts.dtype
      parts = tf.cast(parts, tf.float32)
      parts = self._norm(parts)
      parts = tf.cast(parts, dtype)
    reset, cand, update = tf.split(parts, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = self._act(reset * cand)
    update = tf.nn.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]

class StochasticGRUCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, size, norm=False, act=tf.tanh, update_bias=-1, return_output=True, **kwargs):
        super().__init__()
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
        self._return_output = return_output
        if norm:
            self._norm = tfkl.LayerNormalization(dtype=tf.float32)

    @property
    def state_size(self):
        return self._size

    def call(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(tf.concat([inputs, state], -1))

        if self._norm:
            dtype = parts.dtype
            parts = tf.cast(parts, tf.float32)
            parts = self._norm(parts)
            parts = tf.cast(parts, dtype)

        reset, cand, update = tf.split(parts, 3, -1)
        reset = tf.nn.sigmoid(reset)
        cand = self._act(reset * cand)
        update_p = tf.nn.sigmoid(update + self._update_bias)
        update = tf.cast(
            tfd.Independent(tools.BernoulliDist(probs=update_p), 1).sample(), tf.float16
        )
        if self._return_output:
          output = update * cand + (1 - update) * state
          return output, [output, update, update_p]
        else:
          return cand, [cand, update, update_p]
          