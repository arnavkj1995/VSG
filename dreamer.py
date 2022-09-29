import argparse
import collections
import functools
import os
import pathlib
import sys
import warnings
import json
import h5py

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*TensorFloat-32 matmul/conv*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import ruamel.yaml as yaml
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec
from tqdm import tqdm
import wandb

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import wrappers

class Dreamer(tools.Module):

  def __init__(self, config, logger, dataset):
    self._config = config
    self._logger = logger
    self._float = prec.global_policy().compute_dtype
    self._should_log = tools.Every(config.log_every)
    self._should_log_train_openl = tools.Every(10 * config.log_every)
    self._should_train = tools.Every(config.train_every)
    self._should_pretrain = tools.Once()
    self._should_reset = tools.Every(config.reset_every)
    self._should_expl = tools.Until(int(
        config.expl_until / config.action_repeat))
    self._metrics = collections.defaultdict(tf.metrics.Mean)
    with tf.device('cpu:0'):
      self._step = tf.Variable(count_steps(config.traindir), dtype=tf.int64)
    # Schedules.
    config.actor_entropy = (
        lambda x=config.actor_entropy: tools.schedule(x, self._step))
    config.actor_state_entropy = (
        lambda x=config.actor_state_entropy: tools.schedule(x, self._step))
    config.imag_gradient_mix = (
        lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
    self._dataset = iter(dataset)
    self._wm = models.WorldModel(self._step, config)
    self._task_behavior = models.ImagBehavior(
        config, self._wm, config.behavior_stop_grad)
    reward = lambda f, s, a: self._wm.heads['reward'](f).mode()
    self._expl_behavior = dict(
        greedy=lambda: self._task_behavior,
        random=lambda: expl.Random(config),
        plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
    )[config.expl_behavior]()
    # Train step to initialize variables including optimizer statistics.
    self._train(next(self._dataset))

  def __call__(self, obs, reset, state=None, training=True):
    step = self._step.numpy().item()
    if self._should_reset(step):
        state = None
    if state is not None and reset.any():
        mask = tf.cast(1 - reset, self._float)[:, None]
        state = tf.nest.map_structure(lambda x: x * mask, state)
    if training and self._should_train(step):
        steps = (
            self._config.pretrain
            if self._should_pretrain()
            else self._config.train_steps
        )
        for _ in range(steps):
            self._train(next(self._dataset))
        if self._should_log(step):
            for name, mean in self._metrics.items():
                self._logger.scalar(name, float(mean.result()))
                mean.reset_states()
            openl = self._wm.video_pred(next(self._dataset))
            self._logger.video("train_openl", openl)

            # openl, update_gate = self._wm.video_pred(next(self._dataset))
            # update_gate_img = update_gate[..., None]
            # print("Train update_gate_img: ", update_gate_img.shape)
            # self._logger.image("train_update_gate_img", update_gate_img)
            # self._logger.video("train_openl", openl)

            self._logger.write(fps=True)
    policy_output, state = self._policy(obs, state, training)
    if training:
        self._step.assign_add(len(reset))
        self._logger.step = self._config.action_repeat * self._step.numpy().item()
    return policy_output, state

  @tf.function
  def _policy(self, obs, state, training):
    if state is None:
      batch_size = len(obs['image'])
      latent = self._wm.dynamics.initial(len(obs['image']))
      action = tf.zeros((batch_size, self._config.num_actions), self._float)
    else:
      latent, action = state
    embed = self._wm.encoder(self._wm.preprocess(obs))
    latent, _ = self._wm.dynamics.obs_step(
        latent, action, embed, self._config.collect_dyn_sample)
    if self._config.eval_state_mean:
      latent['stoch'] = latent['mean']
    feat = self._wm.dynamics.get_feat(latent)
    if not training:
      actor = self._task_behavior.actor(feat)
      action = actor.mode()
    elif self._should_expl(self._step):
      actor = self._expl_behavior.actor(feat)
      action = actor.sample()
    else:
      actor = self._task_behavior.actor(feat)
      action = actor.sample()
    try:
      logprob = actor.log_prob(action)
    except Exception:
      logprob = actor.log_prob(tf.cast(action, tf.float32))
    if self._config.actor_dist == 'onehot_gumble':
      action = tf.cast(
          tf.one_hot(tf.argmax(action, axis=-1), self._config.num_actions),
          action.dtype)
    action = self._exploration(action, training)
    policy_output = {'action': action, 'logprob': logprob}
    state = (latent, action)
    return policy_output, state

  def _exploration(self, action, training):
    amount = self._config.expl_amount if training else self._config.eval_noise
    if amount == 0:
      return action
    amount = tf.cast(amount, self._float)
    if 'onehot' in self._config.actor_dist:
      probs = amount / self._config.num_actions + (1 - amount) * action
      return tools.OneHotDist(probs=probs).sample()
    else:
      return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
    raise NotImplementedError(self._config.action_noise)

  @tf.function
  def _train(self, data):
    tqdm.write('Tracing train function.')
    metrics = {}
    post, context, mets = self._wm.train(data)
    metrics.update(mets)
    start = post
    if self._config.pred_discount:  # Last step could be terminal.
      start = {k: v[:, :-1] for k, v in post.items()}
      context = {k: v[:, :-1] for k, v in context.items()}
    reward = lambda f, s, a: self._wm.heads['reward'](
        self._wm.dynamics.get_feat(s)).mode()
    metrics.update(self._task_behavior.train(start, reward)[-1])
    if self._config.expl_behavior != 'greedy':
      if self._config.pred_discount:
        data = {k: v[:, :-1] for k, v in data.items()}
      mets = self._expl_behavior.train(start, context, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    for name, value in metrics.items():
      self._metrics[name].update_state(value)

def count_steps(folder):
    return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))

def make_dataset(episodes, config):
    example = episodes[next(iter(episodes.keys()))]
    types = {k: v.dtype for k, v in example.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in example.items()}
    generator = lambda: tools.sample_episodes(
        episodes, config.batch_length, config.oversample_ends)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(10)
    return dataset

def make_env(config, logger, mode, train_eps, eval_eps):
    suite, task = config.task.split("_", 1)

    if suite == "dmc":
        env = wrappers.DeepMindControl(task, config.action_repeat, config.size)
        env = wrappers.NormalizeActions(env)

    elif suite == "bringbackshapes":
        print (' before env', config.time_limit)
        env = wrappers.BringBackShapes2D(config.action_repeat, False, config.max_distractors,
                                       config.max_objects, config.variable_num_objects,
                                       config.variable_num_distractors, config.variable_goal_position,
                                       config.agent_view_size, config.arena_scale)
        env = wrappers.NormalizeActions(env)


    else:
        raise NotImplementedError(suite)

    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    callbacks = [
        functools.partial(process_episode, config, logger, mode, train_eps, eval_eps)
    ]

    if suite == 'crafter':
        callbacks += [functools.partial(find_success_rate, logger)]

    env = wrappers.CollectDataset(env, callbacks)
    env = wrappers.RewardObs(env)
    return env

def find_success_rate(logger, episode):
    filename = logger._logdir / 'stats.jsonl'
    save_success = logger._logdir / 'success_rates.jsonl'
    crafter_score_txt = logger._logdir / 'crafter_score.txt'
    stats = []
    for ep in open(filename, 'r'):
        stat_dict = json.loads(ep)
        ac_dict = {k: v for k, v in stat_dict.items() if k.startswith('achievement_')}
        stats.append(ac_dict)
    ac_dict = {ac_k: [st_dict[ac_k] for st_dict in stats] for ac_k in sorted(stats[0].keys())}
    success_rates = {ac_key: 100 * (np.array(v) >= 1).mean() for ac_key, v in ac_dict.items()}
    percents = np.array(list(success_rates.values()))
    score = tools.compute_scores(percents)
    with open(save_success, 'w') as fp:
        json.dump(success_rates, fp)
    np.savetxt(crafter_score_txt, np.array([score]))
    logger.scalar('crafter_score', score)

def process_episode(config, logger, mode, train_eps, eval_eps, episode):
  directory = dict(train=config.traindir, eval=config.evaldir)[mode]
  cache = dict(train=train_eps, eval=eval_eps)[mode]
  filename = tools.save_episodes(directory, [episode], config.save_eps)[0]
  length = len(episode['reward']) - 1
  score = float(episode['reward'].astype(np.float64).sum())
  video = episode['image']
  if mode == 'eval':
    cache.clear()
  if mode == 'train' and config.dataset_size:
    total = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
      if total <= config.dataset_size - length:
        total += len(ep['reward']) - 1
      else:
        del cache[key]
    logger.scalar('dataset_size', total + length)
  cache[str(filename)] = episode
  tqdm.write(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_length', length)
  logger.scalar(f'{mode}_episodes', len(cache))
  if mode == 'eval' or config.expl_gifs:
    logger.video(f'{mode}_policy', video[None])
  if mode =='train':
    logger.write()

def main(config):
    folder_name = f"{config.task}/{config.id}/seed{config.seed}"
    logdir = pathlib.Path(config.logdir).expanduser() / folder_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {logdir}")

    if not config.only_eval:
      config_path = logdir / "configs.yaml"
      command_args = dict(defaults=vars(config))
      with open(config_path, "w") as f:
          yaml.dump(command_args, f, default_flow_style=False)

      script_path = logdir / "script.sh"
      with open(script_path, "w") as f:
          f.write("#!/bin/bash")
          f.write("\n")
          f.write("python ")
          f.write(" ".join(sys.argv))

    config.traindir = (
        pathlib.Path(config.traindir).expanduser() if config.traindir != "" else logdir / "train_eps"
    )
    config.evaldir = (
        pathlib.Path(config.evaldir).expanduser() if config.evaldir != "" else logdir / "eval_eps"
    )
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    config.prefill //= config.action_repeat
    config.act = getattr(tf.nn, config.act)

    if config.debug:
        tf.config.experimental_run_functions_eagerly(True)
        os.environ["WANDB_MODE"] = "dryrun"

    message = "No GPU found. To actually train on CPU remove this assert."
    assert tf.config.experimental.list_physical_devices("GPU"), message
    if config.gpu_growth:
        for gpu in tf.config.experimental.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        prec.set_policy(prec.Policy("mixed_float16"))

    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step, config.only_eval)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps)
    train_envs = [make("train") for _ in range(config.envs)]
    eval_envs = [make("eval") for _ in range(config.envs)]
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    prefill = max(0, config.prefill - count_steps(config.traindir))
    print(f"Prefill dataset ({prefill} steps).")
    if hasattr(acts, "discrete"):
        random_actor = tools.OneHotDist(tf.zeros_like(acts.low))
    else:
        random_actor = tfd.Independent(tfd.Uniform(acts.low, acts.high), 1)

    def random_agent(o, d, s):
      action = [random_actor.sample() for _ in d]
      logprob = [random_actor.log_prob(a) for a in action]
      return {'action': action, 'logprob': logprob}, None

    tools.simulate(random_agent, train_envs, prefill)
    tools.simulate(random_agent, eval_envs, episodes=1)
    logger.write()
    logger.step = config.action_repeat * count_steps(config.traindir)

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = iter(make_dataset(eval_eps, config))
    print("Create Dataset")
    agent = Dreamer(config, logger, train_dataset)
    if (logdir / "variables.pkl").exists():
        agent.load(logdir / "variables.pkl")
        agent._should_pretrain._once = False
        print ('loaded pretrained model')

    print (' before train loop ', agent._step, count_steps(config.traindir))

    if config.mode == 'eval':
      tqdm.write("Start evaluation.")
      tqdm.write("Start evaluation openl.")

      video_pred = agent._wm.video_pred(next(eval_dataset))
      logger.video("eval_openl", video_pred)

      eval_policy = functools.partial(agent, training=False)
      tools.simulate(eval_policy, eval_envs, episodes=config.n_eval_eps)
      logger.write()
      exit
    elif config.mode == 'train':
      state = None
      initial = agent._step.numpy().item()
      pbar = tqdm(total=config.steps, initial=initial)
      while agent._step.numpy().item() < config.steps:
          logger.write()
          tqdm.write("Start evaluation.")
          tqdm.write("Start evaluation openl.")

          video_pred = agent._wm.video_pred(next(eval_dataset))
          logger.video("eval_openl", video_pred)

          eval_policy = functools.partial(agent, training=False)
          tools.simulate(eval_policy, eval_envs, episodes=config.n_eval_eps)
          logger.write()
          if config.only_eval:
            break
          tqdm.write("Start training.")
          state = tools.simulate(agent, train_envs, config.eval_every, state=state)
          agent.save(logdir / "variables.pkl")
          pbar.update(agent._step.numpy().item() - initial)
          initial = agent._step.numpy().item()

      pbar.close()
    else:
      raise NotImplementedError(config.mode)

    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
