import atexit
import functools
import sys
import threading
import traceback
import os

import gym
import numpy as np
from PIL import Image

import blox
import blox.mujoco

if 'MUJOCO_RENDERER' in os.environ:
  RENDERER = os.environ['MUJOCO_RENDERER']
else:
  RENDERER = 'glfw'


class DreamerEnv():
  LOCK = threading.Lock()

  def __init__(self, action_repeat, width=64):
    self._action_repeat = action_repeat
    self._width = width
    self._size = (self._width, self._width)

  @property
  def observation_space(self):
    shape = (3,) + self._size
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return space # gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      state = self._env.reset()
    return self._get_obs(state)

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(action)
      total_reward += reward
      if done:
        break
    obs = self._get_obs(state)
    return obs, total_reward, done, info

  def render(self, mode, height, width):
    # return self._env.render(mode)
    self._offscreen.render(self._width, self._width, -1)
    image = np.flip(self._offscreen.read_pixels(self._width, self._width)[0], 1)
    return image

  def _get_obs(self, state):
    self._offscreen.render(self._width, self._width, -1)
    image = np.flip(self._offscreen.read_pixels(self._width, self._width)[0], 1)
    image = np.transpose(image, (2, 0, 1))
    # return {'image': image, 'state': state}
    return image


class MetaWorld(DreamerEnv):
  def __init__(self, name, action_repeat, rand_goal=False, rand_hand=False, rand_obj=False, env_rew_scale=1.0, sparse_reward=True, width=64):
    super().__init__(action_repeat, width)
    from mujoco_py import MjRenderContext
    import metaworld.envs.mujoco.sawyer_xyz.v1 as sawyer
    import metaworld.envs.mujoco.sawyer_xyz.v2 as sawyerv2

    domain, task = name.split('_', 1)
    self._task_keys, task = parse(task, ['closeup', 'strict', 'hardinit', 'frontview1', 'frontview2'])
    frontview2 = False
    self.task_type = self.get_task_type(task)

    self.v2 = v2 = 'V2' in task
    kwargs = {}
    if self._task_keys['strict']:
      kwargs['strict_reward'] = True
    with self.LOCK:
      if not v2:
        self._env = getattr(sawyer, task)()
      else:
        self._env = getattr(sawyerv2, task)(**kwargs)
    self._env.random_init = False
    self._env.max_path_length = np.inf

    self._action_repeat = action_repeat
    self._rand_goal = rand_goal
    self._rand_hand = rand_hand
    self._rand_obj = rand_obj
    self._width = width
    self._size = (self._width, self._width)
    self.env_rew_scale = env_rew_scale
    self.rendered_goal = False
    self.sparse_reward = sparse_reward
    self._max_episode_steps = 150

    self._offscreen = MjRenderContext(self._env.sim, True, 0, RENDERER, True)
    if self._task_keys['closeup']:
      self._offscreen.cam.azimuth = 155
      self._offscreen.cam.elevation = -150
      self._offscreen.cam.distance = 0.9
      self._offscreen.cam.lookat[0] = 0.3
      self._offscreen.cam.lookat[1] = 0.55
    elif "SawyerStickPushEnv" in task and "zoom" in domain:
      self._offscreen.cam.azimuth = 220
      self._offscreen.cam.elevation = -140
      self._offscreen.cam.distance = 0.8
      self._offscreen.cam.lookat[0] = 0.2
      self._offscreen.cam.lookat[1] = 0.65
      self._offscreen.cam.lookat[2] = -0.1
    elif ("Hammer" in task or "Pull" in task) and "zoom1" in domain:
      blox.mujoco.set_camera(
        self._offscreen.cam, azimuth=220, elevation=-140, distance=0.8, lookat=[0.2, 0.65, -0.1])
    elif "SawyerHammerEnv" in task and "zoom2" in domain:
      blox.mujoco.set_camera(
        self._offscreen.cam, azimuth=300, elevation=-130, distance=0.8, lookat=[0.2, 0.65, -0.0])
    elif ("Hammer" in task) and "zoom3" in domain:
      blox.mujoco.set_camera(
        self._offscreen.cam, azimuth=350, elevation=-150, distance=0.8, lookat=[0.2, 0.65, -0.0])
    elif self._task_keys['frontview1']:
      blox.mujoco.set_camera(
        self._offscreen.cam, azimuth=90, elevation=22 + 180, distance=0.82, lookat=[0., 0.55, 0.])
    elif self._task_keys['frontview2']:
      blox.mujoco.set_camera(
        self._offscreen.cam, azimuth=90, elevation=40 + 180, distance=0.91, lookat=[0., 0.65, 0.1])
    elif frontview2:
      blox.mujoco.set_camera(
        self._offscreen.cam, azimuth=90, elevation=41 + 180, distance=0.61, lookat=[0., 0.55, 0.])
    # elif v2:
    #   self._offscreen.cam.azimuth = 120
    #   self._offscreen.cam.elevation = -165
    #   self._offscreen.cam.distance = 1.5
    #   self._offscreen.cam.lookat[0] = 0.5
    #   self._offscreen.cam.lookat[1] = 0
    #   self._offscreen.cam.lookat[2] = 0
    else:
      self._offscreen.cam.azimuth = 205
      self._offscreen.cam.elevation = -165
      self._offscreen.cam.distance = 2.6
      self._offscreen.cam.lookat[0] = 1.1
      self._offscreen.cam.lookat[1] = 1.1
      self._offscreen.cam.lookat[2] = -0.1


  def get_task_type(self, task):
    if 'SawyerReachEnv' in task:
      return 'reach'
    if 'SawyerButtoPressEnv' in task:
      return 'button'
    if 'SawyerPushEnv' in task:
      return 'push'
    if 'SawyerHammerEnv' in task:
      return 'hammer'
    if 'SawyerDoorCloseEnv' in task:
      return 'door'
    if 'SawyerDrawerCloseEnv' in task or 'SawyerDrawerOpenEnv' in task:
      return 'drawer'
    if 'SawyerWindowCloseEnv' in task or 'SawyerWindowOpenEnv' in task:
      return 'window'
    if 'bin' in task.lower():
      return 'pickbin'

  def reset(self):
    self.rendered_goal = False
    # Evaluation config
    if self.task_type == 'push':
      self._env.init_config['obj_init_pos'] = np.array([0., 0.55, 0.02])
      self._env.hand_init_pos = np.array([0., 0.55, 0.05])
    elif self.task_type == 'door':
      self._env.hand_init_pos = np.array([-0.4, 0.5, 0.2])
    elif self.task_type == 'drawer':
      self._env.hand_init_pos = self._env.obj_init_pos + np.array([0, -0.31, 0.25])
    elif self.task_type == 'window':
      self._env.hand_init_pos = self._env.obj_init_pos + np.array([0.2, -0.1, 0.05])


    if self._rand_obj:
      if self.task_type == 'push':
        obj_init_pos = np.random.uniform(
          (-0.2, 0.6, 0.02),
          (0.4, 0.9, 0.02),
          size=(3)
        )
        self._env.init_config['obj_init_pos'] = obj_init_pos
        # Initialize hand above object
        self._env.hand_init_pos = obj_init_pos
        # Initialize hand with fixed height, can be overridden by rand_hand
        self._env.hand_init_pos[2] = 0.05
      elif self.task_type == 'hammer':
        hammer_init_pos = np.random.uniform(
          (-0.1, 0.3, 0.04),
          (0.3, 0.7, 0.04),
          size=(3)
        )
        self._env.init_config['hammer_init_pos'] = hammer_init_pos
        # Initialize hand with fixed displacement, can be overridden by rand_hand
        self._env.hand_init_pos = hammer_init_pos + np.array([0., -0.1, 0.16])
      elif self.task_type == 'door':
        door_init_pos = - np.random.uniform(0, 1.3)
        self._env.init_config['door_init_pos'] = door_init_pos
      elif self.task_type == 'drawer':
        drawer_init_pos = np.random.uniform(0, 0.15)
        self._env.init_config['drawer_init_pos'] = drawer_init_pos
        # -0.16 is the handle protrusion
        self._env.hand_init_pos = self._env.obj_init_pos + np.array([0, -0.16-drawer_init_pos, 0.15])
      elif self.task_type == 'window':
        slider_init_pos = np.random.uniform(0, 0.2)
        self._env.init_config['slider_init_pos'] = slider_init_pos
        self._env.hand_init_pos = self._env.obj_init_pos + np.array([slider_init_pos, -0.1, 0.05])

    if self._rand_hand:
      if self.task_type == 'push':
        self._env.hand_init_pos[2] = np.random.uniform(0.05, 0.2)
      elif self.task_type == 'door':
        self._env.hand_init_pos = np.random.uniform((-0.5, 0.4, 0.05), (0, 0.8, 0.5), size=(3))
      elif self.task_type == 'drawer':
        self._env.hand_init_pos += np.random.uniform((-0.15, -0.15, 0), (0.15, 0.15, 0.3), size=(3))
      elif self.task_type == 'window':
        self._env.hand_init_pos += np.random.uniform((-0.15, -0.3, -0.15), (0.15, 0, 0.15), size=(3))
      else:
        self._env.hand_init_pos = np.random.uniform(
          self._env.hand_low,
          self._env.hand_high,
          size=(self._env.hand_low.size)
        )

    if self._rand_goal:
      self._env.goal = np.random.uniform(
        self._env.goal_space.low,
        self._env.goal_space.high,
        size=(self._env.goal_space.low.size)
      )
    return super().reset()

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(action)
      total_reward += reward if not self.sparse_reward else info['success'] # min(reward, 100000)
      if done:
        break
    obs = self._get_obs(state)
    total_reward *= self.env_rew_scale
    return obs, total_reward, done, info

  def render_goal(self):
    if self.rendered_goal:
      return self.rendered_goal_obj

    obj_init_pos_temp = self._env.init_config['obj_init_pos'].copy()
    if self.task_type == 'push':
      self._env.init_config['obj_init_pos'] = self._env.goal
      self._env.obj_init_pos = self._env.goal
    self._env.hand_init_pos = self._env.goal
    # self._env.hand_init_pos = np.array([0, .6, .0]) # Low
    # self._env.hand_init_pos = np.array([0, .6, .4]) # High
    # self._env.hand_init_pos = np.array([0, .8, .2]) # Forward
    # self._env.hand_init_pos = np.array([0, .4, .2]) # Backward
    # self._env.hand_init_pos = np.array([-.2, .6, .2]) # Left?
    # self._env.hand_init_pos = np.array([-.2, .6, .2]) # Right?
    self._env.reset_model()
    if not self.v2:
      error = np.sum((self._env.init_fingerCOM - self._env.goal) ** 2)
      if error > 0.01:
        print('WARNING: the goal is not rendered correctly')
    action = np.zeros(self._env.action_space.low.shape)
    state, reward, done, info = self._env.step(action)
    goal_obs = MetaWorld._get_obs(self, state)
    self._env.hand_init_pos = self._env.init_config['hand_init_pos']
    self._env.init_config['obj_init_pos'] = obj_init_pos_temp
    self._env.obj_init_pos = self._env.init_config['obj_init_pos']
    self._env.reset()

    self.rendered_goal = True
    self.rendered_goal_obj = goal_obs
    return goal_obs

  @property
  def reward_range(self):
    return (0, 1)

  @property
  def metadata(self):
    return None


class MetaWorldVis(MetaWorld):
  def __init__(self, name, action_repeat, width):
    super().__init__(name, action_repeat)
    self._width = width
    self._size = (self._width, self._width)

  def render_state(self, state):
    assert (len(state.shape) == 1)
    # Save init configs
    hand_init_pos = self._env.hand_init_pos
    init_config = self._env.init_config.copy()
    # Render state
    if self.task_type == 'push':
      hand_pos, obj_pos, hand_to_goal = np.split(state, 3)
      self._env.hand_init_pos = hand_pos
      self._env.init_config['obj_init_pos'] = obj_pos
    if self.task_type == 'hammer':
      hand_pos, obj_pos, _, _ = np.split(state, 4)
      self._env.hand_init_pos = hand_pos
      self._env.init_config['hammer_init_pos'] = obj_pos
      self._env.init_config['nail_init_pos'] = state[7] - 0.64
      # 0 - out, 0.1 - in
      # 0.64 - out, 0.74 - in
    self._env.reset_model()
    obs = self._get_obs(state)
    # Revert environment
    self._env.hand_init_pos = hand_init_pos
    self._env.init_config = init_config
    self._env.reset()
    return obs['image']

  def render_states(self, states):
    assert (len(states.shape) == 2)
    imgs = []
    for s in states:
      img = self.render_state(s)
      imgs.append(img)
    return np.array(imgs)


class DeepMindControl:

  def __init__(self, name, size=(64, 64), camera=None):
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

  def step(self, action):
    time_step = self._env.step(action)
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    reward = time_step.reward or 0
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)


class Collect:

  def __init__(self, env, callbacks=None, precision=32, save_sparse_reward=False):
    self._env = env
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episode = None
    self._save_sparse_reward = save_sparse_reward

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs_or, reward, done, info = self._env.step(action)
    obs = {k: self._convert(v) for k, v in obs_or.items()}
    transition = obs.copy()
    transition['action'] = action
    transition['reward'] = reward
    if self._save_sparse_reward:
      transition['sparse_reward'] = info.get('success')
    transition['discount'] = info.get('discount', np.array(1 - float(done)))
    self._episode.append(transition)
    if done:
      episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
      episode = {k: self._convert(v) for k, v in episode.items()}
      info['episode'] = episode
      for callback in self._callbacks:
        callback(episode)
    return obs_or, reward, done, info

  def reset(self):
    obs = self._env.reset()
    transition = obs.copy()
    transition['action'] = np.zeros(self._env.action_space.shape)
    transition['reward'] = 0.0
    transition['discount'] = 1.0
    if self._save_sparse_reward:
      transition['sparse_reward'] = 0.0
    self._episode = [transition]
    return obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
      dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
    elif np.issubdtype(value.dtype, np.uint8):
      dtype = np.uint8
    else:
      raise NotImplementedError(value.dtype)
    return value.astype(dtype)


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class ActionRepeat:

  def __init__(self, env, amount):
    self._env = env
    self._amount = amount

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    done = False
    total_reward = 0
    current_step = 0
    while current_step < self._amount and not done:
      obs, reward, done, info = self._env.step(action)
      total_reward += reward
      current_step += 1
    return obs, total_reward, done, info


class NormalizeActions:

  def __init__(self, env):
    self._env = env
    self._mask = np.logical_and(
        np.isfinite(env.action_space.low),
        np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)
    self._high = np.where(self._mask, env.action_space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)


class RewardObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    assert 'reward' not in spaces
    spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs

class Async:

  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _EXCEPTION = 4
  _CLOSE = 5

  def __init__(self, ctor, strategy='process'):
    self._strategy = strategy
    if strategy == 'none':
      self._env = ctor()
    elif strategy == 'thread':
      import multiprocessing.dummy as mp
    elif strategy == 'process':
      import multiprocessing as mp
    else:
      raise NotImplementedError(strategy)
    if strategy != 'none':
      self._conn, conn = mp.Pipe()
      self._process = mp.Process(target=self._worker, args=(ctor, conn))
      atexit.register(self.close)
      self._process.start()
    self._obs_space = None
    self._action_space = None

  @property
  def observation_space(self):
    if not self._obs_space:
      self._obs_space = self.__getattr__('observation_space')
    return self._obs_space

  @property
  def action_space(self):
    if not self._action_space:
      self._action_space = self.__getattr__('action_space')
    return self._action_space

  def __getattr__(self, name):
    if self._strategy == 'none':
      return getattr(self._env, name)
    self._conn.send((self._ACCESS, name))
    return self._receive()

  def call(self, name, *args, **kwargs):
    blocking = kwargs.pop('blocking', True)
    if self._strategy == 'none':
      return functools.partial(getattr(self._env, name), *args, **kwargs)
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    promise = self._receive
    return promise() if blocking else promise

  def close(self):
    if self._strategy == 'none':
      try:
        self._env.close()
      except AttributeError:
        pass
      return
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      # The connection was already closed.
      pass
    self._process.join()

  def step(self, action, blocking=True):
    return self.call('step', action, blocking=blocking)

  def reset(self, blocking=True):
    return self.call('reset', blocking=blocking)

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except ConnectionResetError:
      raise RuntimeError('Environment worker crashed.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError(f'Received message of unexpected type {message}')

  def _worker(self, ctor, conn):
    try:
      env = ctor()
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          assert payload is None
          break
        raise KeyError(f'Received message of unknown type {message}')
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print(f'Error in environment process: {stacktrace}')
      conn.send((self._EXCEPTION, stacktrace))
    conn.close()


def parse(n, options):
  # Get key values
  keys = dict((k, k in n) for k in options)
  # Remove keys from name
  n = '_'.join([s for s in n.split('_') if s not in options])
  return keys, n
