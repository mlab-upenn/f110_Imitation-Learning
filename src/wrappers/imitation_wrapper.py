from f110_gym.f110_core import f110Env, f110ActionWrapper, f110ObservationWrapper, f110Wrapper

from collections import deque
import numpy as np
import cv2

__author__ = "Dhruv Karthik <dhruvkar@seas.upenn.edu>"

class SkipEnv(f110Wrapper):
    def __init__(self, env, skip=4):
        """Return only 'skip-th frame"""
        f110Wrapper.__init__(self, env)
        self._skip = skip
        
    def step(self, action):
        """Repeat action & sum reward"""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        #print(obs["steer"])
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def serialize_obs(self):
        return self.env.serialize_obs()

class PreprocessImg(f110ObservationWrapper):
    def __init__(self, env):
        f110ObservationWrapper.__init__(self, env)
        self.observation_space = self.env.observation_space

    def observation(self, obs):
        """ For now, Crop any 'img' observations, in future, add input funclist array to preprocess"""
        new_obs = obs
        src_img = obs["img"]
        scale = 0.7
        src_img = cv2.resize(src_img, None, fx=scale, fy=scale)
        new_obs["img"] = src_img[80:, 12:-12, ...]
        #new_obs["img"] = src_img
        return new_obs

    def serialize_obs(self):
        return self.env.serialize_obs()

class GrayScale(f110ObservationWrapper):
    import cv2
    def __init__(self, env):
        f110ObservationWrapper.__init__(self, env)
        self.observation_space = self.env.observation_space

    def observation(self, obs):
        """ For now, Crop any 'img' observations, in future, add input funclist array to preprocess"""
        new_obs = obs
        src_img = obs["img"]
        new_obs["img"] = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        return new_obs

    def serialize_obs(self):
        return self.env.serialize_obs()

class FrameStack(f110Wrapper):
    def __init__(self, env, k):
        """Stack k last frames. Returns memory efficient lazy array"""
        f110Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.observation_space = self.env.observation_space
    
    def reset(self):
        obs_dict = self.env.reset()
        src_img = obs_dict["img"]
        frame = src_img.copy()
        for i in range(self.k-1):
            src_img = np.dstack((src_img, frame))
            self.frames.append(frame)
        obs_dict["img"] = src_img
        return obs_dict
    
    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        src_img = obs_dict["img"]

        self.frames.append(src_img)
        framelist = list(self.frames)
        final_img = framelist[0]
        framelist = framelist[1:]
        for i in range(self.k-1):
            final_img = np.dstack((final_img, framelist[i]))
        obs_dict["img"] = final_img
        return obs_dict, reward, done, info

    def serialize_obs(self):
        return self.env.serialize_obs()

class LazyFrames(object):
    def __init__(self, frames):
        """ Ensures that common frames between observations are only stored once"""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

def make_imitation_env(skip=10):
    env = f110Env()
    env = PreprocessImg(env)
    env = SkipEnv(env, skip=3)
    env = GrayScale(env)
    env = FrameStack(env, 3)
    return env