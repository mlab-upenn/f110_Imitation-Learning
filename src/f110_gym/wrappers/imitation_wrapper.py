from f110_gym.f110_core import f110Env, f110ActionWrapper, f110ObservationWrapper, f110Wrapper

class SkipEnv(f110Wrapper):
    def __init__(self, env, skip=10):
        """Return only 'skip-th frame"""
        f110Wrapper.__init__(self, env)
        self._skip = skip
        
    def step(self, action):
        """Repeat action & sum reward"""
        print("IN SKIPENV STEP")
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class PreprocessImg(f110ObservationWrapper):
    import cv2

    def __init__(self, env):
        f110ObservationWrapper.__init__(self, env)
        self.observation_space = self.env.observation_space

    def observation(self, obs):
        """ For now, Resize & Crop any 'img' observations """
        new_obs = obs
        for sensor in obs:
            if 'img' in sensor:
                cv_img = obs[sensor]
                cv_img = cv2.resize(cv_img, None, fx=0.5, fy=0.5)
                cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv_img = cv_img[100:200, :, :]
                new_obs["img"] = cv_img
        return new_obs

def make_imitation_env(self, skip=10):
    env = f110Env()
    env = PreprocessImg(env)
    env = SkipEnv(env)
    return env