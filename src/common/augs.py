def toDeg():
    import math
    def call(obs_dict):
        new_dict = obs_dict.copy()
        steer = obs_dict["steer"].copy()
        angle_deg = steer["angle"] * 180.0/math.pi
        steer["angle"] = angle_deg
        new_dict["steer"] = steer
        return new_dict
    return call
