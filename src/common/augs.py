
def mirror(obs_dict):
    """
    Expects cv_img input
    """
    return obs_dict

def toDeg():
    import math
    def call(obs_dict):
        steer = obs_dict["steer"]
        angle_deg = steer["angle"] * 180.0/math.pi
        steer["angle"] = angle_deg
        return obs_dict
    return call