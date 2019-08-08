from __future__ import print_function

def transform_obsarray(obs_array, func):
    ret_array = obs_array.copy()
    for obs_dict in obs_array:
        new_dict = func(obs_dict)
        if new_dict.get("flag", True):
            ret_array.append(new_dict)
    return ret_array