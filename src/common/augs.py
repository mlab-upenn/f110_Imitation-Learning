def flipNonZero():
    import cv2
    def call(obs_dict):
        new_dict = obs_dict.copy()

        #flip steering angle
        steer = obs_dict["steer"].copy()
        if steer["angle"] != 0.0:
            angle_flip = steer["angle"] * -1
            steer["angle"] = angle_flip
            new_dict["steer"] = steer

            #flip image 
            img = new_dict["img"].copy()
            new_dict["img"] = cv2.flip(img, 1)
            return new_dict
        return {"flag":False}
    return call

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
