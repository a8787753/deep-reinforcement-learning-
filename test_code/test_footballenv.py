import gfootball.env as football_env
from gym import spaces
import numpy as np

# nv_args = {, "n_agent": 3, "reward": "scoring"}

env = football_env.create_environment(env_name="academy_3_vs_1_with_keeper",
                                      number_of_left_players_agent_controls=3,
                                      rewards="scoring",
                                      representation="raw")


def _encode_role_onehot(player_role):
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result[player_role] = 1.0
    return np.array(result)


state = env.reset()
obs = state[0]
player_num = obs["active"]

player_pos_x, player_pos_y = obs["left_team"][player_num]
player_direction = np.array(obs["left_team_direction"][player_num])
player_speed = np.linalg.norm(player_direction)
player_role = obs["left_team_roles"][player_num]
player_role_onehot = _encode_role_onehot(player_role)
player_tired = obs["left_team_tired_factor"][player_num]
is_dribbling = obs["sticky_actions"][9]
is_sprinting = obs["sticky_actions"][8]

ball_x, ball_y, ball_z = obs["ball"]
ball_x_relative = ball_x - player_pos_x
ball_y_relative = ball_y - player_pos_y

ball_x_speed, ball_y_speed, _ = obs["ball_direction"]
ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])




