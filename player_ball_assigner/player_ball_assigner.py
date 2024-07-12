import sys
sys.path.append('../')
from utils import get_centre_of_bbox, measure_distance

class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, player_track, ball_bbox):
        ball_position = get_centre_of_bbox(ball_bbox)

        minimum_distance = 9999999
        assigned_player = -1

        for player_id, player in player_track.items():
            player_bbox = player["bbox"]

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if(distance<self.max_player_ball_distance):
                if(distance<minimum_distance):
                    minimum_distance = distance
                    assigned_player = player_id
        return assigned_player
