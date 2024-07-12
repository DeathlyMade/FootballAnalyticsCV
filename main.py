from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
import numpy as np
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator

def main():
    frames = read_video('Input_videos/08fd33_4.mp4')
    
    tracker = Tracker('models/best_yolov5.pt')
    tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path='stubs/track_stubs_yolov5.pkl')
    tracker.add_position_to_track(tracks)

    camera_movement_estimator = CameraMovementEstimator(frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frames, read_from_stub=True, stub_path='stubs/camera_movement_stubs_yolov5.pkl')
    camera_movement_estimator.add_adjust_position_to_tracks(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    tracks = speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks['players'][0])
    for frame_num,player_track in enumerate(tracks['players']):
        for player_id,track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num,player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if(assigned_player!=-1):
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    output_video_frames = tracker.draw_annotations(frames, tracks, team_ball_control)

    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    save_video(output_video_frames, 'Output_videos/output_video.avi')

if __name__ == '__main__':
    main()