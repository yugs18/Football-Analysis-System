from utils import read_video, save_video
from trackers import Tracker

def main():
    video_frames = read_video('Input_Videos/08fd33_4.mp4')

    tracker = Tracker('model/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub = True, stub_path = "stubs/track_stub.pkl")

    output_video_frame = tracker.draw_annotations(video_frames, tracks)

    save_video(video_frames, 'output_videos/video.avi')


if __name__ == "__main__":
    main()