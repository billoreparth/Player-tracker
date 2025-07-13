from utils.video_utils import save_video,read_video
from trackers.tracker import Tracker

def main():
    #read video
    video_frames=read_video('15sec_input_720p.mp4')

    tracker=Tracker('best.pt')
    tracks=tracker.get_objects_tracks(video_frames,read_from_stub=True,stub_path='stubs/track_stubs.pkl')

    output_video_frames=tracker.draw_annotations(video_frames,tracks)
    #save video
    save_video(output_video_frames,'output_videos/test_vid.avi')

if __name__=='__main__':
    main()