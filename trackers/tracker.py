from ultralytics import YOLO
import supervision as sv
import pickle
import os
from utils.bbox_utils import get_bbox_width,get_center_of_bbox
import cv2
import numpy as np 
from scipy.spatial.distance import euclidean

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker=sv.ByteTrack()
        self.persistent_id_counter = 1
        self.persistent_tracks = {}

    def detect_frames(self,frames):
        detections = []
        batch_size=20
        for i in range(0,len(frames),batch_size):
            detections_batch=self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections+=detections_batch
        return detections
    
    def _match_persistent_id(self, new_bbox, frame_num, threshold=50):
        center_x_new = (new_bbox[0] + new_bbox[2]) / 2
        center_y_new = (new_bbox[1] + new_bbox[3]) / 2

        for pid, info in self.persistent_tracks.items():
            old_bbox = info['bbox']
            center_x_old = (old_bbox[0] + old_bbox[2]) / 2
            center_y_old = (old_bbox[1] + old_bbox[3]) / 2

            dist = euclidean((center_x_new, center_y_new), (center_x_old, center_y_old))
            if dist < threshold and frame_num - info['last_seen'] <= 10:
                # Match found
                return pid

        return None
    
    def get_objects_tracks(self,frames,read_from_stub=False,stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f : 
                tracks=pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)

        tracks={
            'players':[],
            'referees':[],
            'ball':[],
            'goalkeeper':[]
        }

        for frame_num,detection in enumerate(detections):
            cls_names=detection.names
            cls_names_inv={v:k for k,v in cls_names.items()}

            detection_supervision=sv.Detections.from_ultralytics(detection)
            detection_with_tracks=self.tracker.update_with_detections(detection_supervision)

            # print(detection_with_tracks)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})
            tracks['goalkeeper'].append({})

            for det in detection_with_tracks:
                bbox = det[0].tolist()
                cls_id = det[3]

                if cls_id == cls_names_inv['player']:
                    persistent_id = self._match_persistent_id(bbox, frame_num)
                    if persistent_id is None:
                        persistent_id = self.persistent_id_counter
                        self.persistent_id_counter += 1

                    self.persistent_tracks[persistent_id] = {'bbox': bbox, 'last_seen': frame_num}
                    tracks['players'][frame_num][persistent_id] = {'bbox': bbox}

                elif cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][0] = {'bbox': bbox}
                elif cls_id == cls_names_inv['goalkeeper']:
                    tracks['goalkeeper'][frame_num][0] = {'bbox': bbox}
            
            for frame_detection in detection_supervision:
                bbox=frame_detection[0].tolist()
                cls_id=frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1]={'bbox':bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f : 
                pickle.dump(tracks,f)

        return tracks

    def draw_ellipse(self,frame,bbox,color,track_id=None):  
        y2=int(bbox[3])
        x_center,_=get_center_of_bbox(bbox)
        width=get_bbox_width(bbox)

        cv2.ellipse(frame,
                    center=(x_center,y2),
                    axes=(int(width),int(0.35*width)),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4)
        
        rectangle_width=40
        rectangle_height=20
        x1_rect=x_center - rectangle_width//2
        x2_rect=x_center + rectangle_width//2
        y1_rect=(y2 - rectangle_height//2) + 15
        y2_rect=(y2 + rectangle_height//2) + 15

        if track_id is not None : 
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED
                          )
            x1_text=x1_rect+12
            if track_id > 99 : 
                x1_text -= 10
            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text),int(y1_rect+15)),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (0,0,0),
                        2
                    )

        
        return frame

    def draw_traingle(self,frame,bbox,color):
        y=int(bbox[1])
        x,_= get_center_of_bbox(bbox)

        traingle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        cv2.drawContours(frame,[traingle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[traingle_points],0,(0,0,0),2)

        return frame 

    def draw_annotations(self,video_frames,tracks):
        output_video_frames=[]
        for frame_num,frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict=tracks['players'][frame_num]
            referee_dict=tracks['referees'][frame_num]
            ball_dict=tracks['ball'][frame_num]
            goalkeeper_dict=tracks['goalkeeper'][frame_num]


            for track_id , player in player_dict.items():
                frame = self.draw_ellipse(frame,player['bbox'],(0,0,255),track_id)

            for track_id , referee in referee_dict.items():
                frame = self.draw_ellipse(frame,referee['bbox'],(0,255,255))

            for track_id , goalkeeper in goalkeeper_dict.items():
                frame = self.draw_ellipse(frame,goalkeeper['bbox'],(255,255,0),track_id)

            for track_id,ball in ball_dict.items():
                frame = self.draw_traingle(frame,ball['bbox'],(0,255,0))




            output_video_frames.append(frame)

        return output_video_frames