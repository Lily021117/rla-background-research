"""Initial points selected by assistant visual inspection of frozen originals.
Coordinates are original 256x256 RGB, top-left origin, not contact-sheet coords.
These are semantic prompts, NOT segmentation ground truth or human review.
"""
import argparse,json
from pathlib import Path
from audit import write_json,sha

# Conservative prototype deliberately labels only the clearly visible tabletop.
# Unprompted wall and cabinet areas stay uncolored; neither is inferred by motion.
PROMPTS = {
'libero_spatial_000010_agentview_camera':([[20,220],[150,235],[240,245],[20,105]],[[20,40],[125,40],[120,80],[68,115],[110,180],[57,180],[160,120],[215,160],[111,200]]),
'libero_spatial_000010_wrist_camera':([[120,100],[170,130],[180,190],[100,70]],[[220,20],[150,10],[50,175],[236,139],[25,70],[35,208],[213,208],[128,240]]),
'libero_spatial_000011_agentview_camera':([[20,220],[150,235],[240,245],[20,105]],[[20,40],[125,40],[120,80],[34,126],[68,150],[70,126],[60,184],[118,193],[163,121],[215,150]]),
'libero_spatial_000011_wrist_camera':([[100,100],[140,170],[120,70]],[[215,20],[140,10],[225,70],[223,145],[60,177],[20,60],[35,208],[213,208],[128,240]]),
'libero_object_000010_agentview_camera':([[20,220],[230,200],[75,80],[225,80]],[[45,130],[112,116],[147,118],[192,117],[117,160],[158,155],[188,165],[125,40],[10,8]]),
'libero_object_000010_wrist_camera':([[120,80],[180,80],[130,180],[60,120]],[[30,25],[80,50],[240,75],[165,150],[92,172],[10,133],[144,4],[35,208],[213,208],[128,240]]),
'libero_object_000011_agentview_camera':([[20,220],[230,200],[75,80],[225,80]],[[45,130],[112,125],[147,119],[158,140],[192,123],[118,167],[192,151],[125,40],[10,8]]),
'libero_object_000011_wrist_camera':([[120,80],[180,80],[130,180],[60,120]],[[48,25],[15,15],[140,3],[240,80],[10,135],[90,173],[156,152],[35,208],[213,208],[128,240]]),
'libero_goal_000010_agentview_camera':([[20,220],[150,235],[240,245],[20,105]],[[20,40],[125,40],[120,80],[210,75],[70,115],[145,120],[170,115],[130,147],[92,162],[130,181],[220,165]]),
'libero_goal_000010_wrist_camera':([[160,100],[140,165],[230,85],[70,200]],[[20,40],[120,20],[120,75],[179,70],[90,138],[15,155],[40,160],[230,165],[35,208],[213,208],[128,240]]),
'libero_goal_000011_agentview_camera':([[20,220],[150,235],[240,245],[20,105]],[[20,40],[125,40],[120,80],[210,75],[70,115],[145,120],[170,115],[130,147],[92,162],[130,181],[220,165]]),
'libero_goal_000011_wrist_camera':([[160,100],[140,165],[230,85],[70,200]],[[20,40],[120,20],[120,75],[179,70],[90,138],[15,155],[40,160],[230,165],[35,208],[213,208],[128,240]]),
'libero_10_000010_agentview_camera':([[20,220],[150,235],[240,245],[20,105]],[[20,40],[125,40],[120,80],[220,40],[47,143],[111,161],[187,126],[190,160]]),
'libero_10_000010_wrist_camera':([[150,120],[200,150],[130,190],[225,40]],[[30,70],[30,143],[150,18],[245,75],[95,95],[35,208],[213,208],[128,240]]),
'libero_10_000011_agentview_camera':([[20,220],[220,235],[220,105],[30,105]],[[30,30],[125,40],[220,25],[127,112],[184,132],[88,160],[130,228]]),
'libero_10_000011_wrist_camera':([[70,100],[220,100],[130,145],[220,30]],[[125,20],[180,135],[30,180],[35,208],[213,208],[128,240]]),
}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
 protocol=json.loads((a.run/'selection_and_protocol.json').read_text());out={}
 for v in protocol['videos']:
  pos,neg=PROMPTS[v['id']]
  out[v['id']]=dict(source_video_sha256=v['sha256'],prompt_coordinate_space='original_256x256_top_left_RGB',
   objects=[dict(object_id=1,semantic='table',keyframes=[dict(frame=0,positive=pos,negative=neg,
    basis='Assistant inspection of fixed original contact sheet: exposed tabletop positives; visible robot, objects, cabinetry and ambiguous image border negatives')])],
   wall_policy='not annotated in this small prototype; uncertain wall/cabinet distinctions excluded',
   inspected_original_frames=v['fixed_review_frames'],human_review=False,assistant_visual_review=True)
 write_json(a.run/'prompts_v1.json',out)
 print('Saved16 video prompt sets, frame0 only; no segmentation run yet')
if __name__=='__main__':main()
