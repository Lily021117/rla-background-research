"""Exhaustive contact sheets from an already saved comparison video; no simulator."""
import argparse
from PIL import Image,ImageDraw
import cv2
from prepare import RUN,dump
def main():
 p=argparse.ArgumentParser();p.add_argument('--suite',required=True);a=p.parse_args();src=RUN/'pilot_replay'/a.suite
 out=src/'all_frame_review';out.mkdir(exist_ok=False);cap=cv2.VideoCapture(str(src/'full_clean_comparison.mp4'));frames=[]
 while True:
  ok,im=cap.read()
  if not ok:break
  frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
 cap.release();pages=[]
 for start in range(0,len(frames),28):
  sheet=Image.new('RGB',(768,7*210),'#222222');draw=ImageDraw.Draw(sheet)
  for j,im in enumerate(frames[start:start+28]):
   x=(j%4)*192;y=(j//4)*210;sheet.paste(Image.fromarray(im).resize((192,192)),(x,y+18));draw.text((x+2,y+1),f'frame {start+j}',fill='white')
  name=out/f'page_{start//28:02d}.jpg';sheet.save(name,quality=95);pages.append(str(name))
 dump(out/'index.json',dict(frames=len(frames),all_frames_included=True,layout='per cell: front original|candidate above wrist original|candidate',pages=pages,review_status='NOT_YET_REVIEWED'))
if __name__=='__main__':main()
