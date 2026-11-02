"""CPU synthetic invariants, explicitly NOT real-data segmentation validation."""
import unittest
import numpy as np
import cv2
import tempfile, hashlib
from pathlib import Path
from paired_hue import array_hash
from paired_hue import hue_rgb, hue_parameter, PairedHueReader, VerifiedMaskProvider

class Tests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.random.default_rng(7).integers(0,256,(2,2,32,32,3),dtype=np.uint8)
        self.mask=np.zeros((2,2,32,32),dtype=bool); self.mask[...,5:20,4:15]=True
    def test_outside_and_replay(self):
        before=np.random.get_state(); d,k=hue_parameter(5,'train',{'id':3},10)
        out=hue_rgb(self.rgb,self.mask,d)
        self.assertTrue(np.array_equal(out[~self.mask],self.rgb[~self.mask]))
        self.assertTrue(np.array_equal(out,hue_rgb(self.rgb,self.mask,d)))
        self.assertGreater(np.abs(out.astype(float)-self.rgb).sum(),0)
        self.assertTrue(np.array_equal(before[1],np.random.get_state()[1]))
        self.assertEqual((d,k),hue_parameter(5,'train',{'id':3},10))
        self.assertNotEqual(d,hue_parameter(5,'train',{'id':3},11)[0])
    def test_gray_empty_disable(self):
        gray=np.full_like(self.rgb,123)
        self.assertTrue(np.array_equal(gray,hue_rgb(gray,self.mask,.2)))
        self.assertTrue(np.array_equal(self.rgb,hue_rgb(self.rgb,np.zeros_like(self.mask),.2)))
        self.assertTrue(np.array_equal(self.rgb,hue_rgb(self.rgb,self.mask,0)))
    def test_sv_quantization(self):
        out=hue_rgb(self.rgb,self.mask,.2)
        for x,y,m in zip(self.rgb.reshape(-1,32,32,3),out.reshape(-1,32,32,3),self.mask.reshape(-1,32,32)):
            a=cv2.cvtColor(x.astype('float32')/255,cv2.COLOR_RGB2HSV)
            b=cv2.cvtColor(y.astype('float32')/255,cv2.COLOR_RGB2HSV)
            self.assertLessEqual(float(np.max(abs(a[...,2]-b[...,2]))),1/255+1e-6)
            self.assertLessEqual(float(np.max(abs(a[...,1]-b[...,1]))),.02)
    def test_missing_mask_rejected(self):
        with self.assertRaises(RuntimeError): VerifiedMaskProvider({}).get('s','0',0,'cam',self.rgb[0,0])
    def test_estimated_mask_gate_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'mask.npz';np.savez_compressed(p,masks=self.mask[0].astype('uint8'))
            entry=dict(source_kind='reviewed_model_segmentation',semantic_status='prototype_fixed_frame_pass',
              model_checkpoint_sha256='test-only',prompt_sha256='test-only',human_ground_truth=False,
              review_evidence_sha256='test-only',alignment_status='reviewed_pass',alignment_evidence_sha256='test-only',
              allowed_classes=['table','wall'],original_rgb_sha256=array_hash(self.rgb[0,0]),path=str(p),
              sha256=hashlib.sha256(p.read_bytes()).hexdigest(),frame_index=0)
            registry={'s/0/0/cam':entry}
            with self.assertRaises(RuntimeError): VerifiedMaskProvider(registry).get('s','0',0,'cam',self.rgb[0,0])
            provider=VerifiedMaskProvider(registry,allow_reviewed_model=True)
            self.assertTrue(np.array_equal(provider.get('s','0',0,'cam',self.rgb[0,0]),self.mask[0,0]))
            with self.assertRaises(RuntimeError): provider.get('s','0',0,'cam',self.rgb[0,1])
            entry['semantic_status']='FAILED_TRACKING'
            with self.assertRaises(RuntimeError): provider.get('s','0',0,'cam',self.rgb[0,0])
    def test_metadata_preserved(self):
        sample={'rgbs':np.moveaxis(self.rgb,-1,2),'frame_id':np.array([0,1]),'qpos':np.arange(10),'task_description':'test'}
        class Dataset:
            def getitem_from_handle(_,handle,apply_augmentation):
                assert not apply_augmentation
                return sample
        class SyntheticMask:
            def get(_,suite,traj,frame,cam,rgb): return self.mask[0,0]
        h=dict(camera_keys=['agentview_camera','wrist_camera'],group_name='s',traj_id='000010',frame_id=0,sampled_horizon=2)
        pair=PairedHueReader(Dataset(),SyntheticMask()).read(h,0,'train')
        self.assertIs(pair['x_prime']['qpos'],sample['qpos'])
        self.assertIs(pair['x_prime']['task_description'],sample['task_description'])
        off=PairedHueReader(Dataset(),VerifiedMaskProvider({}),enabled=False).read(h,0,'train')
        self.assertTrue(np.array_equal(off['x']['rgbs'],off['x_prime']['rgbs']))

if __name__=='__main__': unittest.main()
