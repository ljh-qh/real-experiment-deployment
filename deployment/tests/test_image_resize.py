"""CPU-only image path parity tests through the real chunk-model adapter."""
from pathlib import Path
import sys, unittest
from types import SimpleNamespace
import numpy as np
from PIL import Image
import yaml
P=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(P/"vendor/bridge"),str(P/"vendor/openwam"),str(P/"assets/robot")]
from image_resize import prepare_resize_input
from openwam.deploy.obs_preprocess import ObsPreprocessor
from zmq_chunk_worker import DirectChunkModel
from zmq_offline_bridge import to_openwam,REPRESENTATION

class ResizeTests(unittest.TestCase):
 def setUp(self):
  self.pre=ObsPreprocessor(multiview=True,camera_layout=["head","left","right"],img_height=384,img_width=320)
 def message(self,task):
  root={"badminton":"fixed_observation_diagnosis","top":"top_observations","bottle":"bottle_observations"}[task]
  p=P/"assets/references"/root/"recorded_start"
  return {"type":"observation","task":task,"representation":REPRESENTATION,"observation":{"state":np.load(p/"state80.npy"),"images":{r:np.asarray(Image.open(p/(r+".png")).convert("RGB")) for r in ("main","wrist2","wrist")}}}
 def test_bilinear_default_is_byte_identical_and_preserves_input(self):
  for task in ("badminton","top","bottle"):
   msg=self.message(task);obs=to_openwam(msg,encode_images=False)
   self.assertIs(prepare_resize_input(obs,self.pre,"bilinear"),obs)
   expected=np.asarray(self.pre.preprocess(to_openwam(msg,encode_images=False))["image"])
   actual=np.asarray(self.pre.preprocess(prepare_resize_input(obs,self.pre,"bilinear"))["image"])
   np.testing.assert_array_equal(expected,actual)
 def test_real_worker_lanczos_matches_training_slots_and_keeps_actions_state_prompt(self):
  for task in ("badminton","top","bottle"):
   msg=self.message(task);obs=to_openwam(msg,encode_images=False);seen={}
   actions=np.repeat(msg["observation"]["state"][None],32,axis=0)
   def generate(conditions):
    seen.update(conditions);return {"actions":actions.copy()}
   model=DirectChunkModel(SimpleNamespace(generate=generate),self.pre);model.image_resize="lanczos"
   result,_=model(msg)
   canvas=Image.new("RGB",(320,384))
   for role,size,xy in (("head_camera",(320,256),(0,0)),("left_wrist_camera",(160,128),(0,256)),("right_wrist_camera",(160,128),(160,256))):
    canvas.paste(obs["images"][role].resize(size,Image.Resampling.LANCZOS),xy)
   np.testing.assert_array_equal(np.asarray(seen["first_frame_image"][0]),np.asarray(canvas))
   np.testing.assert_array_equal(seen["proprio"],msg["observation"]["state"])
   np.testing.assert_array_equal(result,actions)
   self.assertEqual(seen["prompt"],obs["prompt"])
 def test_invalid_mode_or_geometry_rejected(self):
  obs=to_openwam(self.message("bottle"),encode_images=False)
  with self.assertRaises(ValueError):prepare_resize_input(obs,self.pre,"other")
  self.pre.img_width=640
  with self.assertRaises(ValueError):prepare_resize_input(obs,self.pre,"lanczos")
 def test_profiles_differ_only_in_resize_flag(self):
  for src,dst in (("openwam_sync","openwam_badminton_sync_lanczos"),("openwam_top_sync","openwam_top_sync_lanczos"),("openwam_bottle_sync_no_dit_cache","openwam_bottle_sync_no_dit_cache_lanczos")):
   a=yaml.safe_load((P/"configs/profiles"/(src+".yaml")).read_text());b=yaml.safe_load((P/"configs/profiles"/(dst+".yaml")).read_text())
   self.assertEqual(b["worker"]["args"].pop("image-resize"),"lanczos");self.assertEqual(a,b)
if __name__=="__main__":unittest.main()
