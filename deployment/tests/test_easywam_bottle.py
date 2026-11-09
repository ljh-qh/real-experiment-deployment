"""EasyWAM bottle resource and preprocessing checks without loading a model."""
import ast,sys,unittest
from pathlib import Path
import numpy as np
from PIL import Image
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/"src"))
class EasyBottleTests(unittest.TestCase):
 def test_image_slots_match_training(self):
  tree=ast.parse((P/"src/easywam_adapter.py").read_text())
  nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("_as_rgb","compose_image")]
  ns={"np":np,"Image":Image}
  exec(compile(ast.Module(body=nodes,type_ignores=[]),"adapter","exec"),ns)
  folder=P/"assets/references/bottle_observations/recorded_start"
  images={k:np.asarray(Image.open(folder/(k+".png")).convert("RGB")) for k in ("main","wrist2","wrist")}
  expected=Image.new("RGB",(320,384))
  for key,size,xy in (("main",(320,256),(0,0)),("wrist2",(160,128),(0,256)),("wrist",(160,128),(160,256))):
   expected.paste(Image.fromarray(images[key]).resize(size,Image.Resampling.LANCZOS),xy)
  np.testing.assert_array_equal(np.asarray(ns["compose_image"](images,image_resize="lanczos")),np.asarray(expected))
 def test_stats_project_exact_training_active_dimensions(self):
  root=P/"assets/checkpoints"
  source=np.load(root/"openwam_bottle/normalization_stats.npy",allow_pickle=True).item()["eef"]
  actual=np.load(root/"easywam_bottle/normalization_stats_62.npy",allow_pickle=True).item()
  active=np.r_[0:9,10:32,34:43,44:66]
  for key in ("min","max","mean","std","q01","q99"):
   np.testing.assert_array_equal(actual[key],np.asarray(source[key])[active])
