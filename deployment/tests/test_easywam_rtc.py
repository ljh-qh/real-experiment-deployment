import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/"src"))
from easywam_rtc import EasyWAMRTC
from easywam_adapter import ACTIVE
class EasyRTCTests(unittest.TestCase):
 def policy(self):
  return SimpleNamespace(device="cpu",mn=np.full(62,-2.,np.float32),mx=np.full(62,2.,np.float32),
   model=SimpleNamespace(infer_action_scheduler=SimpleNamespace(num_train_timesteps=1000),
    _predict_joint_noise=lambda **kw:(torch.ones(1),torch.zeros_like(kw["latents_action"]))))
 def test_normalization_and_no_prefix_identity(self):
  p=self.policy();g=EasyWAMRTC(p)
  kw=dict(latents_action=torch.zeros(1,32,62),timestep_action=torch.tensor([500.]))
  v,a=p.model._predict_joint_noise(**kw);self.assertEqual(float(a.sum()),0)
  prefix=np.ones((24,80),np.float32)
  g.set_request(prefix,14)
  np.testing.assert_array_equal(g.request[0][0,:24].numpy(),np.full((24,62),.5))
  v,a=p.model._predict_joint_noise(**kw)
  self.assertTrue((a[0,:14]<0).all());self.assertTrue((a[0,24:]==0).all())
  self.assertEqual(g.steps,1);g.clear()
  self.assertEqual(float(p.model._predict_joint_noise(**kw)[1].sum()),0)
 def test_rejects_invalid_prefix(self):
  g=EasyWAMRTC(self.policy())
  for prefix,c in ((np.zeros((33,80)),14),(np.zeros((24,62)),14),(np.full((24,80),np.nan),14),(np.zeros((24,80)),25)):
   with self.assertRaises(ValueError):g.set_request(prefix,c)

class RTCDeploymentGateTests(unittest.TestCase):
 def test_live_profile_stays_blocked_by_measured_latency(self):
  import launcher
  cfg=launcher.read_profile(P/"configs/profiles/easywam_bottle_async_rtc_lanczos.yaml",P/"runs/unused")
  self.assertEqual(cfg["task"],"bottle")
  self.assertTrue(cfg["worker"]["args"]["rtc"])
  self.assertEqual(cfg["client"]["feedback_confirm_ms"],0)
  with self.assertRaisesRegex(RuntimeError,"RTC gate"):
   launcher.check_profile(cfg)
 def test_queue_rejects_result_not_ready_at_commit_deadline(self):
  from concurrent.futures import Future
  from rtc_queue import RTCQueue
  pending=Future()
  q=RTCQueue(lambda _:pending,lambda x:x,commit_steps=14,prefix_capacity=24,hz=30)
  from collections import deque
  q.queue=deque({"physical":np.zeros(80)} for _ in range(24))
  q.next(0,0.,{})
  with self.assertRaisesRegex(RuntimeError,"deadline missed"):
   q.next(14,14/30,{})
