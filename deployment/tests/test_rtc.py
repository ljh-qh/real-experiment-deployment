"""CPU tests for RTC mathematics, committed queues, and bounded feedback confirmation."""
import json,sys,unittest
from pathlib import Path
from concurrent.futures import Future
import numpy as np
import torch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"src"
sys.path.insert(0,str(D))
from rtc_guidance import prefix_weights,guidance_gain,guided_velocity,RTCGuidance,ACTIVE
from rtc_queue import RTCQueue
from feedback_guard import confirm_tracking

class GuidanceTests(unittest.TestCase):
 def test_mask(self):
  torch.testing.assert_close(prefix_weights(2,6,10),torch.tensor([1,1,.8,.6,.4,.2,0,0,0,0]))
 def test_full_jacobian_and_sign(self):
  x=torch.ones(1,4,2);target=torch.zeros_like(x);w=torch.ones_like(x)
  def fn(a):return a*0,2*a+.1
  sigma=.25;v=2*x+.1;clean=x-sigma*v;err=target-clean
  _,actual=guided_velocity(fn,x,sigma,target,w,"full_vjp",10)
  expected=v-guidance_gain(sigma,10)*(1-2*sigma)*err
  torch.testing.assert_close(actual,expected)
 def test_identity_is_explicit_approximation(self):
  x=torch.ones(1,4,2);target=torch.zeros_like(x);w=torch.ones_like(x)
  fn=lambda a:(a*0,2*a+.1)
  _,actual=guided_velocity(fn,x,.25,target,w,"identity_vjp",10)
  expected=2*x+.1-guidance_gain(.25,10)*(target-(x-.25*(2*x+.1)))
  torch.testing.assert_close(actual,expected)
 def test_disabled_matches_exactly(self):
  x=torch.randn(1,4,2)
  fn=lambda a:(a*2,a*3)
  a,b=guided_velocity(fn,x,.5,None,torch.zeros_like(x),"full_vjp",10)
  self.assertTrue(torch.equal(a,x*2));self.assertTrue(torch.equal(b,x*3))
 def test_affine_output_inverse_not_input_normalizer(self):
  class Norm:
   def unnormalize(self,x):return x*2+3
   def normalize(self,x):raise AssertionError("Do not use proprio transform")
  arch=SimpleNamespace(normalizer=Norm(),device="cpu",forward=lambda *a,**k:None)
  g=RTCGuidance(arch);prefix=np.full((20,80),5,np.float32);g.set_request(prefix,14)
  target,w=g.request
  np.testing.assert_allclose(target[0,:20,ACTIVE].numpy(),1)
  self.assertTrue(torch.all(w[0,:,~torch.from_numpy(ACTIVE)]==0))
  self.assertTrue(torch.all(target[0,20:]==0))
 def test_endpoint_gain_finite(self):
  self.assertEqual(guidance_gain(1,10),10)
  self.assertLessEqual(guidance_gain(1e-9,10),10)
 def test_invalid_prefix_rejected(self):
  arch=SimpleNamespace(normalizer=SimpleNamespace(unnormalize=lambda x:x),device="cpu",forward=lambda *a,**k:None)
  g=RTCGuidance(arch)
  for prefix,commit in ((np.zeros((4,80)),5),(np.zeros((33,80)),14),(np.full((20,80),np.nan),14)):
   with self.assertRaises(ValueError):g.set_request(prefix,commit)

class QueueTests(unittest.TestCase):
 def setUp(self):
  self.calls=[]
  self.pending=[]
  self.initial=np.arange(32,dtype=np.float32)[:,None]*np.ones((1,80),np.float32)
  def submit(msg):
   self.calls.append(msg);f=Future()
   if len(self.calls)==1:f.set_result({"action":self.initial,"roundtrip_seconds":.2})
   else:self.pending.append(f)
   return f
  self.q=RTCQueue(submit,lambda a:[{"physical":x.copy(),"command":x.copy()} for x in a])
  self.q.prime({})
 def test_committed_prefix_is_never_replaced(self):
  outputs=[]
  for step in range(26):
   outputs.append(self.q.next(step,step/30,{"sequence":step})["command"][0])
  self.assertEqual(outputs,list(range(26)))
  np.testing.assert_equal(self.calls[1]["rtc"]["prefix"],self.initial[12:32])
  self.pending[0].set_result({"action":self.initial+100,"roundtrip_seconds":.4})
  p=self.q.next(26,26/30,{"sequence":26})
  self.assertEqual(p["command"][0],114)
 def test_missed_deadline_stops(self):
  for i in range(26):self.q.next(i,i/30,{})
  with self.assertRaisesRegex(RuntimeError,"deadline missed"):self.q.next(26,26/30,{})
 def test_late_timeline_stops(self):
  for i in range(26):self.q.next(i,i/30,{})
  self.pending[0].set_result({"action":self.initial,"roundtrip_seconds":.4})
  with self.assertRaisesRegex(RuntimeError,"timeline slipped"):self.q.next(26,26/30+.1,{})
 def test_tail_is_preplanned_from_last_committed_packet(self):
  anchors=[]
  def prepare(actions,anchor):
   anchors.append(anchor["command"][0])
   return [{"physical":x.copy(),"command":x.copy()} for x in actions]
  self.q.plan_from_anchor=prepare
  for i in range(26):self.q.next(i,i/30,{})
  self.pending[0].set_result({"action":self.initial+100,"roundtrip_seconds":.4})
  self.assertEqual(anchors,[25])
  self.q.plan=lambda a: (_ for _ in ()).throw(AssertionError("Main thread must not plan adoption"))
  p=self.q.next(26,26/30,{})
  self.assertEqual(p["command"][0],114)
 def test_invalid_horizon(self):
  with self.assertRaises(ValueError):RTCQueue(None,None,commit_steps=17,prefix_capacity=20)

class GuardTests(unittest.TestCase):
 def setUp(self):
  self.t=0.
 def clock(self):return self.t
 def sleep(self,s):self.t+=s
 def test_transient_recovers_without_dispatch(self):
  vals=iter([(1,15.6),(1,15.2),(1,14)])
  r=confirm_tracking(lambda:next(vals),8,15,clock=self.clock,sleep=self.sleep)
  self.assertTrue(r["recovered"]);self.assertAlmostEqual(r["waited_s"],.04)
 def test_persistent_stops_at_100ms(self):
  r=confirm_tracking(lambda:(1,16),8,15,clock=self.clock,sleep=self.sleep)
  self.assertFalse(r["recovered"]);self.assertAlmostEqual(r["waited_s"],.1)
 def test_native_fault_immediate(self):
  def read():raise RuntimeError("native limit")
  with self.assertRaisesRegex(RuntimeError,"native limit"):confirm_tracking(read,8,15,clock=self.clock,sleep=self.sleep)
  self.assertEqual(self.t,0)
 def test_nan_immediate(self):
  with self.assertRaises(RuntimeError):confirm_tracking(lambda:(float("nan"),1),8,15,clock=self.clock,sleep=self.sleep)
 def test_zero_window_preserves_old_stop(self):
  r=confirm_tracking(lambda:(9,1),8,15,confirm_s=0,clock=self.clock,sleep=self.sleep)
  self.assertFalse(r["recovered"]);self.assertEqual(self.t,0)


class ProjectionTests(unittest.TestCase):
 def test_projected_prefix_matches_existing_execution(self):
  import ast,runpy
  from types import SimpleNamespace
  import h5py
  C=ROOT/"vendor/bridge"
  sys.path[:0]=[str(ROOT/"deps/infra"),str(ROOT/"deps/observer")]
  g=runpy.run_path(str(D/"rtc_state_client.py"),run_name="offline_projection")
  tree=ast.parse((D/"rtc_state_client.py").read_text())
  node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="plan_one")
  env=dict(g,args=SimpleNamespace(max_joint_step_deg=0,max_hand_step_deg=2))
  exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),"planner","exec"),env)
  f=ROOT/"assets/references/fixed_observation_diagnosis/recorded_start"
  meta=json.loads((f/"input.json").read_text())
  with h5py.File(Path(meta["episode"])/"robot_states.h5") as h:
   ar=h["robot_state/pico_teleop"];seed=np.r_[ar["real_left_deg"][meta["row"]],ar["real_right_deg"][meta["row"]]]
  actions=np.load(ROOT/"tests/fixtures/actions.npz")["steps10_cache1_recorded_start"]
  legacy=ast.parse((D/"infra_state_client.py").read_text())
  loop=next(n for n in ast.walk(legacy) if isinstance(n,ast.While) and "args.duration" in ast.unparse(n.test))
  body=[];include=False
  for n in loop.body:
   text=ast.unparse(n)
   if "decode_action(action)" in text:include=True
   if text.startswith("hands['left'].send"):break
   if include and not text.startswith("arm.send_joint_positions_deg"):body.append(n)
  code=compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),"legacy","exec")
  hand=None
  for action in actions:
   old=dict(g,args=env["args"],action=action,seed_deg=seed,prev_cmd=seed,prev_hand=hand)
   exec(code,old)
   packet=env["plan_one"](action,seed,hand)
   np.testing.assert_array_equal(packet["q"],old["q_deg_cmd"])
   np.testing.assert_array_equal(packet["hand"],old["hand_targets"])
   expected=g["canonical80"](np.r_[np.deg2rad(packet["q"]),packet["hand"]][None],g["robot"])[0]
   np.testing.assert_array_equal(packet["physical"],expected)
   if hand is not None:self.assertLessEqual(np.max(abs(np.rad2deg(packet["hand"]-hand))),2.000001)
   seed=packet["q"];hand=packet["hand"]

if __name__=="__main__":
 suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
 r=unittest.TextTestRunner(verbosity=2).run(suite)
 (ROOT/"offline_checks/test_rtc_report.json").write_text(json.dumps({"tests":r.testsRun,"failures":len(r.failures),"errors":len(r.errors)},indent=2))
 raise SystemExit(not r.wasSuccessful())
