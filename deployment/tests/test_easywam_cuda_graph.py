import ast,unittest,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils._pytree import tree_flatten,tree_unflatten
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/"src"))
class GraphTests(unittest.TestCase):
 def graph(self):
  tree=ast.parse((P/"src/easywam_cuda_graph.py").read_text())
  node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="GraphDenoise")
  ns={"torch":torch,"tree_flatten":tree_flatten,"tree_unflatten":tree_unflatten}
  exec(compile(ast.Module(body=[node],type_ignores=[]),"graph","exec"),ns)
  g=ns["GraphDenoise"](None)
  g.static,g.spec=tree_flatten({"x":torch.zeros(1,2),"enabled":True})
  g.output=(g.static[0],)
  g.graph=SimpleNamespace(replay=lambda:None)
  return g
 def test_replay_refreshes_inputs_and_returns_independent_outputs(self):
  g=self.graph();a=g(x=torch.ones(1,2),enabled=True)[0]
  b=g(x=torch.full((1,2),2.),enabled=True)[0]
  self.assertTrue(torch.equal(a,torch.ones(1,2)))
  self.assertTrue(torch.equal(b,torch.full((1,2),2.)))
  self.assertEqual(g.replays,2)
 def test_signature_changes_fail(self):
  for kwargs in ({"x":torch.zeros(2,2),"enabled":True},{"x":torch.zeros(1,2),"enabled":False},{"other":torch.zeros(1,2)}):
   with self.assertRaises(ValueError):self.graph()(**kwargs)
 def test_profile_only_changes_execution_and_keeps_live_gate_closed(self):
  import launcher
  a=launcher.read_profile(P/"configs/profiles/easywam_bottle_async_rtc_lanczos.yaml",P/"runs/unused")
  b=launcher.read_profile(P/"configs/profiles/easywam_bottle_async_rtc_cuda_graph.yaml",P/"runs/unused")
  self.assertEqual(b["worker"]["env"].pop("EASYWAM_EXECUTION_PROFILE"),"cuda_graph_fa2")
  with self.assertRaisesRegex(RuntimeError,"RTC gate"):launcher.check_profile(b)
  b["rtc"]["validation_file"]=a["rtc"]["validation_file"]
  self.assertEqual(a,b)
