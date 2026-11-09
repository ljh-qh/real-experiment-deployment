"""Apply the accepted hardware profile to the project-local execution client."""
import argparse,json,os,runpy,signal,sys
from pathlib import Path
import yaml
parser=argparse.ArgumentParser(add_help=False)
parser.add_argument("--output",type=Path,required=True)
args,_=parser.parse_known_args()
OUT=args.output
sys.path.insert(0,str(Path(__file__).resolve().parent))
cfg=yaml.safe_load((OUT/"model_config.yaml").read_text())
source=str(Path(__file__).with_name("rtc_state_client.py"))
g=runpy.run_path(source,run_name="existing_infra_state_client")
g["ARM_CFG"].vel_ratio=cfg["hardware"]["vel_ratio"]
g["ARM_CFG"].acc_ratio=cfg["hardware"]["acc_ratio"]
effective={"pid":os.getpid(),"vel_ratio":g["ARM_CFG"].vel_ratio,"acc_ratio":g["ARM_CFG"].acc_ratio,
           "hz":cfg["client"]["hz"],"sync":cfg["client"]["sync"],"duration":cfg["client"]["duration"],"max_joint_step_deg":cfg["client"]["max_joint_step_deg"],"max_hand_step_deg":cfg["client"]["max_hand_step_deg"],"feedback_confirm_ms":cfg["client"]["feedback_confirm_ms"]}
(OUT/"effective_config.json").write_text(json.dumps(effective,indent=2))
print(json.dumps(effective),flush=True)
# Explicit interrupt handling keeps manual stop available even under a remote launcher.
signal.signal(signal.SIGINT,signal.default_int_handler)
raise SystemExit(g["main"]())
