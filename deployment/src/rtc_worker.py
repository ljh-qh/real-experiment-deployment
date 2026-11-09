"""Experimental RTC worker. No robot SDK imports or hardware commands."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

from deployment_paths import ROOT
sys.path[:0]=[str(ROOT / x) for x in ("vendor/bridge", "vendor/openwam", "assets/robot", "deps/bridge")]

from zmq_chunk_worker import DirectChunkModel,serve_worker,loopback_endpoint
from zmq_offline_bridge import load_codec,REPRESENTATION
from rtc_guidance import RTCGuidance


class RTCChunkModel:
    def __init__(self, config, checkpoint, checkpoint_name, device, mode, guidance, context_cache):
        self.base=DirectChunkModel.load(config,checkpoint,checkpoint_name,device)
        if self.base.engine._dit_cache is not None:
            raise ValueError("RTC must not reuse stale denoising velocities; disable dit_cache")
        architecture=self.base.engine.architecture
        if mode=="full_vjp" and architecture._compiled_mot_run_joint_loop is not None:
            raise ValueError("Full RTC VJP requires the uncompiled inference profile")
        if context_cache:
            if mode!="identity_vjp":
                raise ValueError("Compiled context cache is only validated for identity_vjp")
            from deploy_context_cache import install_context_cache
            install_context_cache(self.base,verify_context=True)
        for parameter in architecture.parameters():
            parameter.requires_grad_(False)
        self.guidance=RTCGuidance(architecture,mode,guidance)
        self.deployment_metadata={"engine":"openwam_rtc","rtc_mode":mode,
             "rtc_is_approximation":mode=="identity_vjp","warmup_completed":False}
    def __call__(self,message):
        rtc=message.get("rtc")
        if rtc is not None:
            self.guidance.set_request(rtc["prefix"],int(rtc["commit_steps"]))
        else:
            self.guidance.clear()
        try:
            actions,timing=self.base(message)
            timing["rtc_guided_steps"]=self.guidance.steps if rtc is not None else 0
            if rtc is not None and self.guidance.steps==0:
                raise RuntimeError("RTC prefix supplied but no guided denoising step executed")
            return actions,timing
        finally:
            self.guidance.clear()
    def warmup(self,fixtures):
        for name in ("recorded_start","recorded_mid","recorded_start"):
            f=Path(fixtures)/name
            msg={"type":"observation","task":"badminton","representation":REPRESENTATION,
                 "observation":{"state":np.load(f/"state80.npy"),"images":{
                 k:np.asarray(Image.open(f/(k+".png")).convert("RGB")) for k in ("main","wrist","wrist2")}}} 
            self(msg)
        self.deployment_metadata["warmup_completed"]=True


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--checkpoint-name",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--rtc-mode",choices=("full_vjp","identity_vjp"),required=True)
    p.add_argument("--rtc-max-guidance",type=float,default=10.)
    p.add_argument("--context-cache",action="store_true")
    p.add_argument("--warmup-fixtures",type=Path,required=True)
    for key in ("observation-endpoint","action-endpoint","control-endpoint"):
        p.add_argument("--"+key,type=loopback_endpoint,required=True)
    p.add_argument("--ready-file",type=Path,required=True)
    p.add_argument("--codec",type=Path,required=True)
    p.add_argument("--max-age-ms",type=int,default=1000)
    a=p.parse_args()
    model=RTCChunkModel(a.config,a.checkpoint,a.checkpoint_name,a.device,a.rtc_mode,a.rtc_max_guidance,a.context_cache)
    model.warmup(a.warmup_fixtures)
    serve_worker(model,load_codec(a.codec),a.observation_endpoint,a.action_endpoint,a.control_endpoint,a.ready_file,a.max_age_ms)


if __name__=="__main__":
    main()
