"""Preview or explicitly execute a measured demonstration origin; default sends no commands."""
import argparse
import json,time,signal
from pathlib import Path
from deployment_paths import ROOT
import numpy as np
from demo_origin import load_reference


def prepare_return_start(measured,limits):
    """Project only hand encoder excursions within the existing 1 degree tolerance."""
    measured=np.asarray(measured,dtype=float);limits=np.asarray(limits,dtype=float)
    if measured.shape!=(58,) or limits.shape!=(58,2) or not np.isfinite(measured).all() or not np.isfinite(limits).all():
        raise ValueError('Invalid initial feedback or limits')
    if np.any(limits[:,0]>limits[:,1]):raise ValueError('Invalid native limits')
    tolerance=np.r_[np.zeros(14),np.ones(44)]
    if np.any(measured<limits[:,0]-tolerance) or np.any(measured>limits[:,1]+tolerance):
        raise ValueError('Initial feedback outside native limits and encoder tolerance')
    command=measured.copy();command[14:]=np.clip(command[14:],limits[14:,0],limits[14:,1])
    adjustments=[dict(index=int(i),measured_deg=float(measured[i]),command_deg=float(command[i])) for i in np.flatnonzero(command!=measured)]
    return command,adjustments


def send_arm_targets(robot,left_deg,right_deg,*,position_mode=False):
    """Commit both arm targets only after every SDK setter succeeds."""
    commands=[(arm,np.asarray(values,dtype=float)) for arm,values in (('A',left_deg),('B',right_deg))]
    if any(q.shape!=(7,) or not np.isfinite(q).all() for _,q in commands):
        raise ValueError('Expected two finite seven-joint targets')
    if not robot.clear_set():raise RuntimeError('Tianji clear_set busy/failed before return target')
    for arm,q in commands:
        if not robot.set_joint_cmd_pose(arm=arm,joints=q.tolist()):
            raise RuntimeError(f'Tianji set_joint_cmd_pose failed for arm {arm}')
        if position_mode:
            if not robot.set_vel_acc(arm=arm,velRatio=20,AccRatio=20):
                raise RuntimeError(f'Tianji set_vel_acc failed for arm {arm}')
            if not robot.set_state(arm=arm,state=1):
                raise RuntimeError(f'Tianji set_state position failed for arm {arm}')
    if not robot.send_cmd():raise RuntimeError('Tianji send_cmd failed for return target')


def request_idle(robot):
    """A busy/failed idle transaction is retried without sending a partial request."""
    if not robot.clear_set():return 'clear_set'
    for arm in ('A','B'):
        if not robot.set_state(arm=arm,state=0):return f'set_state {arm}'
    if not robot.send_cmd():return 'send_cmd'
    return None


def confirm_idle(backend,timeout=5.):
    """Retry only the idle request; never clear faults or send pose targets."""
    deadline=time.monotonic()+timeout;next_request=0.;attempts=0;last_failure=None
    while time.monotonic()<deadline:
        now=time.monotonic()
        if now>=next_request:
            last_failure=request_idle(backend.robot)
            attempts+=1;next_request=now+.5
        states=backend._subscribe()['states']
        if all(int(s['cur_state'])==0 for s in states):return attempts
        time.sleep(.02)
    detail=f'; last transaction failure: {last_failure}' if last_failure is not None else ''
    raise RuntimeError('Idle not acknowledged after repeated requests'+detail)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.output.exists():parser.error('Output already exists; preserve the previous return log')
    target,profile=load_reference(args.reference)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not args.execute:
        report=dict(status='preview',execute=False,hardware_commands=0,reference=str(args.reference),demonstration=profile)
        args.output.write_text(json.dumps(report,indent=2))
        print(json.dumps(report))
        return
    from mujoco_marvin.tianji_backend import TianjiJointBackend,DEFAULT_JOINT_LIMITS_DEG
    from mujoco_marvin.sharpa_backend import load_sharpa,discover,LOWER_RAD,UPPER_RAD
    from policy_hands import PolicyHands
    root=args.root
    import hashlib
    urdf=root/'assets/robot/marvin_dual_arm.urdf'
    if profile['urdf_sha256']!=hashlib.sha256(urdf.read_bytes()).hexdigest():
        raise ValueError('Demonstration and return URDF hashes differ')
    limits=np.vstack([DEFAULT_JOINT_LIMITS_DEG['left'],DEFAULT_JOINT_LIMITS_DEG['right'],np.rad2deg(np.column_stack([LOWER_RAD,UPPER_RAD])),np.rad2deg(np.column_stack([LOWER_RAD,UPPER_RAD]))])
    load_reference(args.reference,limits)
    backend=manager=control=None;owned=False;serials=[]
    report={'target_deg':target.tolist(),'samples':[],'reference':str(args.reference),'demonstration':profile}
    def interrupt(signum,frame):raise RuntimeError('Interrupted')
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    try:
     backend=TianjiJointBackend('192.168.1.190',sdk_dir=str(root/'sdk/tianji'),initialize=False)
     if not backend._verify_connection():raise RuntimeError('Feedback not advancing')
     sub=backend._subscribe()
     if any(int(s['err_code']) or int(s['cur_state']) not in (0,1) for s in sub['states']):raise RuntimeError('Controller not ready')
     sdk=load_sharpa();manager,devices=discover(sdk);by_sn={d.sn:d for d in devices};hands={}
     for side,sn in [('left','C65E9038C65F'),('right','CA56933FCA57')]:
      if sn not in by_sn or by_sn[sn].hand_side!=getattr(sdk.HandSide,side.upper()):raise RuntimeError('Hand identification failed')
      hands[side]=manager.connect(sn,skip_tactile=True);serials.append(sn)
     control=PolicyHands(sdk,hands)
     if np.any(target<limits[:,0]) or np.any(target>limits[:,1]):raise RuntimeError('Reference outside limits')
     control.acquire();left,right=backend.read_joints_deg();measured_initial=np.r_[left,right,np.rad2deg(control.read())]
     initial,adjustments=prepare_return_start(measured_initial,limits)
     report.update(measured_initial_deg=measured_initial.tolist(),initial_hand_limit_adjustments=adjustments)
     delta=float(np.max(np.abs(target-initial)));duration=max(2.,1.5*delta/15,np.sqrt(6*delta/22.5));steps=int(np.ceil(duration*30));report.update(duration_s=steps/30,initial_deg=initial.tolist())
     owned=True
     send_arm_targets(backend.robot,left,right,position_mode=True)
     deadline=time.monotonic()+5
     while any(int(s['cur_state'])!=1 for s in backend._subscribe()['states']):
      if time.monotonic()>deadline:raise RuntimeError('Mode not acknowledged')
      time.sleep(.02)
     start=time.monotonic();previous=initial
     for i in range(steps+31):
      sub=backend._subscribe();left,right=backend.read_joints_deg();measured=np.r_[left,right,np.rad2deg(control.read())]
      if any(int(s['err_code']) or int(s['cur_state'])!=1 for s in sub['states']):raise RuntimeError('Controller fault/mode change')
      if not np.isfinite(measured).all() or np.max(np.abs(measured-previous))>20:raise RuntimeError('Tracking error')
      if time.monotonic()-start-i/30>.1:raise RuntimeError('Deadline miss')
      u=min(1.,i/steps);q=initial+(3*u*u-2*u*u*u)*(target-initial)
      # Validate both hands before sending either arm's target.
      hand_rad=np.deg2rad(q[14:]);lo=np.tile(LOWER_RAD,2);hi=np.tile(UPPER_RAD,2)
      if np.any(hand_rad<lo-1e-12) or np.any(hand_rad>hi+1e-12):raise RuntimeError('Return target outside native hand limits')
      control.check_ownership()
      send_arm_targets(backend.robot,q[:7],q[7:14])
      # Clamp only floating-point roundoff from the degree/radian conversion.
      control.write(np.clip(hand_rad,lo,hi));previous=q
      report['samples'].append({'t':time.monotonic(),'measured_deg':measured.tolist(),'command_deg':q.tolist()})
      time.sleep(max(0,start+(i+1)/30-time.monotonic()))
     left,right=backend.read_joints_deg();measured=np.r_[left,right,np.rad2deg(control.read())]
     report['arm_error_deg']=float(np.max(np.abs(measured[:14]-target[:14])));report['hand_error_deg']=float(np.max(np.abs(measured[14:]-target[14:])))
     if report['arm_error_deg']>.2 or report['hand_error_deg']>3:raise RuntimeError('Initial pose not reached')
     report['status']='completed'
    except BaseException as exc:report.update(status='aborted',error=str(exc))
    finally:
     if owned:
      try:
       report['idle_request_attempts']=confirm_idle(backend)
       report['arms_shutdown']=[0,0]
      except Exception as exc:report['shutdown_error']=str(exc)
     if control:
      try:control.close();report['hands_shutdown']='disabled; SDK source'
      except Exception as exc:report['hand_shutdown_error']=str(exc)
     if backend:
      try:backend.close()
      except Exception as exc:report['backend_close_error']=str(exc)
     if manager:
      failures={}
      for sn in serials:
       try:
        if manager.disconnect(sn) is False:raise RuntimeError('Sharpa disconnect failed')
       except Exception as exc:failures[sn]=str(exc)
      if failures:report['manager_disconnect_errors']=failures
     args.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)
    if report['status']!='completed' or any(report.get(key) for key in ('shutdown_error','hand_shutdown_error','backend_close_error','manager_disconnect_errors')):raise SystemExit(1)


if __name__=='__main__':main()
