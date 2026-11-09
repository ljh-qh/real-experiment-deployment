"""Hardware-free asynchronous planning and continuous joint command shaping."""
import numpy as np
from scipy.spatial.transform import Rotation
from adapter import decode_action


class FastLeftIK:
    def __init__(self, reference, limits_deg, side="left"):
        self.side=side
        if side not in ("left","right"): raise ValueError("Unknown arm")
        self.chain=[]
        self.bounds=np.deg2rad(np.asarray(limits_deg))
        for joint in reference.chain('dual_arm_mount','TCP_Link_'+('L' if side=='left' else 'R')):
            origin=joint.find('origin')
            xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            tf=np.eye(4); tf[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix(); tf[:3,3]=xyz
            axis=None
            if joint.get('type')!='fixed':
                if joint.get('type') not in ('continuous','revolute'): raise ValueError('Unsupported joint')
                node=joint.find('axis');axis=np.fromstring(node.get('xyz','1 0 0'),sep=' ') if node is not None else np.array([1.,0.,0.])
                axis/=np.linalg.norm(axis)
            self.chain.append((tf,axis))

    def forward_jacobian(self,q):
        tf=np.eye(4); origins=[]; axes=[]; i=0
        for fixed,axis in self.chain:
            tf=tf@fixed
            if axis is not None:
                origins.append(tf[:3,3].copy());axes.append(tf[:3,:3]@axis)
                moving=np.eye(4);moving[:3,:3]=Rotation.from_rotvec(q[i]*axis).as_matrix()
                tf=tf@moving;i+=1
        axes=np.asarray(axes)
        jac=np.vstack([np.cross(axes,tf[:3,3]-np.asarray(origins)).T,axes.T])
        return tf,jac

    def solve(self,target,seed):
        q=np.asarray(seed).copy()
        for _ in range(80):
            tf,jac=self.forward_jacobian(q)
            error=np.r_[target[:3,3]-tf[:3,3],Rotation.from_matrix(target[:3,:3]@tf[:3,:3].T).as_rotvec()]
            if np.linalg.norm(error[:3])<.00005 and np.linalg.norm(error[3:])<np.deg2rad(.05):
                return q
            # Damped geometric Jacobian; local seeds maintain redundant-arm continuity.
            step=jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*1e-6,error)
            step*=min(1.,np.deg2rad(3)/max(np.max(np.abs(step)),1e-12))
            q=np.clip(q+step,self.bounds[:,0],self.bounds[:,1])
        raise ValueError('Policy pose IK failed')

    def plan(self,current_deg,actions):
        actions=np.asarray(actions)
        if actions.shape!=(32,80) or not np.isfinite(actions).all(): raise ValueError('Invalid chunk')
        seed=np.deg2rad(current_deg); points=[]
        for action in actions:
            poses,_=decode_action(action);seed=self.solve(poses[self.side],seed);points.append(np.rad2deg(seed))
        points=np.asarray(points)
        jumps=np.abs(np.diff(points,axis=0))
        if np.max(jumps)>10:
            step,joint=np.unravel_index(np.argmax(jumps),jumps.shape)
            raise ValueError(f'IK discontinuity: side={self.side}, step={step+1}, joint={joint}, delta={jumps[step,joint]:.4f}deg')
        return points


class SmoothCommands:
    """Rate/acceleration bounded joint targets; no TCP radius clipping."""
    def __init__(self,initial,velocity_deg_s=30.,acceleration_deg_s2=90.,dt=1/30):
        self.q=np.asarray(initial,dtype=float).copy();self.v=np.zeros_like(self.q)
        self.vmax=velocity_deg_s;self.amax=acceleration_deg_s2;self.dt=dt
    def step(self,target):
        delta=np.asarray(target)-self.q
        desired=np.sign(delta)*np.minimum(self.vmax,np.sqrt(2*self.amax*np.abs(delta)))
        desired=np.clip(desired,-np.abs(delta)/self.dt,np.abs(delta)/self.dt)
        self.v+=np.clip(desired-self.v,-self.amax*self.dt,self.amax*self.dt)
        self.q+=self.v*self.dt
        return self.q.copy()


def action_index(observation_ns,now_ns,speed=1.):
    # action[t] predicts state[t+1]. Advance along the predicted timeline during inference.
    return max(0,int((now_ns-observation_ns)*speed//(1e9/30))-1)


def sample_slow_chunk(points,initial,observation_ns,now_ns,speed):
    if not 0 < speed <= 1: raise ValueError('Speed must be in (0,1]')
    phase=max(0.,(now_ns-observation_ns)/1e9*30*speed)
    if phase>=len(points)+1: return None
    if phase<1: return np.asarray(initial)+(points[0]-initial)*phase
    position=phase-1
    lo=min(int(position),len(points)-1);hi=min(lo+1,len(points)-1)
    return points[lo]+(points[hi]-points[lo])*(position-lo)
