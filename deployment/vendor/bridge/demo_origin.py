"""Read measured demonstration starts; never import a robot backend or send commands."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def nearest_indices(source_ns, query_ns, max_skew_ms=50.):
    source_ns, query_ns = np.asarray(source_ns, np.int64), np.asarray(query_ns, np.int64)
    if source_ns.ndim != 1 or not len(source_ns) or np.any(np.diff(source_ns) <= 0):
        raise ValueError('Hand timestamps must be strictly increasing')
    right = np.searchsorted(source_ns, query_ns).clip(0, len(source_ns)-1)
    left = np.maximum(0, right-1)
    index = np.where(abs(source_ns[left]-query_ns) <= abs(source_ns[right]-query_ns), left, right)
    skew = abs(source_ns[index]-query_ns)/1e6
    if np.any(skew > max_skew_ms):
        raise ValueError('Demonstration arm/hand alignment exceeds 50ms')
    return index, skew


def stable_start(times_ns, positions, *, window_s=.4, search_s=1.5,
                 arm_range_deg=2., hand_range_deg=5.):
    """Choose an actual sample inside the earliest stable initial window."""
    times_ns, q = np.asarray(times_ns, np.int64), np.asarray(positions, float)
    if q.shape != (len(times_ns), 58) or not len(q) or not np.isfinite(q).all():
        raise ValueError('Expected finite measured [T,58] joints')
    if np.any(np.diff(times_ns) <= 0):raise ValueError('Arm timestamps must be strictly increasing')
    times = (times_ns-times_ns[0])/1e9
    for lo in np.flatnonzero(times <= search_s):
        hi = np.searchsorted(times, times[lo]+window_s, side='left')
        if hi >= len(times):break
        rows = q[lo:hi+1]
        if len(rows) < 6 or np.max(np.diff(times[lo:hi+1])) > .15:continue
        ranges = np.ptp(rows, axis=0)
        if ranges[:14].max() > arm_range_deg or ranges[14:].max() > hand_range_deg:continue
        # Do not manufacture a pose by averaging different timestamps or episodes.
        scale = np.r_[np.full(14, arm_range_deg), np.full(44, hand_range_deg)]
        score = np.mean(((rows-np.median(rows, axis=0))/scale)**2, axis=1)
        index = lo+int(np.argmin(score))
        return index, dict(first_row=int(lo),last_row=int(hi),start_s=float(times[lo]),
                           end_s=float(times[hi]),arm_range_deg=float(ranges[:14].max()),
                           hand_range_deg=float(ranges[14:].max()))
    raise ValueError('No stable initial window in the first 1.5 seconds')


def read_start(episode):
    import h5py
    episode = Path(episode)
    with h5py.File(episode/'robot_states.h5', 'r') as f:
        arms, hands = f['robot_state/pico_teleop'], f['robot_state/sharpa_joints']
        times = arms['sys_time_ns'][:]
        # Read the initial search region plus the stability window only.
        count = int(np.searchsorted(times, times[0]+2_100_000_000))
        times = times[:count]
        hand_rows, skew = nearest_indices(hands['sys_time_ns'][:], times)
        hq = np.c_[hands['left_rad'][:], hands['right_rad'][:]]
        q = np.c_[arms['real_left_deg'][:count], arms['real_right_deg'][:count], np.rad2deg(hq[hand_rows])]
        index, stable = stable_start(times, q)
        serials = {}
        for side in ('left','right'):
            serial = hands[side+'_sn'][int(hand_rows[index])]
            serials[side] = serial.decode() if isinstance(serial, bytes) else str(serial)
    return dict(episode=episode.name,episode_path=str(episode.resolve()),arm_row=int(index),
                hand_row=int(hand_rows[index]),timestamp_ns=int(times[index]),
                hand_skew_ms=float(skew[index]),stable_window=stable,
                hand_serials=serials,target_deg=q[index].tolist())


def load_reference(path, limits=None):
    profile = json.loads(Path(path).read_text())
    if profile.get('schema') != 'openwam_demo_origin_v1' or profile.get('units') != 'deg':
        raise ValueError('Expected a demonstration origin profile in degrees')
    if profile.get('joint_order') != ['left_arm7','right_arm7','left_hand22','right_hand22']:
        raise ValueError('Origin joint order mismatch')
    if profile.get('source_kind') != 'measured_demonstration':
        raise ValueError('Origin must come from measured demonstration feedback')
    q = np.asarray(profile['target_deg'],float)
    if q.shape != (58,) or not np.isfinite(q).all():raise ValueError('Invalid origin target')
    if limits is not None and (np.any(q < limits[:,0]) or np.any(q > limits[:,1])):
        raise ValueError('Demonstration origin outside native joint limits')
    source = Path(profile['episode_path'])/'robot_states.h5'
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != profile['source_sha256']:
        raise ValueError('Demonstration source missing or changed')
    import h5py
    with h5py.File(source,'r') as f:
        a,h=f['robot_state/pico_teleop'],f['robot_state/sharpa_joints']
        ai,hi=profile['arm_row'],profile['hand_row']
        measured=np.r_[a['real_left_deg'][ai],a['real_right_deg'][ai],
                       np.rad2deg(h['left_rad'][hi]),np.rad2deg(h['right_rad'][hi])]
        if not np.allclose(q,measured,atol=1e-5,rtol=0):raise ValueError('Origin differs from its measured source row')
        if int(a['sys_time_ns'][ai])!=profile['timestamp_ns'] or abs(int(a['sys_time_ns'][ai])-int(h['sys_time_ns'][hi]))>50_000_000:
            raise ValueError('Origin source time alignment mismatch')
        for side,expected in [('left','C65E9038C65F'),('right','CA56933FCA57')]:
            sn=h[side+'_sn'][hi];sn=sn.decode() if isinstance(sn,bytes) else str(sn)
            if sn!=expected:raise ValueError('Demonstration hand serial/side mismatch')
    return q, profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--urdf',type=Path,required=True)
    p.add_argument('--legacy-arms-log',type=Path,required=True)
    p.add_argument('--legacy-hands-log',type=Path,required=True)
    p.add_argument('--candidate-episodes',nargs='+',default=['episode_0084','episode_0149','episode_0212','episode_0232'])
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from kinematics import URDFKinematics
    kin = URDFKinematics(args.urdf)
    # Obtain joint names from the same chain used by the training FK.
    names={side:[j.get('name') for j in kin.chain('dual_arm_mount','TCP_Link_'+tip)
                 if j.get('type')!='fixed'] for side,tip in [('left','L'),('right','R')]}
    def tcp(q):
        return {side:kin.forward(np.deg2rad(np.asarray(q)[None,offset:offset+7]),names[side],
                    'dual_arm_mount','TCP_Link_'+tip)[0].tolist()
                for side,offset,tip in [('left',0,'L'),('right',7,'R')]}
    a=json.loads(args.legacy_arms_log.read_text())['samples'][0]
    h=json.loads(args.legacy_hands_log.read_text())['samples'][0]
    old=np.r_[a['left_deg'],a['right_deg'],h['hand_deg']];old_tcp=tcp(old)
    rows=[];failures=[]
    for episode in sorted(args.episodes_root.glob('episode_*')):
        try:
            row=read_start(episode);row['tcp_mount']=tcp(row['target_deg'])
            row['height_above_legacy_mm']={s:1000*(row['tcp_mount'][s][2][3]-old_tcp[s][2][3]) for s in names}
            rows.append(row)
        except (ValueError,KeyError,OSError) as exc:failures.append(dict(episode=episode.name,error=str(exc)))
    candidates=[x for x in rows if x['episode'] in args.candidate_episodes]
    if not candidates:raise ValueError('No reviewed demonstration has a stable start')
    features=np.array([x['target_deg'][:14] for x in candidates])
    distances=np.mean(abs(features[:,None]-features[None,:]),axis=2)
    selected=dict(candidates[int(np.argmin(distances.sum(axis=1)))])
    selected.update(schema='openwam_demo_origin_v1',units='deg',source_kind='measured_demonstration',
                    joint_order=['left_arm7','right_arm7','left_hand22','right_hand22'],
                    source_sha256=hashlib.sha256((Path(selected['episode_path'])/'robot_states.h5').read_bytes()).hexdigest(),
                    urdf_sha256=hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
                    selection='joint-space medoid of stable starts in previously reviewed episodes',
                    candidate_episodes=[x['episode'] for x in candidates])
    (args.output/'demo_origin.json').write_text(json.dumps(selected,indent=2))
    report=dict(hardware_commands=0,source='measured_feedback',legacy_tcp_mount=old_tcp,
                selected_episode=selected['episode'],rows=rows,excluded=failures)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(episodes=len(rows),excluded=len(failures),selected=selected['episode'],
                         selected_height_above_legacy_mm=selected['height_above_legacy_mm'],hardware_commands=0)))


if __name__=='__main__':main()
