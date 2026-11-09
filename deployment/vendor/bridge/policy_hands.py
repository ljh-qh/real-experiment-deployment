"""Explicit GLOVE -> SDK ownership; leave hands disabled under SDK at shutdown."""
import time
import numpy as np
from mujoco_marvin.sharpa_backend import SharpaReplayBackend,check_error

class PolicyHands(SharpaReplayBackend):
    def __init__(self,sdk,hands):
        self.sdk=sdk;self.hands=hands;self.manager=None;self._owned=set();self.initial_state={}
        for side,hand in hands.items():
            err,source=hand.get_control_source();check_error(err,'source read')
            err,mode=hand.get_control_mode();check_error(err,'mode read')
            self.initial_state[side]=(source,mode)
        self.read()

    def acquire(self):
        self._owned.update(self.hands)
        for side,hand in self.hands.items():
            check_error(hand.set_enable_state(False),side+' disable before transfer')
        self.wait_disabled()
        self.enable()
        deadline=time.monotonic()+2
        while True:
            try: self.check_ownership();break
            except RuntimeError:
                if time.monotonic()>deadline: raise
                time.sleep(.02)

    def wait_disabled(self):
        deadline=time.monotonic()+3
        while True:
            enabled=[]
            for side,hand in self.hands.items():
                err,value=hand.get_enable_state();check_error(err,'disable confirmation');enabled.append(value)
            if not any(enabled): return
            if time.monotonic()>deadline: raise RuntimeError('Hands did not disable')
            time.sleep(.02)

    def check_ownership(self):
        for side,hand in self.hands.items():
            err,source=hand.get_control_source();check_error(err,side+' source')
            err,enabled=hand.get_enable_state();check_error(err,side+' enabled')
            err,mode=hand.get_control_mode();check_error(err,side+' mode')
            if source!=self.sdk.ControlSource.SDK or not enabled or mode!=self.sdk.ControlMode.POSITION:
                raise RuntimeError(side+' hand lost SDK position ownership')
        self.read()

    def close(self):
        if not self._owned: return
        failures=[]
        for side in self._owned:
            try: check_error(self.hands[side].set_enable_state(False),side+' disable')
            except Exception as exc: failures.append(str(exc))
        try: self.wait_disabled()
        except Exception as exc: failures.append(str(exc))
        if failures: raise RuntimeError('; '.join(failures))
        self._owned.clear()
