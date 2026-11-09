"""Request-local Wan text-context KV reuse, with unchanged 10-step denoising."""
from types import MethodType
import torch


def install_context_cache(model, *, verify_context=False):
    architecture=model.engine.architecture
    if type(architecture).__name__!='DualSystemSelfAttnArchitecture':
        raise ValueError('Context cache only audited for the dual self-attention architecture')
    modules=[block.cross_attn for block in architecture.video_backbone.dit.blocks]
    if any(module.has_image_input for module in modules):
        raise ValueError('Image-token cross attention requires a separate cache implementation')
    driver=architecture._mot_driver or architecture.build_mot_driver()

    def cached_forward(self,x,y,ctx_mask=None):
        q=self.norm_q(self.q(x))
        return self.o(self.attn(q,self._deploy_k,self._deploy_v,attn_mask=ctx_mask))

    for module in modules:
        module.forward=MethodType(cached_forward,module)
    def loop(vstate,astate):
        return driver.run_joint_loop(vstate,astate,use_gradient_checkpointing=False,
                                     use_gradient_checkpointing_offload=False)
    compiled=torch.compile(loop,mode='reduce-overhead',dynamic=False)
    state={'ready':False,'context':None,'generations':0}
    def dispatch(vstate,astate):
        if not state['ready']:
            for module in modules:
                k=module.norm_k(module.k(vstate.context));v=module.v(vstate.context)
                if not hasattr(module,'_deploy_k'):
                    module._deploy_k=k.clone();module._deploy_v=v.clone()
                else:
                    if module._deploy_k.shape!=k.shape or module._deploy_v.shape!=v.shape:
                        raise ValueError('Context-cache shape changed; rebuild the worker explicitly')
                    module._deploy_k.copy_(k);module._deploy_v.copy_(v)
            state['context']=vstate.context.detach().clone() if verify_context else None
            state['ready']=True
        elif verify_context and not torch.equal(state['context'],vstate.context):
            raise RuntimeError('Context changed inside a generation; cache is invalid')
        return compiled(vstate,astate)
    architecture._compiled_mot_run_joint_loop=dispatch
    original=model.engine.generate
    def generate(conditions):
        state['ready']=False;state['context']=None;state['generations']+=1
        result=original(conditions)
        if architecture._compiled_mot_run_joint_loop is not dispatch:
            raise RuntimeError('Compiled context-cache path fell back; reject this optimization')
        return result
    model.engine.generate=generate
    return state
