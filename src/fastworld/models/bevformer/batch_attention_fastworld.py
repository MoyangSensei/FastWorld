"""Replace only private FastWorld module instances; never alter the registry."""
from .spatial_cross_attention import SpatialCrossAttention
from .temporal_self_attention import TemporalSelfAttention
from .encoder import BEVFormerEncoder
from .spatial_cross_attention_fastworld import SpatialCrossAttentionFastWorld
from .temporal_self_attention_fastworld import TemporalSelfAttentionFastWorld
from .bev_encoder_fastworld import BEVFormerEncoderFastWorld

def use_batch_independent_attention_fastworld(model):
    mapping={SpatialCrossAttention:SpatialCrossAttentionFastWorld,TemporalSelfAttention:TemporalSelfAttentionFastWorld,BEVFormerEncoder:BEVFormerEncoderFastWorld}
    changed=[]
    for parent in model.modules():
        for name,child in list(parent.named_children()):
            cls=mapping.get(type(child))
            if cls is not None:
                parent._modules[name]=cls.from_existing(child)
                changed.append(cls.__name__)
    return changed
