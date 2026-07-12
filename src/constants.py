from .tasks.mix_group_addition import MixCyclicGroupAddition
from .tasks.mix_group_addition import MixRosetteGroupAddition
from .tasks.mix_group_addition import MixDihedralGroupAddition
from .tasks.mix_group_addition import MixMonoidAddition

TASK_MAP = {
    "mixcyclic": MixCyclicGroupAddition,
    "mixdihedral": MixDihedralGroupAddition,
    "mixrosette": MixRosetteGroupAddition,
    "mixmonoid": MixMonoidAddition
}
