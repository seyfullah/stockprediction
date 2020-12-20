from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import PostPre

source_layer = Input(n=100, traces=True)
target_layer = LIFNodes(n=1000, traces=True)

connection = Connection(
    source=source_layer,
    target=target_layer,
    update_rule=PostPre,
    nu=(1e-4, 1e-2))

