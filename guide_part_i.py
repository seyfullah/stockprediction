from bindsnet.network import Network

network = Network()

from bindsnet.network.nodes import LIFNodes

# Create a layer of 100 LIF neurons with shape (10, 10).
layer = LIFNodes(n=100, shape=(10, 10))

network.add_layer(
    layer=layer, name="LIF population"
)
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

# Create two populations of neurons, one to act as the "source"
# population, and the other, the "target population".
source_layer = Input(n=100)
target_layer = LIFNodes(n=1000)

# Connect the two layers.
connection = Connection(
    source=source_layer, target=target_layer
)

network.add_layer(
    layer=source_layer, name="A"
)
network.add_layer(
    layer=target_layer, name="B"
)
network.add_connection(
    connection=connection, source="A", target="B"
)

from bindsnet.network.monitors import Monitor

# Create a monitor.
monitor = Monitor(
    obj=target_layer,
    state_vars=("s", "v"),  # Record spikes and voltages.
    time=500,  # Length of simulation (if known ahead of time).
)

network.add_monitor(monitor=monitor, name="B")

from bindsnet.network.monitors import NetworkMonitor

network_monitor = NetworkMonitor(
    network,
    network.layers,
    connection,
    state_vars=("s", "v"),
    time= 500,
)
