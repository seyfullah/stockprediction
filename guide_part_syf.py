import os
import torch
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from StockDataset import StockDataset

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
else:
    torch.manual_seed(0)
    device = "cpu"

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Simulation time.
time = 500

# Create the network.
network = Network()

# Create and add input, output layers.
source_layer = Input(n=8)
target_layer = LIFNodes(n=5)

network.add_layer(layer=source_layer, name="A")
network.add_layer(layer=target_layer, name="B")

# Create connection between input and output layers.
forward_connection = Connection(
    source=source_layer,
    target=target_layer,
    w=0.05 + 0.1 * torch.randn(source_layer.n, target_layer.n),  # Normal(0.05, 0.01) weights.
)

network.add_connection(connection=forward_connection, source="A", target="B")

# Create recurrent connection in output layer.
recurrent_connection = Connection(
    source=target_layer,
    target=target_layer,
    w=0.025 * (torch.eye(target_layer.n) - 1),  # Small, inhibitory "competitive" weights.
)

network.add_connection(connection=recurrent_connection, source="B", target="B")

# Create and add input and output layer monitors.
source_monitor = Monitor(
    obj=source_layer,
    state_vars=("s",),  # Record spikes and voltages.
    time=time,  # Length of simulation (if known ahead of time).
)
target_monitor = Monitor(
    obj=target_layer,
    state_vars=("s", "v"),  # Record spikes and voltages.
    time=time,  # Length of simulation (if known ahead of time).
)

network.add_monitor(monitor=source_monitor, name="A")
network.add_monitor(monitor=target_monitor, name="B")

# Create input spike data, where each spike is distributed according to Bernoulli(0.1).
# input_data = torch.bernoulli(0.1 * torch.ones(time, source_layer.n)).byte()
stock_dataset = StockDataset(csv_file='AAPL_data.csv')

(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = stock_dataset.get_feature_importance_data()



# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    return torch.from_numpy(df.values).float().to(device)


X_train_FI_tensor = df_to_tensor(X_train_FI)
y_train_FI_tensor = df_to_tensor(y_train_FI)

inputs = {"A": X_train_FI_tensor}

# Simulate network on input data.
network.run(inputs=inputs, time=time)

# Retrieve and plot simulation spike, voltage data from monitors.
spikes = {
    "A": source_monitor.get("s"), "B": target_monitor.get("s")
}
voltages = {"B": target_monitor.get("v")}

plt.ioff()
plot_spikes(spikes)
plot_voltages(voltages, plot_type="line")
plt.show()
