import matplotlib.pyplot as plt
import numpy as np

# Create Random Data

# X-Axis
data_x = np.linspace(0, 1000, 25)

# Y-Axis
np.random.seed(502)
data_t = np.random.rand(25)
increment = 1
data_y = []
for each in data_t:
    data_y.append(each * increment)
    increment += .065 * increment

# State-Based Approach — plt.subplot()
plt.figure(facecolor='lightgrey')
plt.subplot(2, 2, 1)
plt.plot(data_x, data_y, 'r-')
plt.subplot(2, 2, 2)
plt.plot(data_x, data_y, 'b-')
plt.subplot(2, 2, 4)
plt.plot(data_x, data_y, 'g-')
# plt.suptitle("Your Title Here")
# plt.xlabel("X Axis")
# plt.ylabel("Y Axis")
plt.show()

# Object-Oriented Approach — plt.subplots()
fig, ax = plt.subplots(2, 2)
fig.set_facecolor('lightgrey')
ax[0, 0].plot(data_x, data_y, 'r-')
ax[0, 1].plot(data_x, data_y, 'b-')
fig.delaxes(ax[1, 0])
ax[1, 1].plot(data_x, data_y, 'g-')
# fig.suptitle("Your Title Here")
# ax[1,1].set_xlabel("X Axis")
# ax[1,1].set_ylabel("Y Axis")
plt.show()
