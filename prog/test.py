import matplotlib.pyplot as plt
import numpy as np

# 创建一些示例数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建第一个y轴
fig, ax1 = plt.subplots()

# 绘制第一个y轴的数据
line1, = ax1.plot(x, y1, 'b-', label='sin(x)')
ax1.set_xlabel('X轴')
ax1.set_ylabel('sin(x)', color='b')
ax1.tick_params('y', colors='b')

# 创建第二个y轴
ax2 = ax1.twinx()

# 绘制第二个y轴的数据
line2, = ax2.plot(x, y2, 'r-', label='cos(x)')
ax2.set_ylabel('cos(x)', color='r')
ax2.tick_params('y', colors='r')

# 将标签固定在坐标轴上
ax1.yaxis.set_label_coords(-0.1, 0.5)
ax2.yaxis.set_label_coords(1.1, 0.5)

# 在图形上表示x轴
ax1.axhline(0, color='black',linewidth=0.5)
ax2.axhline(0, color='black',linewidth=0.5)

# 添加图例
lines = [line1, line2]
labels = [line.get_label() for line in lines]
plt.legend(lines, labels, loc='upper right')

plt.title('图形标题')
plt.show()