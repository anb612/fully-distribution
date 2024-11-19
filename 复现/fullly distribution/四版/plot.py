import matplotlib.pyplot as plt
import numpy as np

def plot_agent_trajectories(agent_trajectories):
    # 创建一个图形
    plt.figure(figsize=(8, 6))
    
    # 遍历字典中的每个agent及其轨迹
    for agent_name, trajectory in agent_trajectories.items():
        # 处理每个元素为2x1的二维数组
        trajectory = np.array([np.array(point).flatten() for point in trajectory])  # 确保每个点是1维数组
        
        # 绘制连线
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=agent_name)
        
        # 标记起始位置
        plt.scatter(trajectory[0, 0], trajectory[0, 1], marker='*', s=100, label=f'{agent_name} start')

    # 添加标题和标签
    plt.title('Agent Trajectories')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 显示图例
    plt.legend(loc='best')
    
    # 展示图形
    plt.grid(True)
    plt.show()

# # 示例用法
# agent_trajectories = {
#     'Agent 1': [np.array([[0], [0]]), np.array([[1], [2]]), np.array([[2], [3]]), np.array([[3], [5]])],
#     'Agent 2': [np.array([[1], [1]]), np.array([[2], [2]]), np.array([[3], [3]]), np.array([[4], [4]])],
# }

# plot_agent_trajectories(agent_trajectories)


def plot_event_intervals(agent_i_name, agent_j_name, event_sleep_time_recording, event_interval_time):
    """
    画出事件间隔图像，其中红色表示 τ12(t12_k)，蓝色表示检查触发条件的持续时间。
    
    参数：
    - time_points: 事件发生的时间点列表
    - red_heights: 每个时间点对应的红色线条高度
    - blue_heights: 每个时间点对应的蓝色线条高度
    """
    plt.figure(figsize=(10, 6))

    # 绘制事件间隔
    t = 0
    for total_height, h_red in zip(event_interval_time, event_sleep_time_recording):
        # 绘制红色线条
        plt.vlines(t, 0, h_red, colors='red', linewidth=4, alpha=0.8)
        plt.scatter(t, h_red, color='red', marker='*', s=100, zorder=3)

        # 绘制蓝色线条
        plt.vlines(t, h_red, total_height, colors='blue', linewidth=2, alpha=0.8)
        plt.scatter(t, total_height, color='blue', marker='o', s=60, zorder=3)
        t += total_height

    # 设置图形标签和标题
    plt.xlabel('Time t')
    plt.ylabel('Event interval')
    plt.title(f"Event intervals of edge ({agent_i_name}, {agent_j_name})")
    plt.grid(True)

    # 显示图形
    plt.show()


# def test_plot_event_intervals():
#     # 测试数据
#     agent_i_name = 'AgentA'
#     agent_j_name = 'AgentB'
#     event_sleep_time_recording = [1, 2, 3]
#     event_interval_time = [4, 5, 6]

#     # 调用函数
#     plot_event_intervals(agent_i_name, agent_j_name, event_sleep_time_recording, event_interval_time)

#     # 获取当前图形并清除它，以便进行测试
#     fig = plt.gcf()
#     fig.canvas.draw()

#     # 将图形转换为图像数据
#     img = BytesIO()
#     fig.savefig(img, format='png')
#     img.seek(0)
#     plot_url = base64.b64encode(img.getvalue()).decode()

#     # 检查图像数据是否有效
#     if plot_url:
#         print("测试通过：图像数据有效")
#     else:
#         print("测试失败：图像数据无效")

# test_plot_event_intervals()



def plot_evolution(time_points_dict, zeta_values):
    """
    画出自适应变量 ζij 的演变曲线。
    
    参数：
    - time_points_dict: 字典，key 为 edge 对象，value 为对应的时间点列表
    - zeta_values: 字典，key 为 edge 对象，value 为对应的 ζij 值列表
    """
    plt.figure(figsize=(10, 6))

    # 绘制每个边的 ζij 演变曲线
    for edge, zeta in zeta_values.items():
        time_points = time_points_dict[edge]
        label = f'ζ{edge.head.index}{edge.tail.index} / ζ{edge.tail.index}{edge.head.index}'
        plt.plot(time_points, zeta, label=label)
        # print(f"edge:{edge.index}")
        # print(f"zeta.shape{(len(zeta))}")
        # print(f"time_points.shape{len(time_points)}")
        

    # 设置图形标签和标题
    plt.xlabel('Time t')
    plt.ylabel('ζij(t) ∈ ℰ')
    plt.title('Evolution of ζij')
    plt.legend()
    plt.grid(True)

    # 显示图形
    plt.show()

def plot_theta_evolution(time_points_dict, theta_values):
    """
    画出自适应变量 θij 的演变曲线。
    
    参数：
    - time_points_dict: 字典，key 为 edge 对象，value 为对应的时间点列表
    - theta_values: 字典，key 为 edge 对象，value 为对应的 θij 值列表，其中包含两个值
    """
    plt.figure(figsize=(10, 6))

    # 绘制每个边的 θij 演变曲线
    for edge, theta in theta_values.items():
        time_points = time_points_dict[edge]
        label_head_tail = f'θ{edge.head.index}{edge.tail.index}'
        label_tail_head = f'θ{edge.tail.index}{edge.head.index}'
        plt.plot(time_points, [val[0] for val in theta], label=label_head_tail)
        plt.plot(time_points, [val[1] for val in theta], label=label_tail_head)

    # 设置图形标签和标题
    plt.xlabel('Time t')
    plt.ylabel('θij(t) ∈ ℰ')
    plt.title('Evolution of θij')
    plt.legend()
    plt.grid(True)

    # 显示图形
    plt.show()