import pandas as pd
import random
import numpy as np

#从数据集中读取出租车和乘客基本信息
def get_from_txt(filename):
    with open(filename, 'r') as f:
        lines=f.readlines()
    lines_people=lines[:1080]
    lines_taxi=lines[1080:]

    taxi_list,passenger_list=list(),list()
    i=0 #乘客编号
    j=0 #出租车编号
    for one in lines_people:
        a=one.split(',')[0:-1]

        for b in a[2:]:
            c=b.split(' ')
            passenger_list.append([i,a[1],c[1],c[2],c[3],c[4],c[5]]) #乘客编号，出现时间（第几分钟），能够最长等待的时间，出现xy，目的地xy
            i+=1
    for one in lines_taxi:
        a=one.split(',')[0:-1]
        for b in a[2:]:
            c = b.split(' ')
            taxi_list.append([j,a[1],c[1],c[2]]) #出租车编号，出现时间（第几分钟)，出现xy，
    return taxi_list,passenger_list

class TaxiDispatchEnv():
    def __init__(self, data_file):
        self.data_file = data_file
        self.current_time = 0
        self.city_size = (10000, 10000)  # 城市大小，单位为米 (10km * 10km)
        self.grid_size = (1000, 1000)  # 每块区域的大小，单位为米 (1km * 1km)
        self.num_grids = (self.city_size[0] // self.grid_size[0], self.city_size[1] // self.grid_size[1])
        self.max_passengers = 7000  # 假设的最大乘客数
        self.max_taxis = 1641  # 假设的最大出租车数
        self.observation_space = self.calculate_observation_space()
        self.action_space = self.calculate_action_space()
        self.passengers = []  # 乘客列表
        self.taxis = []  # 出租车列表

        self.total_reward = 0  # 记录总奖励
        self.total_income = 0  # 记录总收入
        self.total_wait_time = 0  # 记录总等待时间
        self.policy_actions = []  # 记录每个时间点采取的动作

        self.load_data()

    def calculate_observation_space(self):
        # 归一化后的时间 + 每个乘客的5个状态 + 每个出租车的3个状态
        return 1 + 5 * self.max_passengers + 3 * self.max_taxis

    def calculate_action_space(self):
        # 定义的动作数量
        return 6  # 例如，0到5共6个动作
    def load_data(self):
        taxi_list, passenger_list=get_from_txt(self.data_file)

        for details in passenger_list:
            self.passengers.append({
                'id': details[0],
                'time': int(details[1]),
                'max_wait': int(details[2]),
                'start_x': float(details[3]),
                'start_y': float(details[4]),
                'target_x': float(details[5]),
                'target_y': float(details[6]),
                'status': 'waiting',
                'wait_time': 0
            })
        for details in taxi_list:
            self.taxis.append({
                'id': details[0],
                'start_time': int(details[1]),
                'start_x': float(details[2]),
                'start_y': float(details[3]),
                'status': 'idle'
            })

    def step(self, action):
        # 根据动作更新出租车和乘客状态
        if action == 0:
            self.default_action()
        elif action == 1:
            self.action_one()
        elif action == 2:
            self.action_two()
        elif action == 3:
            self.action_three()
        elif action == 4:
            self.action_four()
        elif action == 5:
            self.action_five()
        if random.random() < 0.1:  # 假设每个时间步有10%的概率出现新乘客
            self.generate_new_passenger()
        # 更新时间和状态
        self.update_status()

        # 检查是否结束
        done = self.current_time >= 1080

        # 计算奖励
        reward = self.calculate_reward()

        return self.get_state(), reward, done,self.get_training_record()

    def get_training_record(self):
        # 获取训练记录
        return {
            "total_reward": self.total_reward,
            "total_income": self.total_income,
            "total_wait_time": self.total_wait_time,
            "policy_actions": self.policy_actions
        }
    def update_status(self):
        self.current_time += 1
        for passenger in self.passengers:
            if passenger['status'] == 'waiting':
                passenger['wait_time'] += 1
                self.total_wait_time += 1  # 累加等待时间
                if passenger['wait_time'] > passenger['max_wait']:
                    passenger['status'] = 'abandoned'

        for taxi in self.taxis:
            if taxi['status'] == 'serving':
                # 更新出租车位置到乘客目的地
                current_passenger = next((p for p in self.passengers if p['id'] == taxi['current_passenger_id']), None)
                if current_passenger and current_passenger['status'] == 'picked_up':
                    taxi['start_x'], taxi['start_y'] = current_passenger['target_x'], current_passenger['target_y']
                    # 当乘客到达目的地时
                    if taxi['start_x'] == current_passenger['target_x'] and taxi['start_y'] == current_passenger['target_y']:
                        # 计算旅程距离
                        distance = self.calculate_distance(taxi['start_x'], taxi['start_y'], current_passenger['target_x'], current_passenger['target_y'])

                        # 基于距离计算收入
                        income = distance * 0.1  # 假设每米0.1单位的收入
                        self.total_income += income
                        # 更新乘客状态
                        current_passenger['status'] = 'arrived'
                        # 更新出租车状态
                        taxi['status'] = 'idle'
                        taxi['current_passenger_id'] = None

            elif taxi['status'] == 'idle':
                if self.current_time - taxi['start_time'] >= 480:  # 假设出租车的工作时间不超过8小时
                    taxi['status'] = 'off_duty'


    def calculate_reward(self):
        reward = 0
        for passenger in self.passengers:
            if passenger['status'] == 'arrived':
                distance = np.sqrt((passenger['start_x'] - passenger['target_x']) ** 2 + (
                            passenger['start_y'] - passenger['target_y']) ** 2)
                time_bonus = max(0, passenger['max_wait'] - passenger['wait_time'])
                reward += 20 + time_bonus - distance * 0.1  # 成功送达乘客的基础奖励，加上时间窗口奖励，减去基于距离的惩罚

            elif passenger['status'] == 'waiting':
                reward -= 0.5 * passenger['wait_time']  # 等待时间过长的惩罚

        for taxi in self.taxis:
            if taxi['status'] == 'idle':
                reward -= 1  # 对空闲出租车施加小额惩罚

        return reward

    def reset(self):
        self.current_time = 0
        self.passengers = []
        self.taxis = []
        self.load_data()
        return self.get_state()

    def get_state(self):
        # 初始化状态数组
        state_size = self.calculate_observation_space()
        state = np.zeros(state_size)


        # 当前时间
        state[0] = self.current_time / 1080  # 归一化时间

        # 乘客信息
        for i, passenger in enumerate(self.passengers):
            offset = 1 + i * 5
            state[offset] = passenger['start_x'] / self.city_size[0]  # 归一化位置
            state[offset + 1] = passenger['start_y'] / self.city_size[1]
            state[offset + 2] = passenger['target_x'] / self.city_size[0]
            state[offset + 3] = passenger['target_y'] / self.city_size[1]
            state[offset + 4] = passenger['wait_time'] / passenger['max_wait']  # 归一化等待时间

        # 出租车信息
        for i, taxi in enumerate(self.taxis):
            offset = 1 + self.max_passengers * 5 + i * 3
            state[offset] = taxi['start_x'] / self.city_size[0]  # 归一化位置
            state[offset + 1] = taxi['start_y'] / self.city_size[1]
            state[offset + 2] = 1 if taxi['status'] == 'idle' else 0  # 二进制状态

        return state

    def render(self):
        print(f"Time: {self.current_time}")
        print("Taxis:")
        for taxi in self.taxis:
            print(f"  Taxi {taxi['id']}: Position ({taxi['start_x']}, {taxi['start_y']}), Status: {taxi['status']}")
            break #仅输出了第一个出租车的信息，若要全部输出，删掉break
        print("Passengers:")
        for passenger in self.passengers:
            print(f"  Passenger {passenger['id']}: Status: {passenger['status']}, Wait Time: {passenger['wait_time']}")
            break #仅输出了第一个乘客的信息，若要全部输出，删掉break

    def default_action(self):
        for taxi in self.taxis:
            if taxi['status'] == 'idle':
                nearest_passenger = self.find_nearest_passenger(taxi)
                if nearest_passenger:
                    self.update_taxi_and_passenger_status(taxi, nearest_passenger)

    def action_one(self):
        for taxi in self.taxis:
            if taxi['status'] == 'idle':
                nearest_passenger = self.find_nearest_passenger_within_radius(taxi, 1)
                if nearest_passenger:
                    # 更新出租车和乘客状态
                    self.update_taxi_and_passenger_status(taxi, nearest_passenger)

    def action_two(self):
        for taxi in self.taxis:
            if taxi['status'] == 'idle':
                dense_area = self.find_dense_area_nearby(taxi)
                taxi['start_x'], taxi['start_y'] = dense_area

    def action_three(self):
        for taxi in self.taxis:
            if taxi['status'] == 'idle':
                random_area = self.select_random_area()
                taxi['start_x'], taxi['start_y'] = random_area

    def action_four(self):
        for taxi in self.taxis:
            if self.should_move_to_less_dense_area(taxi):
                less_dense_area = self.find_less_dense_area()
                taxi['start_x'], taxi['start_y'] = less_dense_area

    def action_five(self):
        for taxi in self.taxis:
            if self.should_move_to_more_dense_area(taxi):
                more_dense_area = self.find_more_dense_area()
                taxi['start_x'], taxi['start_y'] = more_dense_area

    def find_nearest_passenger(self, taxi):
        min_distance = float('inf')
        nearest_passenger = None
        for passenger in self.passengers:
            if passenger['status'] == 'waiting':
                # Calculate the distance between the taxi and this passenger
                distance = self.calculate_distance(taxi['start_x'], taxi['start_y'],
                                                   passenger['start_x'], passenger['start_y'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_passenger = passenger
        return nearest_passenger

    def find_nearest_passenger_within_radius(self, taxi, radius):
        min_distance = float('inf')
        nearest_passenger = None
        for passenger in self.passengers:
            if passenger['status'] == 'waiting':
                # Calculate the distance between the taxi and this passenger
                distance = self.calculate_distance(taxi['start_x'], taxi['start_y'],
                                                   passenger['start_x'], passenger['start_y'])
                if distance < min_distance and distance <= radius:
                    min_distance = distance
                    nearest_passenger = passenger
        return nearest_passenger

    def calculate_distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def update_taxi_and_passenger_status(self, taxi, passenger):
        # 乘客上车
        if passenger['status'] == 'waiting':
            passenger['status'] = 'picked_up'
            taxi['status'] = 'serving'
            taxi['current_passenger_id'] = passenger['id']

        # 检查是否到达目的地
        elif taxi['status'] == 'serving' and taxi['current_passenger_id'] == passenger['id']:
            if taxi['start_x'] == passenger['target_x'] and taxi['start_y'] == passenger['target_y']:
                passenger['status'] = 'arrived'
                taxi['status'] = 'idle'
                taxi['current_passenger_id'] = None
                # 计算收益等

    def select_random_area(self):
        # 假设城市地图的坐标范围为 0 到 100（示例）
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        return x, y

    def find_dense_area_nearby(self, taxi):
        # 基于某种逻辑确定附近的密集区域
        # 示例：随机选择一个区域作为密集区域
        return random.randint(0, 100), random.randint(0, 100)

    def find_less_dense_area(self):
        # 查找一个较少密集的区域
        # 示例：随机选择一个区域作为较少密集区域
        return random.randint(0, 100), random.randint(0, 100)

    def find_more_dense_area(self):
        # 查找一个更密集的区域
        # 示例：随机选择一个区域作为更密集区域
        return random.randint(0, 100), random.randint(0, 100)

    def should_move_to_less_dense_area(self, taxi):
        # 判断逻辑：如果出租车在某个区域停留超过特定时间且未拾到乘客
        # 示例：随机返回True或False
        return random.choice([True, False])

    def should_move_to_more_dense_area(self, taxi):
        # 同上
        return random.choice([True, False])

    def generate_new_passenger(self):
        # 检查当前乘客数量是否已达到最大限制
        if len(self.passengers) < self.max_passengers:
            new_passenger = {
                'id': len(self.passengers) + 1,
                'time': self.current_time,
                'max_wait': random.randint(5, 15),
                'start_x': random.uniform(0, self.city_size[0]),
                'start_y': random.uniform(0, self.city_size[1]),
                'target_x': random.uniform(0, self.city_size[0]),
                'target_y': random.uniform(0, self.city_size[1]),
                'status': 'waiting',
                'wait_time': 0
            }
            self.passengers.append(new_passenger)


env = TaxiDispatchEnv('./data_small.txt')



# #测试环境
# num_episodes = 10  # 设置训练回合数
# for episode in range(num_episodes):
#     print(f"Episode {episode + 1}:")
#
#     # 重置环境状态
#     state = env.reset()
#
#     for step in range(2000):  # 每个回合的最大步数
#         print(f"  Step {step + 1}:")
#         action = random.choice([0, 1, 2, 3, 4, 5])  # 随机选择一个动作
#         state, reward, done = env.step(action)
#         env.render()  # 可视化当前状态
#
#         if done:
#             break
#     print(f"Episode {episode + 1} finished\n")
