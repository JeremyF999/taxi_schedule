import pandas as pd
import random
import numpy as np

class TaxiDispatchEnv:
    def __init__(self, data_file):
        self.data_file = data_file
        self.current_time = 0
        self.passengers = []  # 乘客列表
        self.taxis = []  # 出租车列表
        self.load_data()

    def load_data(self):
        data = pd.read_csv(self.data_file, header=None)
        for index, row in data.iterrows():
            entities = row[1].split(',')  # 分割整行数据为单独的实体
            for entity in entities:
                details = entity.split()  # 分割单个实体的内部数据
                if row[0] == 0:  # 乘客信息
                    self.passengers.append({
                        'id': details[0],
                        'time': self.current_time,
                        'max_wait': int(details[1]),
                        'start_x': float(details[2]),
                        'start_y': float(details[3]),
                        'target_x': float(details[4]),
                        'target_y': float(details[5]),
                        'status': 'waiting',
                        'wait_time': 0
                    })
                elif row[0] == 1:  # 出租车信息
                    self.taxis.append({
                        'id': details[0],
                        'start_time': self.current_time,
                        'start_x': float(details[1]),
                        'start_y': float(details[2]),
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

        return self.get_state(), reward, done

    def update_status(self):
        self.current_time += 1
        for passenger in self.passengers:
            if passenger['status'] == 'waiting':
                passenger['wait_time'] += 1
                if passenger['wait_time'] > passenger['max_wait']:
                    passenger['status'] = 'abandoned'

        for taxi in self.taxis:
            if taxi['status'] == 'serving':
                # 更新出租车位置到乘客目的地
                current_passenger = next((p for p in self.passengers if p['id'] == taxi['current_passenger_id']), None)
                if current_passenger and current_passenger['status'] == 'picked_up':
                    taxi['start_x'], taxi['start_y'] = current_passenger['target_x'], current_passenger['target_y']
                    if taxi['start_x'] == current_passenger['target_x'] and taxi['start_y'] == current_passenger[
                        'target_y']:
                        current_passenger['status'] = 'arrived'
                        taxi['status'] = 'idle'
                        taxi['current_passenger_id'] = None
            elif taxi['status'] == 'idle':
                if self.current_time - taxi['start_time'] >= 480:  # 假设出租车的工作时间不超过8小时
                    taxi['status'] = 'off_duty'

    # 更新出租车状态的逻辑
    # ...
    def calculate_reward(self):
        reward = 0
        for passenger in self.passengers:
            if passenger['status'] == 'arrived':
                reward += 20  # 成功送达乘客的奖励
            elif passenger['status'] == 'waiting':
                reward -= 0.5 * passenger['wait_time']  # 等待时间过长的惩罚
        return reward

    def reset(self):
        self.current_time = 0
        self.passengers = []
        self.taxis = []
        self.load_data()
        return self.get_state()

    def get_state(self):
        # 返回当前环境的状态
        return {
            'time': self.current_time,
            'passengers': self.passengers,
            'taxis': self.taxis
        }

    def render(self):
        print(f"Time: {self.current_time}")
        print("Taxis:")
        for taxi in self.taxis:
            print(f"  Taxi {taxi['id']}: Position ({taxi['start_x']}, {taxi['start_y']}), Status: {taxi['status']}")
        print("Passengers:")
        for passenger in self.passengers:
            print(f"  Passenger {passenger['id']}: Status: {passenger['status']}, Wait Time: {passenger['wait_time']}")

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
                distance = self.calculate_distance(taxi, passenger)
                if distance < min_distance:
                    min_distance = distance
                    nearest_passenger = passenger
        return nearest_passenger

    def calculate_distance(self, taxi, passenger):
        return np.sqrt((taxi['start_x'] - passenger['start_x']) ** 2 + (taxi['start_y'] - passenger['start_y']) ** 2)

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
        # 增加条件以控制乘客的生成
        if len(self.passengers) < 20:  # 限制场景中的最大乘客数
            new_passenger = {
                'id': len(self.passengers) + 1,
                'time': self.current_time,
                'max_wait': random.randint(5, 15),  # 乘客最长等待时间
                'start_x': random.randint(0, 100),
                'start_y': random.randint(0, 100),
                'target_x': random.randint(0, 100),
                'target_y': random.randint(0, 100),
                'status': 'waiting',
                'wait_time': 0
            }
            self.passengers.append(new_passenger)



env = TaxiDispatchEnv('./data_small.txt')

# 测试
# for step in range(100):
#     print(f"Step {step}:")
#     action = random.choice([0, 1, 2, 3, 4, 5])  # 随机选择一个动作
#     state, reward, done = env.step(action)
#     env.render()  # 可视化当前状态
#     print(f"Reward: {reward}")
#     if done:
#         break