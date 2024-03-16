import random

#大城市等比缩小100倍
num_people=5000 #总需求人次，一天中产生的需求人次 #真实50万
num_taxi=100 #一天中投入的总车辆数  #真实1万
city_length=10 #城市边长100km，城市为正方形 #真实100km

num_minutes=18*60  #需要调度18*60=1080分钟，6：00-24：00

taxi_speed=0.7  #出租车平均车速 0.7/minute

with open('data_small.txt', 'w',encoding='utf-8') as f:
    for minute in range(num_minutes):
        people=random.randint(3,8)  #平均每分钟5个人
        line='0,'+str(minute)+','
        for p in range(people):
            x1=random.uniform(0,10*1000)
            y1=random.uniform(0,10*1000)
            x2=random.uniform(0,10*1000)
            y2 = random.uniform(0, 10 * 1000)
            wait_time=random.randint(5,15)
            line+=' '.join([str(p),str(wait_time),str(x1),str(y1),str(x2),str(y2)])
            line+=','
        f.write(line+'\n')
    for minute in range(num_minutes):
        taxi=random.randint(1,2)
        line='1,'+str(minute)+','
        for t in range(taxi):
            x1 = random.uniform(0, 10 * 1000)
            y1 = random.uniform(0, 10 * 1000)
            line+=' '.join([str(t),str(x1),str(y1)])
            line+=','
        f.write(line+'\n')


