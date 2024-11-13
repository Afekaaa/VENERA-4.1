import krpc
import time


conn = krpc.connect()
vessel = conn.space_center.active_vessel

obt_frame = vessel.orbit.body.reference_frame



# for i in range(-3, 7):
#     print(f'Stage {i}: {vessel.resources_in_decouple_stage(stage=i, cumulative=False).names}')

#старт
vessel.auto_pilot.target_pitch_and_heading(90, 90)
vessel.auto_pilot.engage()
vessel.control.throttle = 1
vessel.control.activate_next_stage()
time.sleep(4)
print('Launch')
vessel.control.activate_next_stage()

flag = True

fuel = vessel.resources_in_decouple_stage(stage=5, cumulative=True).amount('Kerosene')
while True:
    print('Высота:', vessel.flight().mean_altitude) # скорость, высота, две координаты
    print('Скорость:', vessel.flight(obt_frame).speed)
    print('Координаты:', ', '.join(map(str, vessel.position(vessel.orbit.body.reference_frame))))
    print('Топливо', fuel)
    print("=============================")
    time.sleep(2)

    if flag:
        k = fuel
        fuel = vessel.resources_in_decouple_stage(stage=5, cumulative=True).amount('Kerosene')

    if flag and k == fuel:
        vessel.control.activate_next_stage()
        flag = False