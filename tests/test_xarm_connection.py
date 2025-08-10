import zerorpc
import time

# 连接到服务器
client = zerorpc.Client()
client.connect("tcp://127.0.0.1:4242")

# 测试获取位置
pose = client.get_ee_pose()
print(f"Current pose: {pose}")

# 测试获取关节位置
joints = client.get_joint_positions()
print(f"Current joints: {joints}")

# 测试小幅度关节运动
print("Testing small joint movement...")
try:
    # 获取当前关节位置
    current_joints = client.get_joint_positions()
    
    # 创建一个小的运动：第一个关节移动0.1弧度（约5.7度）
    target_joints = current_joints.copy()
    target_joints[0] += 0.1  # 第一个关节向正方向移动0.1弧度
    
    print(f"Moving from: {current_joints}")
    print(f"Moving to: {target_joints}")
    
    # 执行运动
    client.move_to_joint_positions(target_joints, 3.0)  # 3秒内完成运动
    print("Movement command sent!")
    
    # 等待运动完成
    time.sleep(4)
    
    # 检查新位置
    new_joints = client.get_joint_positions()
    print(f"New joints: {new_joints}")
except Exception as e:
    print(f"Movement test failed: {e}")
    print("joint movement test failed, skipping end effector movement test.")

try:
    # 测试末端执行器位置
    print("Testing end effector movement...")
    current_pose = client.get_ee_pose()
    
    # 创建一个小的末端执行器运动：向上移动0.1米
    target_pose = current_pose.copy()
    target_pose[2] += 0.1  # Z轴向上移动0.1米
    
    print(f"Moving from: {current_pose}")
    print(f"Moving to: {target_pose}")
    
    # 执行末端执行器运动
    client.update_desired_ee_pose(target_pose)  
    print("End effector movement command sent!")
    
    # 等待运动完成
    time.sleep(4)
    
    # 检查新位置
    new_pose = client.get_ee_pose()
    print(f"New pose: {new_pose}")
    
except Exception as e:
    print(f"Movement test failed: {e}")
    print("End effector movement test failed, skipping final checks.")

print("Test completed!")