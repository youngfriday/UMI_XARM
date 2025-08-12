# Deploy UMI Policy on xArm Robot

> This is a repository for deploying UMI policy on [xArm6/7](https://www.cn.ufactory.cc/xarm) robot arm and gripper.

https://github.com/user-attachments/assets/65599d36-7b7f-42c9-9a5f-107c39351694

## 🛠️ Installation

See the [UMI repository](https://github.com/real-stanford/universal_manipulation_interface) for installation. 

## 🦾 Real-World Evaluation
- For the hardware setup, please refer to the [Fast-UMI](https://fastumi.com/), which opened source the adapted UMI gripper for xArm. 

> Fast-UMI gripper's camera position is a little different from original UMI gripper, so you may feel some action bias when using the ckpt from [Data-Scaling-Laws](https://github.com/Fanqi-Lin/Data-Scaling-Laws) (also you can see the action bias in my demo)

- For real-world evaluation, please refer [xarm_instruction](xarm_instruction.md)

> ATTENTION: MAKE SURE YOUR HAND ON THE EMERGENCY STOP SWITCH!

## 🙏 Acknowledgement
- Thank the authors of [UMI](https://github.com/real-stanford/universal_manipulation_interface) and [Data-Scaling-Laws](https://github.com/Fanqi-Lin/Data-Scaling-Laws) for sharing their codebases and checkpoints.
- Thank the authors of [Fast-UMI](https://fastumi.com/) for sharing their adapted gripper model.

## 👀 FUTURE WORK

- Detailed UMI reproduction record blog
- Data Collection & Training details 
- Redesigned UMI camera mount for xArm
