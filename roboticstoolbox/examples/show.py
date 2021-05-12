import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
from roboticstoolbox.backends.swift import Swift
import time

# Make and instance of the Swift simulator and open it
env = Swift()
env.launch()

# Make a panda model and set its joint angles to the ready joint configuration
panda = rtb.models.Panda()
panda.q = panda.qr

# Set a desired and effector pose an an offset from the current end-effector pose
Tep = panda.fkine(panda.q) * sm.SE3.Tx(0.2) * sm.SE3.Ty(0.2) * sm.SE3.Tz(0.45)

# Add the robot to the simulator
env.add(panda)

# Simulate the robot while it has not arrived at the goal
arrived = False
time.sleep(2)
while not arrived:

    # Work out the required end-effector velocity to go towards the goal
    v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, 1)
    
    # Set the Panda's joint velocities
    panda.qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v
    
    # Step the simulator by 50 milliseconds
    env.step(0.05)
    time.sleep(0.05)