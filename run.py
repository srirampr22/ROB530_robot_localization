from system.RobotSystem import *
from world.world2d import *


def main():
    print("Running the robot system")
    world = world2d()
    
    robot_system = RobotSystem(world)
    print("Running the filter")
    robot_system.run_filter()

if __name__ == '__main__':
    main()