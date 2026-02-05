Step by step to run udp raw streaming

1. Connect all devices to tp-link at 2.4GHz (5GHz if pi is vers. 4)
2. ssh to raspberry pi
    `ssh nhmrc@192.168.0.236`
3. Navigate to nhmrc-mvp (~/Projects/nhmrc-mvp)
4. Run rpi_main.py in the rpi conda environment
    ```
    conda activate rpi
    python rpi_main.py
    ```
5. Open a new terminal and run ROS master
    `roscore`
6. Open a new terminal and navigate to ~/Documents/nhmrc-mvp
7. Start reading the IMU and RGB-D data from WiFi
    `python udp_to_ros.py`
8. In a new terminal, check if the inputs are coming in
    `python ros_publish_check.py`

To run the basic YOLO sonification pipeline, close any other running Ubuntu
side terminals and open a new one.
9. Navigate to ~/Documents/nhmrc-mvp
10. Run the pc_main.py script
    `python pc_main.py`

There are also wired connection options for faster and more streaming
streaming. If running pc_main.py on the ARIA PC, use the nhmrc conda
environment, `conda activate nhmrc`.
