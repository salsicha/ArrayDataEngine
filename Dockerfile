FROM ubuntu:noble AS build

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    TERM=xterm \
    PYTHONIOENCODING=UTF-8 \
    ROS2_DISTRO=jazzy \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/venv/bin:$PATH" \
    ROS_DISTRO=jazzy

# This prevents ROS setup.bash from failing
SHELL ["/bin/bash","-c"]

# ROS2
RUN apt update && apt install -q -y --no-install-recommends \
    curl gnupg2 lsb-release python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*
RUN curl --insecure -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 packages
RUN apt update && apt install -q -y --no-install-recommends \
    python3-serial imagemagick libboost-all-dev libgts-dev libjansson-dev ros-$ROS2_DISTRO-rviz2 \
    ros-$ROS2_DISTRO-foxglove-bridge ros-$ROS2_DISTRO-foxglove-compressed-video-transport ros-$ROS2_DISTRO-foxglove-msgs \
    ros-$ROS2_DISTRO-ros-core ros-$ROS2_DISTRO-sensor-msgs-py libeigen3-dev \
    ros-$ROS2_DISTRO-ros2bag ros-$ROS2_DISTRO-rclpy ros-$ROS2_DISTRO-rosbag2-storage-default-plugins \
    python3-rosdep python3-colcon-ros linuxptp python3-colcon-common-extensions \
    ros-$ROS2_DISTRO-rosbridge-suite ros-$ROS2_DISTRO-rosbag2 ros-$ROS2_DISTRO-gps-msgs \
    ros-$ROS2_DISTRO-tf2-msgs software-properties-common build-essential gcc \
    # pyROS packages
    python3-image-geometry x11-apps \
    # psycopg2 dependencies
    libpq-dev build-essential \
    python3-tk mlocate \
    ros-$ROS2_DISTRO-cv-bridge \
    libgl1 libgomp1 libegl1 \
    xorg-dev libxcb-shm0 libglu1-mesa-dev python3-dev clang \
    libc++-dev libc++abi-dev libsdl2-dev ninja-build libxi-dev \
    libtbb-dev libosmesa6-dev libudev-dev autoconf libtool && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN python3 -m venv /venv
# EM is preventing msgs from building...
RUN pip3 uninstall em
RUN pip3 install rosdep colcon-common-extensions pytest-rerunfailures numpy lark empy==3.3.4

RUN mkdir /dataengine

COPY requirements.txt /dataengine/requirements.txt
RUN pip3 install -r /dataengine/requirements.txt

# Clean up
RUN apt remove -y --auto-remove build-essential gcc && \
    rm -rf /root/.cache

COPY engine /dataengine/engine
COPY setup.py /dataengine/setup.py

# TODO: build package


#####################################################################
FROM scratch

COPY --from=build / /

WORKDIR /dataengine/

ENTRYPOINT ["/entrypoint.sh"]
