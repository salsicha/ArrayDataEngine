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
    python3-rosdep python3-colcon-ros python3-colcon-common-extensions \
    ros-$ROS2_DISTRO-rosbridge-suite ros-$ROS2_DISTRO-rosbag2 ros-$ROS2_DISTRO-gps-msgs \
    ros-$ROS2_DISTRO-tf2-msgs software-properties-common build-essential gcc \
    x11-apps libpq-dev build-essential python3-tk python3-pandas unzip python3-opencv \
    ros-$ROS2_DISTRO-cv-bridge python3-numpy vim python3-sklearn python3-skimage \
    python3-scipy python3-tqdm wget python3-dev ninja-build clang && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN python3 -m venv /venv

RUN curl https://bootstrap.pypa.io/get-pip.py | python
RUN python -m ensurepip --upgrade
RUN python -m pip install --upgrade setuptools

# EM is preventing msgs from building...
RUN pip3 uninstall em

RUN mkdir /dataengine

COPY requirements.txt /dataengine/requirements.txt
RUN pip3 install -r /dataengine/requirements.txt

## ROS2_numpy
RUN . /opt/ros/$ROS2_DISTRO/setup.bash && \
    wget https://github.com/Box-Robotics/ros2_numpy/archive/refs/tags/v2.0.12-jazzy.zip && \
    unzip v2.0.12-jazzy.zip && rm v2.0.12-jazzy.zip && \
    cd ros2_numpy-2.0.12-jazzy && mkdir build && cd build && cmake .. && make install

WORKDIR /dataengine

COPY engine /dataengine/engine
COPY setup.py /dataengine/setup.py
RUN python3 setup.py develop

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh


#####################################################################
FROM scratch

COPY --from=build / /

WORKDIR /notebooks/

ENTRYPOINT ["/entrypoint.sh"]
CMD ["jupyter-lab", "--ip", "0.0.0.0", "--no-browser", \
    "--allow-root", "--ServerApp.token=docker_jupyter", \
    "--NotebookApp.allow_password_change=False"]
