#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash 
source /venv/bin/activate
source /usr/local/share/ros2_numpy/local_setup.bash
exec "$@"

