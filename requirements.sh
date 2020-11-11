#!/bin/sh

echo " "
echo "requirements.sh prompts you to install the modules required for this program!"
echo " "
sudo apt install python3-pip -y 
pip3 install numpy
pip3 install pandas
pip3 install sklearn
pip3 install datetime
pip3 install matplotlib

echo " "
echo "Now, you are all set to execute this program!"
echo " "