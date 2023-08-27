.PHONY: all python_requirements apt_upgrade apt_upgate upgrade_pip install_pip

all : python_requirements 
python_requirements:
	python3 -m pip install -r requirements.txt
apt_upgrade: apt_update
	apt-get upgrade -y
apt_update: 
	apt-get update -y
upgrade_pip: apt_upgrade
	pip install --upgrade pip