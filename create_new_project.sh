#!/bin/bash

SECONDS=0
if [ $# -eq 0 ]
  then
    echo "Please enter a project name..."
    duration=$SECONDS
    echo "Created the project in $(($duration / 60)) minutes and $(($duration % 60)) seconds."
    exit 1
fi

git clone https://github.com/asvskartheek/Training-Framework.git
mv Training-Framework $1
cd $1
python -m venv venv
source ./venv/bin/activate
echo "Installing Bare Necessities..."
pip install -q --upgrade pip
pip install -q jupyter
pip install -q jupyterlab

echo "Installing Regularly used libraries..."
pip install -q numpy
pip install -q pandas
pip install -q torch
pip install -q pytorch_lightning

# Install mentioned packages
for i in "${@:2}"
do echo "Installing $i..."
   pip install -q "$i"
done
pip list > requirements.txt

# Remove README and shell script
rm -y README.md
rm -y create_new_project.sh

# Re-init git to remove previous history
git init
git add .
git commit -am "Setup Project"

duration=$SECONDS
echo "Created the project in $(($duration / 60)) minutes and $(($duration % 60)) seconds."

# Uses Visual Studio Code by default
# WARNING: MAKE SURE THAT VS Code is installed and added to PATH.
code .