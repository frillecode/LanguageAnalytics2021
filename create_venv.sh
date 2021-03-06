#!/usr/bin/env bash

VENVNAME=cds-lang

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate

pip --version
pip install --upgrade pip
pip --version

# problems when installing from requirements.txt
pip install ipython
pip install jupyter
pip install matplotlib

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"

