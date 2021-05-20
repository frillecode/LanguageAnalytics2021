#!/usr/bin/env bash

VENVNAME=cds-lang

python -m venv $VENVNAME
source $VENVNAME/Scripts/activate
pip install --upgrade pip

# problems when installing from requirements.txt
pip install ipython
pip install jupyter
pip install matplotlib

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"