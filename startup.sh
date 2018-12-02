#!/bin/bash
echo ""
cat resources/title.txt
echo ""
echo ""

export PYTHONWARNINGS="ignore"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PICTETROOT=$PWD
export FLASK_APP=flaskr
#export FLASK_ENV=development
export PYTHONPATH=$PWD/engine:$PYTHONPATH
echo "Setup done!"

if [ "$1" == "test" ]; then
    python engine/runsearch.py Donald Trump
elif [ "$1" == "web" ]; then
    flask run --host=0.0.0.0 --port=80
elif [ "$1" == "train" ]; then
    python engine/wikiutils.py
    python engine/twitter.py
    python engine/googleutils.py
    python engine/checkPolitician.py
    python engine/checkFamous.py
fi

