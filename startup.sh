echo ""
cat resources/title.txt
echo ""
echo ""

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PICTETROOT=$PWD
export FLASK_APP=flaskr
export FLASK_ENV=development
export PYTHONPATH=$PWD/engine:$PYTHONPATH
echo "Setup done!"

if [ "$1" == "test" ]; then
    python engine/runsearch.py Donald Trump
elif [ "$1" == "web" ]; then
    flask run
fi




