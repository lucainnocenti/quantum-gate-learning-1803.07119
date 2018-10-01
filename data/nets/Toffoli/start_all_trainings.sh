# use this script to start all trainings in the current folder
dirs=$(find . -name 'toff*' -not -empty -type d -maxdepth 1 -not -exec sh -c 'ls -1 {} | grep -q pickle' \; -print)

for f in $dirs; do screen -dm bash -c "cd \"$f\"; . activate theano; python start_training.py"; done
