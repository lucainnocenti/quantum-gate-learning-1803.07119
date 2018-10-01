dirs=$(ls -d *initvalues*)
for f in $dirs; do screen -dm bash -c "cd \"$f\"; . activate theano; python start_training.py"; done
