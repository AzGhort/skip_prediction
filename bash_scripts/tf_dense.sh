#!/bin/bash
#PBS -N tf_dense
#PBS -l select=1:ncpus=16:mem=16gb:scratch_local=20gb
#PBS -l walltime=4:00:00 
#PBS -m ae

set -e

trap 'clean_scratch' TERM EXIT

cd $SCRATCHDIR

ONEFILETRAIN=one_file_train_set
ONEFILETEST=one_file_test_set
MINITRAIN=training_set_mini
MINITEST=test_set_mini

TFDIR=$SCRATCHDIR/skip_prediction/tf
TRAINDIR=$SCRATCHDIR/$ONEFILETRAIN
TESTDIR=$SCRATCHDIR/$ONEFILETEST
DATADIR='/storage/praha1/home/ghort/'

NONEPREPR='NonePreprocessor'
MINMAXSCALER='MinMaxScaler'
STANDARDSCALER='StandardScaler'

cp -r $DATADIR/$ONEFILETRAIN $SCRATCHDIR
cp -r $DATADIR/$ONEFILETEST $SCRATCHDIR
cp -r $DATADIR/skip_prediction $SCRATCHDIR

export PYTHONPATH=$PYTHONPATH:$SCRATCHDIR/skip_prediction

mkdir -p $SCRATCHDIR/results

module add python-3.6.2-gcc
module add tensorflow-2.0.0-gpu-python3

cd $SCRATCHDIR

python $SCRATCHDIR/skip_prediction/models/tf_dense_nn_model.py --tf_folder $TFDIR --train_folder $TRAINDIR --test_folder $TESTDIR --tf_preprocessor $NONEPREPR > $SCRATCHDIR/results/results_nopr.txt
python $SCRATCHDIR/skip_prediction/models/tf_dense_nn_model.py --tf_folder $TFDIR --train_folder $TRAINDIR --test_folder $TESTDIR --tf_preprocessor $MINMAXSCALER >> $SCRATCHDIR/results/results_minmax.txt
python $SCRATCHDIR/skip_prediction/models/tf_dense_nn_model.py --tf_folder $TFDIR --train_folder $TRAINDIR --test_folder $TESTDIR --tf_preprocessor $STANDARDSCALER >> $SCRATCHDIR/results/results_std.txt

cp -r $SCRATCHDIR/results $DATADIR/tf_results || export CLEAN_SCRATCH=false
