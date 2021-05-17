# bpo

To run the IST experiment:
1. Clean up the data by running `ist_clean.py`
This assumes you have access to the original IST data found [here](https://datashare.is.ed.ac.uk/handle/10283/128). Place the IST_original.csv in your main directory. Set the `MAINDIR` variable in `ist_clean.py` to point to that file, then run:
```shell
python ist_clean.py --n_reps 1
```

1. Run the main experiment
Set the `MAINDIR` variable in `ist_experiment.py` to point to the right directory.
To run the model misspecification experiment (with sigmoid outcomes), run
```shell
python ist_experiment.py --reps "0,1" --sim "sig"
```
To run the heteroskedasticity experiment, run
```shell
python ist_experiment.py --reps "0,1" --sim "hsk"
```