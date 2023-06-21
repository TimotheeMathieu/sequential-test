# sequential-test
Sequential and group sequential hypothesis testing in python.


## Installation 

```
pip install git+https://github.com/TimotheeMathieu/sequential-test
```

## Usage

See the example script in `examples` directory.

## Limitation

For now, the boundaries of the group sequential tests are precompted using R library `ldbounds` and its power values are pre-computed with Monte-Carlo estimation. See the `scripts` directory for the scripts that generated `sequential_test/boundaries.csv` and `sequential_test/power_dataframe.csv`.
