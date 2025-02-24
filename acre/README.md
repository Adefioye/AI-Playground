# Datasets were created in this manner

- Create `Comp`, `IID` and `Sys` datasets with train, test and val using:
```
./process_acre_all.sh text
```

- Create `combined` folder using by aggregating all train, test and val datasets from `Comp`, `IID` and `Sys` into corresponding train, test and val datasets using:
```
python combine_all_data.py 
```
