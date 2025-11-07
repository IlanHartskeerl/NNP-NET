# NNP-NET

This repository is the original implementation of NNP-NET used in the thesis. It also contains and implementation of tsNET(*) and PMDS. The tsNET implementation is based on the bhtsne implementation that can be found [here](https://github.com/lvdmaaten/bhtsne). The PMDS implementation was taken from [OGDF](https://github.com/ogdf/ogdf) and modified in order to support more than 3 output dimensions.

## Build

This project can be compiled using cmake using a c++ compiler as follows:

```mkdir build
cd build
cmake ../
make
```

This project has been validated to compile using both g++ and visual studios msvc compiler. It is also required to have python and tensorflow installed. If layouts for time-series data are made, then python library skippy is also required.

## Usage

This project can be used as follows:

`./NNPNET TestGraphs/3elt.mtx`

This will create a graph layout using NNP-NET of the graph at the relative path *TestGraphs/3elt.mtx* using all default settings, saving the output to *TestGraphs/3elt_out.vna*.

`./NNPNET TestGraphs -m tsNET* -o out`

Will create graph layouts using tsNET* for all graphs in folder *TestGraphs* and save the output to the *out* folder.

Supported input file types are .vna, .mtx and .dot. Output will always be done to a .vna file.

## Arguments

| Argument | Short | Description | Default value |
| ----------- | ----------- | ----------- | ----------- |
| --method | -m | Changes which graph layout method is used. Options are: NNP-NET, tsNET, tsNET* and PMDS | NNP-NET |
| --output | -o | Output path | *original_filename* + "_out" |
| --dimensions | -d | Number of output dimensions | 2 |
| --smoothing | -l | Number of laplacian smooothing passes used by NNP-NET | 3 |
| --theta | -t | Theta value used by approximate tsNET(*). Use 0 for the exact version | 0.25 |
| --gpu | -g | Whether tensorflow is allowed to use the GPU | False |
| --perplexity | -p | Perplexity used by tsNET(*) | 40 |
| --embedding_size | -e | Number of embedding dimensions used by NNP-NET | 50 |
| --subgraph_size | -s | Target size for the subgraph used for NNP-NET | 10000 |
| --pmds_pivots |  | Number of pivot points used by PMDS | 250 |
| --batch_size | -b | Batch size used for training NNP as part of NNP-NET | 64 |
| --training_epochs |  | Training epochs used for training NNP as part of NNP-NET | 40 |
| --use_float | -f | Uses float instead of doubles for PMDS. Only effects PMDS calls. Doubles might be needed for very large graphs | True |
| --time_series |  | Loads time series data and uses stable PMDS and NNP-NET for the layout. Does not work for tsNET(*) | False |

## Time series data

Layouts for time-series data can be created by setting the `--time_series` flag to true. This will only have an effect when using either PMDS or NNP-NET. The first time-step has to be a standard graph file in any supported format. All subsequent time-steps only describe the change from one time-step to the next. An example of the expected format can be found in folder TestTimeseries. In order to run this example, you would use the following command:

`./NNPNET TestTimeseries/3elt.mtx --time_series 1`
