# Generating plots
Run the python script `plot.py` with the following arguments to generate plots for the normal vector add implementation and the streamed version:
```bash
python3 plot.py vector_add
python3 plot.py stream_add
```

This will create two files `vector_add.png` and `stream_add.png`.

Running the python script `compare_segment_size.py` will produce another plot comparing the execution time on a vector with different segment size per stream. The values are hardcode and if you want to produce a new plot with new values you need to manually recompile `stream_add.cu` with different values for:
```c++
constexpr uint STREAMS = 4;
```
for a given vector size and then paste the obtained values into `compare_segment_size.py`