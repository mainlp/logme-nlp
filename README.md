# LEEP
Implementation of LEEP: A New Measure to Evaluate Transferability of Learned Representations: http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf 

## Requirements

```
pip3 install --user numpy
```

## Example

The class expects a path to a `.txt` file consisting of the following:
```
# <unique gold labels target dataset>
# <unique gold labels source dataset>
<output probabilities of pretrained model of source set Z applied on target set Y> <gold label instance from Y>
```
For example, this is in your `output.txt` file:
```
# [A, B, C, D]
# [U, V, W, X, Y, Z]
[0.00342972157523036, 0.03722788393497467, 1.3426588907350379e-07, 0.8358138203620911, 0.007074566558003426, 0.11645391583442688] A
...
```
To run the script:
```
leep = LogExpectedEmpiricalPrediction("output.txt")
print(leep.compute_leep())
```