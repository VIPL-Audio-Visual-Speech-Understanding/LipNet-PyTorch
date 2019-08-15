# LipNet: End-to-End Sentence-level Lipreading

PyTorch implementation of the method described in the paper 'LipNet: End-to-End Sentence-level Lipreading' by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas (https://arxiv.org/abs/1611.01599).

## Results

|       Scenario          |  CER  |  WER  |
|:-----------------------:|:-----:|:-----:|
|    Unseen speakers      |  6.7% |  13.6% |
|   Overlapped speakers   |  2.0%  |  5.6%  |
|:-----------------------:|:-----:|:-----:|
|    Unseen speakers (Ours)      |  7.1% (+0.4%) |  13.6% (+0.0%) |
|   Overlapped speakers (Ours)   |  1.8% (-0.2%)  |  4.2% (-1.4%)  |

## Dataset

|       Scenario          |  Train  |  Validation  |
|:-----------------------:|:-----:|:-----:|
|    Unseen speakers      |  28775  |  3971  |
|   Overlapped speakers   |  24076  |  8670  |
|:-----------------------:|:-----:|:-----:|
|    Unseen speakers (Ours)     |  28837 |  3986 |
|   Overlapped speakers (Ours)  |  24459  |  8364  |

**Notes**:

- Contribution in sharing the results of this model is highly appreciated

## Dependencies

* PyTorch 1.0+
