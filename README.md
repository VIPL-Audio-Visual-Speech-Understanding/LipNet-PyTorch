# LipNet: End-to-End Sentence-level Lipreading

PyTorch implementation of the method described in the paper 'LipNet: End-to-End Sentence-level Lipreading' by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas (https://arxiv.org/abs/1611.01599). This program achieves the state-of-art results excepted the unseen speakers' character error rate.


## Results

|       Scenario          |  CER  |  WER  |
|:-----------------------:|:-----:|:-----:|
|    Unseen speakers      |  **6.7%** |  13.6% |
|   Overlapped speakers   |  2.0%  |  5.6%  |
|    Unseen speakers (Ours)      |  6.8%|  **13.5%** |
|   Overlapped speakers (Ours)   |  **1.7%**  |  **4.1%**  |

**Notes**:

- Contribution in sharing the results of this model is highly appreciated

## Data Statistics

|       Scenario          |  Train  |  Validation  |
|:-----------------------:|:-----:|:-----:|
|    Unseen speakers      |  28775  |  3971  |
|   Overlapped speakers  |  24331  |  8415  |
|    Unseen speakers (Ours)     |  28837 |  3986 |
|   Overlapped speakers (Ours)  |  24408  |  8415  |


## Preprocessing

Face detection and alignment is needed to generate training data. We provide examples in `extract_frame.py`, `face_det_sfd.py` and `extract_lip.py`. 

## Training And Testing

```
python main.py
```

Data path and hyperparameters are configured in `options.py`. Please pay attention that you may need to modify `options.py` to make the program work as expected.


## Dependencies

* PyTorch 1.0+
* opencv-python



