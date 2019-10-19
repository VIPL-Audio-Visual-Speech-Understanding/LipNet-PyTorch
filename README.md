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

Link of processed lip images and text: https://pan.baidu.com/s/165swxO8A-GUZ03-InTL6VA
Google Drive: https://drive.google.com/drive/folders/1Wn2EJw2101nF59eNDXEto6qXqfgDDucL?usp=sharing

Download all parts and concatenate the files using the command 

```
cat GRID_LIP_160x80_TXT.zip.* > GRID_LIP_160x80_TXT.zip
unzip GRID_LIP_160x80_TXT.zip
rm GRID_LIP_160x80_TXT.zip
```

We provide examples of face detection and alignment in `scripts/` folder for your own dataset.

## Training And Testing

```
python main.py
```

Data path and hyperparameters are configured in `options.py`. Please pay attention that you may need to modify `options.py` to make the program work as expected.


## Dependencies

* PyTorch 1.0+
* opencv-python



