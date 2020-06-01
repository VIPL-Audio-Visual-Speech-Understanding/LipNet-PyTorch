# LipNet: End-to-End Sentence-level Lipreading

The state-of-art PyTorch implementation of 'LipNet: End-to-End Sentence-level Lipreading' by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas (https://arxiv.org/abs/1611.01599). This version achieves **the best performance in all evaluation metrics**.

![LipNet Demo](demo.gif)


## Results

|       Scenario          |  Image Size (W x H) |  CER  |     WER    |
|:-----------------------:|:-------------------:|:-----:|:----------:|
|    Unseen speakers (Origin)    | 100 x 50 |    6.7%   |    13.6%   |
|   Overlapped speakers (Origin) | 100 x 50 |    2.0%   |    5.6%    |
|    Unseen speakers (Ours)      | 128 x 64 |  **6.7%** |  **13.3%** |
|   Overlapped speakers (Ours)   | 128 x 64 |  **1.9%** |  **4.6%**  |

**Notes**:

- Contribution in sharing the results of this model is highly appreciated

## Data Statistics

|             Scenario            |   Train   |  Validation  |
|:-------------------------------:|:---------:|:------------:|
|    Unseen speakers (Origin)     |   28775   |     3971     |      
|   Overlapped speakers (Origin)  |   24331   |     8415     |
|    Unseen speakers (Ours)       |   28837   |     3986     |
|   Overlapped speakers (Ours)    |   24408   |     8415     |


## Preprocessing

Link of processed lip images and text: 

BaiduYun: 链接:https://pan.baidu.com/s/1I51Xf-DzP1UgrXF-S0L5tg  密码:jf0l

Google Drive: https://drive.google.com/drive/folders/1Wn2EJw2101nF59eNDXEto6qXqfgDDucL

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

## Bibtex
    @article{assael2016lipnet,
	  title={LipNet: End-to-End Sentence-level Lipreading},
	  author={Assael, Yannis M and Shillingford, Brendan and Whiteson, Shimon and de Freitas, Nando},
	  journal={GPU Technology Conference},
	  year={2017},
	  url={https://github.com/Fengdalu/LipNet-PyTorch}
	}


## License

The MIT License


