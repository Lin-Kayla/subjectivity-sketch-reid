<h2 align="center">Beyond Domain Gap: Exploiting Subjectivity in Sketch-Based Person Retrieval
</h2>
<p align="center">Kejun Lin, Zhixiang Wang, Zheng Wang, Yinqiang Zheng, Shin'ichi Satoh
</p>

<div align="center">
<img src="./assets/teaser.jpg" width=300px alt="teaser"></image>
</div>

## Dataset

Our proposed MaSk1K (Short for <u>Ma</u>rket-<u>Sk</u>etch-<u>1K</u>) is available <a href="https://drive.google.com/drive/folders/1XjFPM1yVHpE38sSDTFgM5s9aX2r-oYRC?usp=sharing">here</a>.

TODO

Our annotated attributes for PKU-Sketch is available <a href="">here</a>.

Download the dataset and PKU-Sketch attributes into your \<data_root\>.

Download the dataset and Market1501 attributes from <a href="">here</a>, and put it into your <data_root>.

## Guide For Market-Sketch-1K
### requirements
download the necessary dependencies using cmd.
```bash
pip install -r requirements.txt
```

### preprocess
```python
python preprocess.py --data_path=<data_path> --attribute_path=<attribute_path> --train_style <train_style> [--train_mq]
```

 - `<data_path>` should be replaced with the path to your data.
 - `<attribute_path>` should point to the attribute data.
 - `<train_style>` refers to the styles you want to include in your training set. You can use any combination of styles A-F, such as B, AC, CEF, and so on.
-  `[--train_mq]` argument is optional and can be used to enable multi-query during training.

### start training
```
python train.py --train_style <train_style> --test_style <test_style> [--train_mq] [--test_mq]
```

 - `<train_style>` and `<test_style>` should be replaced with the styles you want to use for your training and testing sets, respectively. Just like in the preprocessing step, you can use any combination of styles A-F.
 - `[--train_mq]` argument is used for enabling multi-query during training, and `[--test_mq]` serves a similar purpose during testing.

### Evaluation
TODO
<!-- ```python -->
<!--  python test.py [] -->
<!-- ``` -->

## Guide For PKU-Sketch
TODO

## Acknowledgements
TODO
Our code was build on the amazing codebase <a href="">Cross-modal-Re-ID</a> and <a href="">CMAlign</a> and <a href="">CLIP</a>. 

## Citation
TODO
If you find our work helpful, please consider citing our work using the following bibtex.
```
@article{
    author=...
}
```