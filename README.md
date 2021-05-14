# SaRoCo

## License Agreement

**Copyright (C) 2021 - Ana-Cristina Rogoz, Mihaela Găman, Radu Tudor Ionescu**

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

You are free to **share** (copy and redistribute the material in any medium or format) and **adapt** (remix, transform, and build upon the material) under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial** — You may not use the material for commercial purposes.
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
- **No additional restrictions** — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

## Citation

Please cite the corresponding work (see citation.bib file to obtain the citation in [BibTex](citation.bib) format) if you use this data set and software (or a modified version of it) in any scientific work:

**[1] Ana-Cristina Rogoz, Mihaela Gaman, Radu Tudor Ionescu. SaRoCo: Detecting Satire in a Novel Romanian Corpus of News Articles. In Proceedings of ACL, pp. TBD, 2021. [(link to paper)](https://arxiv.org/abs/2105.06456).**

## Dataset

The dataset contains 55,608 news articles from multiple real and satirical news sources, of which 27,980 are regular and 27,628 satirical news reports. We provide the data in csv format, in three files following the train/validation/test splits used in our experiments that are detailed in the paper:
- data/train.csv
- data/validation.csv
- data/test.csv
 
The data format is as follows:

```
index, title, content, label
```
  
Each sample in the dataset contains an index, the title, content and the associated automatically assigned label, which can be 0 for legitimate news and 1 for the satirical ones.

The three subsets described in the paper feature the following sample counts:
- training: 18,000 regular articles, 17,949 satirical news reports
- validation: 4,986 regular articles, 4,878 satirical news reports
- test: 4,994 regular, 4,801 satirical samples

## Baselines

### Fine-tuned Romanian BERT

### Character-Level Convolutional Neural Network

