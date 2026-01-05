# Title
## Introduction
This is the code base for `Stories for the Win: Creating and Evaluating a Synthetic Dutch Storytelling Dataset for Pretraining`.    
The file structure is as follows:
1. `dataset_evaluation`: consists of all the code for the metrics
2. `datasets`: consists of the code to create a lexicon from a dataset. It also contains the cleaned code for CSTC.
3. `prompting`: some examples how to generate (pos experiment and baseline) and prompting for different ages. 
4. `vendi-score`: a forked version of the official vendi-score package. There were some parts incongruent with newer python versions. 
5. `paper_usage_metrics.ipynb`: example usage of the metrics for two english datasets
6. `paper_model_choices.ipynb`: code to try and evaluate different models
7. `paper_prompt_additions.ipynb`: code to try and evaluate different evaluation metrics
8. `paper_new_dataset_statistics`: code to sample from new dataset

## How to reproduce
1. Use `paper_model_choices.ipynb` to use a model.
2. Use the code in `prompting` to generate different subsets to compare.
3. Use the code in `datasets` to create the reference corpora needed for the metrics grammaticality and creative perplexity.
4. Use the code in `paper_prompt_additions.ipynb` to run the metrics on different prompt elements.