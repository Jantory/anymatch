We use eight widely recognized benchmark datasets from the Magellan repository, 
along with the WDC dataset, which is a recent addition from the e-commerce data. 
For detailed source information about these datasets, please visit the links provided below:

| Dataset        |                                                                                              Link | 
|:---------------|--------------------------------------------------------------------------------------------------:|
| wdc            |                                    [wdc](https://webdatacommons.org/largescaleproductcorpus/v2/ ) | 
| abt_buy        |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  ) | 
| amazon_google  |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  ) | 
| beer           |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  ) | 
| dblp_acm       |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  ) | 
| dblp_scholar   |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  ) | 
| fodors_zagat   |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  ) | 
| itunes_amazon  |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  ) | 
| walmart_amazon |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  ) | 

The training and validation set of the datasets originate from their respective sources,
while the test set follows the same construction rule from the 
"_Entity Matching using Large Language Models_", [paper link](https://arxiv.org/abs/2310.11244), [code link](https://github.com/wbsg-uni-mannheim/MatchGPT/tree/main/LLMForEM ).
paper. Additionally, we utilize the medium partition of the WDC dataset. To specify, 
a separate file named 'test.pkl.gz' is located in the 'abt\_buy', 'amazon\_google', 
'dblp\_acm', 'dblp\_scholar', 'walmart\_amazon', and 'wdc' folders. These files are 
specifically designed for evaluation purposes as described in the aforementioned paper. 
We will apply the same test sets for evaluation in our experiments. 
For the remaining three datasets, we will employ their original test sets for 
our evaluations.

