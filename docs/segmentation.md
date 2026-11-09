# Segmentation

Segmentation is for the comparator methods only: hybrid retrieval and Legal-BERT.

Passages are about 120 words, with the overlap set in `configs/config.yaml` (`segmentation.overlap_words`, currently 20). The last passage keeps whatever tail is left, so no words are dropped.

The whole-policy LLM judge reads the full cleaned policy. These passages must not gate, filter, or skip that judgment.
