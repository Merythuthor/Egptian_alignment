from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from typing import List, Dict, Any
from collections import defaultdict


class CustomMultiTaskCollator:


    def __init__(self, tokenizer, mlm_probability=0.15, mask_strategy="word"):
        self.tokenizer = tokenizer

        if mask_strategy == "word":
            print("💡 CustomMultiTaskCollator is using [DataCollatorForWholeWordMask] internally.")
            self.mlm_collator = DataCollatorForWholeWordMask(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=mlm_probability
            )
        else:
            print("💡 CustomMultiTaskCollator is using [DataCollatorForLanguageModeling] internally.")
            self.mlm_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=mlm_probability
            )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:


        pass_through_keys = {
            "lang", "text", "text_en", "translation",
            "align_pairs", "sent_flags", "contrastive_pairs",
            "translation_labels"
        }


        tensorizable_features = []
        pass_through_data = defaultdict(list)

        for feature in features:
            tensor_part = {}
            for key, value in feature.items():

                if key not in pass_through_keys:
                    tensor_part[key] = value
                else:
                    pass_through_data[key].append(value)
            tensorizable_features.append(tensor_part)


        batch = self.mlm_collator(tensorizable_features)


        batch.update(pass_through_data)

        return batch