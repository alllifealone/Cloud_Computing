import torch
from torch.utils.data import Dataset

class ReviewDataset(Dataset):
    def __init__(self, ratings, labels, comment_counts, time_gaps, rating_entropy, rating_deviations, review_times, user_tenures, texts, tokenizer, max_length):
        self.ratings = ratings
        self.labels = labels
        self.comment_counts = comment_counts
        self.time_gaps = time_gaps
        self.rating_entropy = rating_entropy
        self.rating_deviations = rating_deviations
        self.review_times = review_times
        self.user_tenures = user_tenures
        self.texts = texts
        # self.content_length = content_lengths
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        rating = self.ratings[idx]
        comment_count = self.comment_counts[idx]
        time_gap = self.time_gaps[idx]
        rating_entropy = self.rating_entropy[idx]
        rating_deviation = self.rating_deviations[idx]
        review_time = self.review_times[idx]
        user_tenure = self.user_tenures[idx]

        text = self.texts[idx]

        # content_length = self.content_length[idx]
        label = self.labels[idx]
        behavior_feature = torch.tensor([rating, comment_count, time_gap, rating_entropy, rating_deviation, review_time, user_tenure])
        # behavior_feature = torch.tensor([rating])
        behavior_feature = behavior_feature.float()
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'behavior_feature': behavior_feature,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }