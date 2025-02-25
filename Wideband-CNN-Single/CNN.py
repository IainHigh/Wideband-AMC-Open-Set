# CNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulationClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # Basic CNN
        self.conv1 = nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(64, 64)
        # final layer => produce:
        #   - policy_logits (2) => "split" vs "stop"
        #   - freq => 1 float
        #   - mod_logits => distribution over num_classes
        out_dim = 2 + 1 + num_classes  # 2=policy logits, 1=freq, rest=mod logits
        self.fc2 = nn.Linear(64, out_dim)

    def forward(self, x):
        # x shape => [batch_size, 2, L]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x).squeeze(-1)  # [batch_size, 64]
        x = F.relu(self.fc1(x))
        out = self.fc2(x)  # shape => [batch_size, 2 + 1 + num_classes]
        return out

    def stochastic_recursive_inference(
        self, x, freq_start, freq_end, log_probs_accum, depth=0
    ):
        """
        A policy gradient recursion that can produce multiple signals.
        log_probs_accum: a list we append the log probability for each step
        returns: a list of (freq_pred, mod_pred)
        """
        out = self.forward(x)  # shape => [1, 2 + 1 + num_classes]
        # parse
        policy_logits = out[:, :2]  # shape => [1,2]
        freq_pred_raw = out[:, 2]  # shape => [1]
        mod_logits = out[:, 3:]  # shape => [1, num_classes]

        # We interpret policy_logits as "split vs stop"
        policy_dist = torch.distributions.Categorical(logits=policy_logits)
        action = policy_dist.sample()  # 0 => split, 1 => stop
        log_p = policy_dist.log_prob(action)
        log_probs_accum.append(log_p)

        # If maximum depth reached, we force a prediction
        if depth == 2:
            action = torch.tensor(1, device=x.device)

        if action.item() == 0:
            # "split => multiple signals" => we subdivide
            mid = 0.5 * (freq_start + freq_end)
            # do left
            left_preds = self.stochastic_recursive_inference(
                x, freq_start, mid, log_probs_accum, depth + 1
            )
            # do right
            right_preds = self.stochastic_recursive_inference(
                x, mid, freq_end, log_probs_accum, depth + 1
            )
            return left_preds + right_preds
        else:
            # "stop => produce final freq + mod"
            # We interpret freq_pred_raw as a fraction of the sub-band => map to [freq_start, freq_end]
            fraction = torch.sigmoid(freq_pred_raw)  # in [0,1]
            center_freq = freq_start + (freq_end - freq_start) * fraction.item()

            mod_dist = F.softmax(mod_logits, dim=1)
            mod_action = torch.distributions.Categorical(mod_dist).sample()
            # we do not store log prob for mod_action here if we treat mod as part of the final cost approach
            # or do we do a separate REINFORCE for mod? For demonstration, let's keep it simple and rely on the final cost.

            return [(center_freq, mod_action.item())]
