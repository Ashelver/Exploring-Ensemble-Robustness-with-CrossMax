import random
import torch
from collections import Counter
import torch.nn.functional as F
from itertools import combinations


# --------- Average Voting Ensemble Function ---------
def average_voting_ensemble(logits_list):
    """
    Average voting ensemble: averages logits from all models.
    
    Args:
        logits_list: list of [B, C] tensors
    
    Returns:
        Y: averaged logits of shape [B, C]
    """
    # logits_list: list of [B, C] tensors
    Z = torch.stack(logits_list, dim=1)  # [B, N, C]
    Y = Z.mean(dim=1)  # [B, C]
    return Y

# --------- Majority Voting Ensemble Function ---------
def majority_voting_ensemble(logits_list):
    """
    Majority voting ensemble: counts votes for each class from all models.
    
    Args:
        logits_list: list of [B, C] tensors
    
    Returns:
        majority_logits: vote counts for each class of shape [B, C]
    """
    # logits_list: list of [B, C] tensors
    Z = torch.stack(logits_list, dim=1)     # [B, N, C]
    preds = Z.argmax(dim=2)                 # [B, N]
    
    B, N = preds.shape
    C = Z.shape[2]
    majority_logits = torch.zeros(B, C, device=Z.device)

    # Count votes for each class
    for i in range(B):
        for cls in preds[i]:
            majority_logits[i, cls] += 1

    return majority_logits  # Each position represents the vote count for that class

# --------- CrossMax Ensemble Function ---------
def crossmax_ensemble(logits_list, k=2, N=6):
    """
    CrossMax ensemble: selects top-k values from randomly chosen N models.
    
    Args:
        logits_list: list of [B, C] tensors
        k: number of top values to consider
        N: number of models to randomly select
    
    Returns:
        Z_hat: normalized logits of selected models [B, N, C]
        Y: CrossMax output of shape [B, C]
    """
    # logits_list: list of [B, C] tensors
    total_models = len(logits_list)
    B, C = logits_list[0].shape

    # Randomly select N models
    if total_models > N:
        selected_indices = sorted(random.sample(range(total_models), N))
        selected_logits_list = [logits_list[i] for i in selected_indices]
    else:
        selected_logits_list = logits_list
        N = total_models  # Adjust actual number of selected models

    # Build Z: [B, N, C]
    Z = torch.stack(selected_logits_list, dim=1)  # [B, N, C]

    # Normalize logits
    Z_hat = Z - Z.max(dim=2, keepdim=True)[0]     # Subtract max value from each logit vector (per class)
    Z_hat = Z_hat - Z_hat.max(dim=1, keepdim=True)[0]  # Further normalization (per model)

    # CrossMax: take the k-th largest value from top-k as output
    Y = Z_hat.topk(k=k, dim=1).values[:, -1, :]  # [B, C]

    return Z_hat, Y


def test_ensemble(models, test_loader, SHOW_LOGITS=False, PRINT_LIMIT=5):
    import torch
    from collections import Counter

    total = 0
    printed = 0
    correct_counts = {name: 0 for name in models}
    for method in ['crossmax_exact', 'crossmax_ambiguous', 'avg_vote', 'maj_vote']:
        correct_counts[method] = 0

    crossmax_exact_details = {
        'correct': 0,
        'wrong': 0,
        'errors': [],  # (index, label, prediction, pred_set, votes)
        'ambiguous_correct': 0,
        'ambiguous_total': 0,
        'ambiguous_details': [],  # list of dicts with pred_set and votes
        'certain_correct': 0,
        'certain_total': 0,
    }

    ambiguous_info = {
        'ambiguous_total': 0,              # total number of ambiguous samples
        'ambiguous_correct': 0,            # number of ambiguous predictions containing the correct label
        'ambiguous_counts': [],            # size of each prediction set
        'ambiguous_samples': [],           # store (index, prediction_set)
        'certain_total': 0,                # number of samples with a unique maximum prediction
        'certain_correct': 0               # number of unique predictions that are correct
    }


    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs_dict = {name: model(images) for name, model in models.items()}
            logits_list = list(outputs_dict.values())

            for name, outputs in outputs_dict.items():
                predicted = outputs.argmax(dim=1)
                correct_counts[name] += (predicted == labels).sum().item()

            Z_hat, logits_crossmax = crossmax_ensemble(logits_list)
            max_vals, _ = logits_crossmax.max(dim=1, keepdim=True)
            is_max = (logits_crossmax == max_vals)

            preds_list = [logits.argmax(dim=1) for logits in logits_list]
            preds_stack = torch.stack(preds_list, dim=1)

            pred_crossmax_exact = []
            B = logits_crossmax.size(0)
            M = len(logits_list)
            k = 2

            for i in range(B):
                max_classes = torch.where(is_max[i])[0]  # prediction set
                pred_set = set(max_classes.tolist())
                target = labels[i].item()

                if len(pred_set) == 1:
                    chosen_class = max_classes[0].item()
                    crossmax_exact_details['certain_total'] += 1
                    if chosen_class == target:
                        crossmax_exact_details['certain_correct'] += 1
                else:
                    # ambiguous
                    votes = []
                    for current_N in [5, 4, 3]:
                        if current_N > M:
                            continue
                        _, Y = crossmax_ensemble(logits_list, k=k, N=current_N)
                        max_val = Y[i].max().item()
                        max_indices = (Y[i] == max_val).nonzero(as_tuple=True)[0]
                        for cls in max_indices.tolist():
                            if cls in pred_set:
                                votes.append(cls)

                    if votes:
                        count = Counter(votes)
                        chosen_class = count.most_common(1)[0][0]
                    else:
                        chosen_class = max_classes[0].item()

                    crossmax_exact_details['ambiguous_total'] += 1
                    if chosen_class == target:
                        crossmax_exact_details['ambiguous_correct'] += 1

                    # store details for each ambiguous sample (regardless of correctness)
                    crossmax_exact_details['ambiguous_details'].append({
                        'index': batch_idx * test_loader.batch_size + i,
                        'label': target,
                        'prediction': chosen_class,
                        'prediction_set': list(pred_set),
                        'votes': votes
                    })

                pred_crossmax_exact.append(chosen_class)

                if chosen_class == target:
                    crossmax_exact_details['correct'] += 1
                else:
                    crossmax_exact_details['wrong'] += 1
                    crossmax_exact_details['errors'].append({
                        'index': batch_idx * test_loader.batch_size + i,
                        'label': target,
                        'prediction': chosen_class,
                        'prediction_set': list(pred_set),
                        'votes': votes if len(pred_set) > 1 else []
                    })

            pred_crossmax_exact = torch.tensor(pred_crossmax_exact, device=labels.device)
            correct_counts['crossmax_exact'] += (pred_crossmax_exact == labels).sum().item()

            # CrossMax-Ambiguous
            for i in range(logits_crossmax.size(0)):
                max_classes = torch.where(is_max[i])[0].tolist()
                target = labels[i].item()

                if len(max_classes) > 1:
                    # ambiguous prediction set
                    ambiguous_info['ambiguous_total'] += 1
                    ambiguous_info['ambiguous_counts'].append(len(max_classes))
                    ambiguous_info['ambiguous_samples'].append((batch_idx * test_loader.batch_size + i, max_classes))

                    if target in max_classes:
                        ambiguous_info['ambiguous_correct'] += 1
                        correct_counts['crossmax_ambiguous'] += 1

                else:
                    # deterministic prediction
                    ambiguous_info['certain_total'] += 1
                    pred = max_classes[0]
                    if target == pred:
                        ambiguous_info['certain_correct'] += 1
                        correct_counts['crossmax_ambiguous'] += 1

            # Average Voting
            logits_avg = average_voting_ensemble(logits_list)
            pred_avg = logits_avg.argmax(dim=1)
            correct_counts['avg_vote'] += (pred_avg == labels).sum().item()

            # Majority Voting
            logits_maj = majority_voting_ensemble(logits_list)
            pred_maj = logits_maj.argmax(dim=1)
            correct_counts['maj_vote'] += (pred_maj == labels).sum().item()

            total += labels.size(0)

            if SHOW_LOGITS and printed < PRINT_LIMIT:
                for i in range(min(PRINT_LIMIT - printed, images.size(0))):
                    target = labels[i].item()
                    print(f"\n[Example {printed+1}] Target: {target}")
                    print("-" * 40)
                    print(f"{'Model':>18} | Logits")
                    print("-" * 18 + "+---------------------------------------------------------")
                    for name in models:
                        logits = outputs_dict[name][i].cpu().numpy()
                        logits_str = ", ".join([f"{x:6.3f}" for x in logits])
                        print(f"{name:>18} | [{logits_str}]")

                    Z_hat_np = Z_hat[i].cpu().numpy()
                    print(f"\n{'Z_hat (CrossMax One-Hot Inputs):':>18}")
                    print("[")
                    for row in Z_hat_np:
                        row_str = ", ".join([f"{v:7.3f}" for v in row])
                        print(f"  [{row_str}],")
                    print("]")

                    crossmax_logits_np = logits_crossmax[i].cpu().numpy()
                    logits_str = ", ".join([f"{x:6.3f}" for x in crossmax_logits_np])
                    print("-" * 40)
                    printed += 1

    print("\n======== Final Test Accuracies ========")
    for name, correct in correct_counts.items():
        acc = 100 * correct / total
        print(f"{name:>20}: {acc:.2f}%")

        if name == 'crossmax_ambiguous':
            total_ambiguous = ambiguous_info['ambiguous_total']
            correct_ambiguous = ambiguous_info['ambiguous_correct']
            total_certain = ambiguous_info['certain_total']
            correct_certain = ambiguous_info['certain_correct']

            acc_certain = 100 * correct_certain / total_certain if total_certain > 0 else 0.0
            acc_ambiguous = 100 * correct_ambiguous / total_ambiguous if total_ambiguous > 0 else 0.0

            print(f"{' '*8}(Unique predictions): {correct_certain}/{total_certain} = {acc_certain:.2f}%")
            print(f"{' '*8}(Prediction set contains label): {correct_ambiguous}/{total_ambiguous} = {acc_ambiguous:.2f}%")


    # CrossMax Exact Details
    print("\n======= CrossMax-Exact Prediction Breakdown =======")
    print(f"   Certain (set size == 1): {crossmax_exact_details['certain_correct']}/{crossmax_exact_details['certain_total']} = "
          f"{100 * crossmax_exact_details['certain_correct'] / max(1, crossmax_exact_details['certain_total']):.2f}%")
    print(f" Ambiguous (set size > 1): {crossmax_exact_details['ambiguous_correct']}/{crossmax_exact_details['ambiguous_total']} = "
          f"{100 * crossmax_exact_details['ambiguous_correct'] / max(1, crossmax_exact_details['ambiguous_total']):.2f}%")

    # print("\nExamples of ambiguous prediction sets and votes:")
    # for item in crossmax_exact_details['ambiguous_details'][:10]:
    #     print(f"[Index {item['index']}] True label: {item['label']}, Predicted: {item['prediction']}, "
    #           f"Prediction Set: {item['prediction_set']}, Votes: {item['votes']}")





def evaluate_diversity_confidence(models, test_loader):
    """
    Evaluate ensemble diversity and confidence metrics on test set:
    - CrossMax Confidence (fraction unique CrossMax ambiguous prediction)
    - Prediction Diversity (pairwise disagreement rate)
    - Softmax Confidence (average max softmax probability across models)
    - Prediction Entropy (entropy of averaged softmax probabilities)
    - Mutual Information (difference between total entropy and average individual entropy)
    - CrossMax ambiguous prediction set size statistics (min, max, avg)
    """

    M = len(models)
    total_samples = 0

    # Accumulators
    unique_prediction_count = 0  # CrossMax confidence numerator
    diversity_sum = 0.0
    softmax_conf_sum = 0.0
    entropy_sum = 0.0
    mi_sum = 0.0

    # For CrossMax ambiguous prediction set size statistics
    ambiguous_set_sizes = []

    model_names = list(models.keys())

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            batch_size = images.size(0)

            # Collect logits from all models
            logits_list = []
            preds_list = []
            softmax_list = []
            for name in model_names:
                logits = models[name](images)  # (B, num_classes)
                logits_list.append(logits)
                softmax_probs = F.softmax(logits, dim=1)
                softmax_list.append(softmax_probs)
                preds_list.append(logits.argmax(dim=1))  # predicted classes

            _, logits_crossmax = crossmax_ensemble(logits_list)

            max_vals, _ = logits_crossmax.max(dim=1, keepdim=True)
            is_max = (logits_crossmax == max_vals)

            for i in range(batch_size):
                max_classes = torch.where(is_max[i])[0].tolist()
                if len(max_classes) == 1:
                    unique_prediction_count += 1
                # Record ambiguous set size: can use prediction_sets[i] if available
                ambiguous_set_sizes.append(len(max_classes))

            # --- Prediction Diversity ---
            disagreement_sum = 0
            pair_count = 0
            for i, j in combinations(range(M), 2):
                disagreements = (preds_list[i] != preds_list[j]).float().sum().item()
                disagreement_sum += disagreements
                pair_count += batch_size
            diversity_sum += disagreement_sum / pair_count

            # --- Softmax Confidence ---
            max_softmax_per_model = [probs.max(dim=1)[0] for probs in softmax_list]  # list of (B,)
            avg_max_softmax = torch.stack(max_softmax_per_model, dim=1).mean(dim=1)  # (B,)
            softmax_conf_sum += avg_max_softmax.sum().item()

            # --- Prediction Entropy ---
            avg_softmax = torch.stack(softmax_list, dim=0).mean(dim=0)  # (B, num_classes)
            entropy = -(avg_softmax * avg_softmax.log()).sum(dim=1)  # (B,)
            entropy_sum += entropy.sum().item()

            # --- Mutual Information ---
            epsilon = 1e-12
            entropies_per_model = []
            for probs in softmax_list:
                probs_clamped = torch.clamp(probs, min=epsilon)
                entropy_m = -(probs_clamped * torch.log(probs_clamped)).sum(dim=1)
                entropies_per_model.append(entropy_m)
            avg_entropies = torch.stack(entropies_per_model, dim=0).mean(dim=0)
            avg_probs = torch.mean(torch.stack(softmax_list, dim=0), dim=0)
            avg_probs_clamped = torch.clamp(avg_probs, min=epsilon)
            entropy = -(avg_probs_clamped * torch.log(avg_probs_clamped)).sum(dim=1)
            mi = entropy - avg_entropies
            mi = torch.where(torch.isnan(mi), torch.zeros_like(mi), mi)
            mi_sum += mi.sum().item()

            total_samples += batch_size

    ambiguous_set_sizes = torch.tensor(ambiguous_set_sizes)
    ambiguous_stats = {
        'CrossMax_AmbiguousSet_Min': ambiguous_set_sizes.min().item() if ambiguous_set_sizes.numel() > 0 else 0,
        'CrossMax_AmbiguousSet_Max': ambiguous_set_sizes.max().item() if ambiguous_set_sizes.numel() > 0 else 0,
        'CrossMax_AmbiguousSet_Mean': ambiguous_set_sizes.float().mean().item() if ambiguous_set_sizes.numel() > 0 else 0,
    }

    metrics = {
        'CrossMax_Confidence': unique_prediction_count / total_samples if total_samples > 0 else 0.0,
        'Prediction_Diversity': diversity_sum / (total_samples if total_samples > 0 else 1),
        'Softmax_Confidence': softmax_conf_sum / total_samples if total_samples > 0 else 0.0,
        'Prediction_Entropy': entropy_sum / total_samples if total_samples > 0 else 0.0,
        'Mutual_Information': mi_sum / total_samples if total_samples > 0 else 0.0,
        **ambiguous_stats
    }

    print("=== Ensemble Diversity and Confidence Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics
