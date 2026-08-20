import json
import math

# STEP 1: LOAD DATASET (SAFELY HANDLES NESTED DICTIONARIES)
def load_data(filepath="text_segmentation_dataset.json"):
    with open(filepath, 'r', encoding='utf-8') as f:

        data = json.load(f)
    
    test_cases = data.get("test_cases", [])
    word_counts = {}

    # Extract word frequencies
    for k, v in data.items():
        if k == "test_cases":
            continue
        if isinstance(v, dict):
            # Handles nested dictionary structures
            for w, c in v.items():
                if isinstance(c, int):
                    word_counts[w] = c
        elif isinstance(v, int):
            # Handles top-level word-count pairs
            word_counts[k] = v
    
    return word_counts, test_cases


# STEP 2: GREEDY LONGEST-MATCH SEGMENTATION
def greedy_segmentation(text, vocab):
    max_len = max((len(w) for w in vocab), default=1)
    i = 0
    n = len(text)
    words = []
    
    while i < n:
        matched = False
        # Try matching the longest possible substring starting at index i
        for j in range(min(i + max_len, n), i, -1):
            sub = text[i:j]
            if sub in vocab:
                words.append(sub)
                i = j
                matched = True
                break
        
        # If no word matches in dictionary, advance by 1 character
        if not matched:
            words.append(text[i])
            i += 1
            
    return " ".join(words)


# STEP 3: DYNAMIC PROGRAMMING SEGMENTATION (LOG-PROBABILITY)
def dp_segmentation(text, word_counts):
    total_words = sum(v for v in word_counts.values() if isinstance(v, int)) or 1
    vocab = set(word_counts.keys())
    max_len = max((len(w) for w in vocab), default=1)
    n = len(text)
    
    # Penalty log probability for unknown words
    unknown_word_penalty = math.log(1 / (total_words * 1000))
    
    def get_log_prob(word):
        if word in word_counts and isinstance(word_counts[word], int):
            return math.log(word_counts[word] / total_words)
        return unknown_word_penalty * len(word)

    # dp[i] stores (max_log_prob, backtrack_split_index) for prefix text[:i]
    dp = [(-float('inf'), -1)] * (n + 1)
    dp[0] = (0.0, 0)
    
    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            word = text[j:i]
            prob = get_log_prob(word)
            
            score = dp[j][0] + prob
            if score > dp[i][0]:
                dp[i] = (score, j)
                
    # Backtrack to reconstruct the best word sequence
    words = []
    curr = n
    while curr > 0:
        prev = dp[curr][1]
        words.append(text[prev:curr])
        curr = prev
        
    words.reverse()
    return " ".join(words)


# STEP 4: EVALUATION METRICS (EDIT DISTANCE & ACCURACY)
def edit_distance(str1, str2):
    """Calculates edit distance between two strings."""
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # Deletion
                                   dp[i][j-1],      # Insertion
                                   dp[i-1][j-1])    # Substitution
    return dp[m][n]


def evaluate(word_counts, test_cases):
    vocab = set(word_counts.keys())
    
    greedy_exact_matches = 0
    dp_exact_matches = 0
    
    total_greedy_edit_dist = 0
    total_dp_edit_dist = 0
    total_cases = len(test_cases)
    
    print(f"\n{'Input String':<35} | {'Ground Truth':<35} | {'Greedy Pred':<35} | {'DP Pred':<35}")
    print("-" * 145)
    
    for case in test_cases:
        inp = case["input"]
        gt = case["ground_truth"]
        
        pred_greedy = greedy_segmentation(inp, vocab)
        pred_dp = dp_segmentation(inp, word_counts)
        
        # Count exact matches
        if pred_greedy == gt:
            greedy_exact_matches += 1
        if pred_dp == gt:
            dp_exact_matches += 1
            
        # Calculate Edit Distance
        dist_greedy = edit_distance(pred_greedy, gt)
        dist_dp = edit_distance(pred_dp, gt)
        
        total_greedy_edit_dist += dist_greedy
        total_dp_edit_dist += dist_dp
        
        print(f"{inp[:33]:<35} | {gt[:33]:<35} | {pred_greedy[:33]:<35} | {pred_dp[:33]:<35}")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"Total Test Cases: {total_cases}")
    print("\n--- 1. GREEDY APPROACH ---")
    print(f"Accuracy (Exact Match): {(greedy_exact_matches / total_cases) * 100:.2f}%")
    print(f"Average Edit Distance: {total_greedy_edit_dist / total_cases:.2f}")
    
    print("\n--- 2. DYNAMIC PROGRAMMING APPROACH ---")
    print(f"Accuracy (Exact Match): {(dp_exact_matches / total_cases) * 100:.2f}%")
    print(f"Average Edit Distance: {total_dp_edit_dist / total_cases:.2f}")

if __name__ == "__main__":
    word_counts, test_cases = load_data()
    evaluate(word_counts, test_cases)