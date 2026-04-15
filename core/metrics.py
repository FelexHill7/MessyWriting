"""
metrics.py — Evaluation metrics for handwriting recognition.

Pure functions with no model or data dependencies:
  - char_error_rate()  — CER via Levenshtein edit distance
  - align_chars()      — character-level alignment for confusion analysis
"""


def char_error_rate(pred, target):
    """Character Error Rate (CER) via Levenshtein edit distance.

    Returns: float in [0, 1] (or > 1 if pred is much longer than target).
    """
    n = len(target)
    m = len(pred)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if target[i - 1] == pred[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m] / max(n, 1)


def align_chars(pred, target):
    """Align predicted and target strings using Levenshtein DP.

    Returns a list of (true_char, pred_char) pairs including substitutions.
    Deletions use '\u2205' as the missing character marker.
    """
    n, m = len(target), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if target[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and (target[i - 1] == pred[j - 1] or
                                  dp[i][j] == dp[i - 1][j - 1] + 1):
            pairs.append((target[i - 1], pred[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((target[i - 1], "\u2205"))
            i -= 1
        else:
            pairs.append(("\u2205", pred[j - 1]))
            j -= 1
    return list(reversed(pairs))
