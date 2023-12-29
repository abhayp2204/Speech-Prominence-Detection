S = "hackerrank"
T = "fpelqanxyk"
k = 6
dp = {}

def cost(a, b):
    diff = abs(ord(a) - ord(b))
    return min(diff, 26 - diff)

from math import inf
def f(s, t, k):
    if k < 0: return -inf
    if s == len(S): return 0
    if t == len(T): return 0
    if (s, t, k) in dp: return dp[(s, t, k)]

    match = 1 + f(s + 1, t + 1, k - cost(S[s], T[t]))
    skip1 = f(s + 1, t, k)
    skip2 = f(s, t + 1, k)

    dp[(s, t, k)] = max(match, skip1, skip2)
    return dp[(s, t, k)]

ans = f(0, 0, k)
print("ans = ", ans)
