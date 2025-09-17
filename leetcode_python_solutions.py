# -*- coding: utf-8 -*-
"""
LeetCode Python 解法汇总（由 Excel 导出）
来源文件：leetcode 刷题记录表.xlsx / sheet: python
生成方式：自动整理（题号、复杂度、笔记保留为注释）
"""

################################################################################
# 专题：相向双指针 
https://www.bilibili.com/video/BV1bP411c7oJ/?spm_id_from=333.1387.collection.video_card.click&vd_source=d996d505ca380f869ce8821dbe18355d
################################################################################

########################################
# 题号：167. Two Sum II - Input Array Is Sorted
# 复杂度：时间 O(N) | 空间 O(1)
# 笔记：
#   1. 两数之和 有序版本，相向双指针方法，可以削减搜索空间
#
########################################
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers) - 1
        while True: # left < right
            s = numbers[left] + numbers[right]
            if target == s:
                break

            if s > target:
                right -= 1

            if s < target:
                left += 1
        return [left+1, right+1]

########################################
# 题号：15. 3Sum
# 复杂度：时间 O(N^2) | 空间 O(1)
# 笔记：
#   2. 三数之和，采用 枚举 + 相向双指针法，后两个数 之和 等于 第一个枚举数，是第一题的target变种版，相当于target我自己找，但是还是枚举
#
########################################
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        # The order of the triplets is not important
        # i < j < k
        # The solution set must not contain duplicate triplets.
        ans = []
        n = len(nums)

        for i in range(n - 2):
            x = nums[i]
            # Skip duplicate elements for the first number
            if i > 0 and x == nums[i - 1]:
                continue
            # Pruning
            if x + nums[i+1] + nums[i+2] > 0:
                break
            if x + nums[-1] + nums[-2] < 0:
                continue 

            # Initialize two pointers
            j = i + 1
            k = n - 1

            while j < k:
                s = x + nums[j] + nums[k]

                if s > 0:
                    k -= 1  # Decrease sum by moving the right pointer left

                elif s < 0:
                    j += 1  # Increase sum by moving the left pointer right

                else:
                    ans.append([x, nums[j], nums[k]]) # Found a valid triplet
                    j += 1
                    
                    # Skip duplicate elements for the second number
                    while nums[j] == nums[j - 1] and j < k:
                        j += 1

        return ans

########################################
# 题号：2824. Count Pairs Whose Sum is Less than Target
# 复杂度：时间 O(nlogn) | 空间 O(1)
# 笔记：
#   3. 2824题，统计和小于目标的下标对，因为是统计数对数量，所以排序不影响结果，在双指针使用同时，结果的计算很关键，比如x + y 小于target 那么说明 x + y往左遍历的任意一个数 都满足，此时可以一次获得多个结果
#
########################################
class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        nums.sort()
        ans = left = 0
        right = len(nums) - 1

        while left < right:
            if nums[left] + nums[right] < target:
                ans += right - left
                left += 1
            else:
                right -= 1
                
        return ans

########################################
# 题号：16. 3Sum Closest
# 复杂度：时间 O(N^2) | 空间 O(1)
# 笔记：
#   4. 最接近三数之和，基于三数和遍历方法，要考虑差值
#
########################################
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        # Step 1 ── Sort so we can use the two-pointer pattern and ordered pruning
        nums.sort()
        n = len(nums)

        # `min_diff` keeps the smallest |sum-target| we have seen so far
        min_diff = inf
        # `ans` will store the corresponding sum that is closest to `target`
        ans = 0                      # will be updated before return

        # Enumerate the first element of the triplet
        for i in range(n - 2):
            x = nums[i]

            # Skip duplicate first elements to avoid redundant work
            if i > 0 and x == nums[i - 1]:
                continue

            # ---------- ❶  左端剪枝：三数中的“最小组合”已超过 target ----------
            # With x fixed, the *smallest* possible sum we can form is x + nums[i+1] + nums[i+2]
            s = x + nums[i + 1] + nums[i + 2]
            if s > target:
                # Already larger than target; check if it beats our current best
                if s - target < min_diff:
                    min_diff = s - target
                    ans = s
                # Because all后续 sums will only get bigger, we can break the outer loop
                break

            # ---------- ❷  右端剪枝：三数中的“最大组合”仍然小于 target ----------
            # The *largest* possible sum with current x is x + nums[-2] + nums[-1]
            s = x + nums[-2] + nums[-1]
            if s < target:
                if target - s < min_diff:
                    min_diff = target - s
                    ans = s
                # Even the largest sum is too small ⇒ move to next i
                continue

            # ---------- ❸  Two-pointer search for middle and right ----------
            j, k = i + 1, n - 1
            while j < k:
                s = x + nums[j] + nums[k]

                # Perfect hit: exact target — we cannot get any closer
                if s == target:
                    return s

                if s > target:
                    # Update answer if this sum is closer from the *high* side
                    if s - target < min_diff:
                        min_diff = s - target
                        ans = s
                    k -= 1  # Need a smaller sum → move right pointer left
                else:  # s < target
                    # Update answer if this sum is closer from the *low* side
                    if target - s < min_diff:
                        min_diff = target - s
                        ans = s
                    j += 1  # Need a larger sum → move left pointer right

        # After checking all possibilities, `ans` holds the closest sum
        return ans

########################################
# 题号：18. 4Sum
# 复杂度：时间 O(N^3) | 空间 O(1)
# 笔记：
#   5. 四数和：三数和基础上，枚举2个数，然后剩余数使用双指针，跳过 和 优化都不要忘了
#
########################################
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        # Sort so we can exploit ordering for pruning and two-pointer scanning
        nums.sort()
        ans = []
        n = len(nums)

        # ───────────────── 1st pointer (index a) ─────────────────
        for a in range(n - 3):
            x = nums[a]

            # Skip duplicate 1st elements
            if a > 0 and x == nums[a - 1]:
                continue

            # Lower-bound pruning: smallest possible sum already exceeds target
            if x + nums[a + 1] + nums[a + 2] + nums[a + 3] > target:
                break  # nothing later can work (array is sorted)

            # Upper-bound pruning: largest possible sum still below target
            if x + nums[-3] + nums[-2] + nums[-1] < target:
                continue  # try next a

            # ───────────────── 2nd pointer (index b) ─────────────────
            for b in range(a + 1, n - 2):
                y = nums[b]

                # Skip duplicate 2nd elements
                if b > a + 1 and y == nums[b - 1]:
                    continue

                # Lower-bound pruning for (a, b) pair
                if x + y + nums[b + 1] + nums[b + 2] > target:
                    break  # further b values will only increase the sum

                # Upper-bound pruning for (a, b) pair
                if x + y + nums[-2] + nums[-1] < target:
                    continue  # need a larger b

                # ─────────────── two-pointer search for c, d ───────────────
                c, d = b + 1, n - 1
                while c < d:
                    s = x + y + nums[c] + nums[d]

                    if s > target:
                        d -= 1          # need a smaller sum
                    elif s < target:
                        c += 1          # need a larger sum
                    else:
                        # Found a quadruplet exactly hitting the target
                        ans.append([x, y, nums[c], nums[d]])

                        # Move c past duplicates
                        c += 1
                        while c < d and nums[c] == nums[c - 1]:
                            c += 1

                        # Move d past duplicates
                        d -= 1
                        while c < d and nums[d] == nums[d + 1]:
                            d -= 1

        return ans

########################################
# 题号：611. Valid Triangle Number
# 复杂度：时间 O(N^2) | 空间 O(1)
# 笔记：
#   6. 有效三角形：枚举的数，未必总是第一个，要灵活使用
#
########################################
class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        # 1️⃣  Sort the array so that a ≤ b ≤ c
        nums.sort()
        ans = 0

        # 2️⃣  Iterate k from right to left; nums[k] is the current *largest* side c
        for k in range(len(nums) - 1, 1, -1):
            c = nums[k]

            # —— Prune-A: if the two *smallest* sides already exceed c,
            #            then ANY triple in nums[0..k] forms a triangle
            if nums[0] + nums[1] > c:
                # add C(k+1, 3) = (k+1)·k·(k-1)/6 combinations at once
                ans += (k + 1) * k * (k - 1) // 6
                break   # all remaining (smaller) c will also satisfy, we're done

            # —— Prune-B: if the two largest remaining sides cannot beat c,
            #            no (i, j) works for this k → skip
            if nums[k - 2] + nums[k - 1] <= c:
                continue

            # 3️⃣  Two-pointer scan on the prefix [0, k-1]
            i, j = 0, k - 1
            while i < j:
                # Check if the pair (nums[i], nums[j]) can join with c
                if nums[i] + nums[j] > c:
                    # Every index in [i, j-1] paired with nums[j] also works
                    ans += j - i
                    j -= 1          # shrink right pointer to search new pairs
                else:
                    i += 1          # nums[i] too small, move left pointer right

        return ans



################################################################################
# 专题：相向双指针2 
https://www.bilibili.com/video/BV1Qg411q7ia/?vd_source=d996d505ca380f869ce8821dbe18355d
################################################################################

########################################
# 题号：11. Container With Most Water
# 复杂度：时间 O(N) | 空间 O(1)
########################################
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0 , len(height) - 1
        ans = 0 #Initially, the maximum area is zero

        while left < right:
            #Calculate the width
            width = right - left
            # Calaulate the area with the current boundaries
            area = min(height[left], height[right]) * width

            #Update maximum area if the current area is greater
            ans = max(ans, area)

            #Move the pointer pointing to the shorter line
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return ans

########################################
# 题号：42. Trapping Rain Water
# 复杂度：时间 O(N) | 空间 O(1)
########################################
from typing import List

class Solution:
    def trap(self, height: List[int]) -> int:
        # ans  —— accumulated water
        # left —— left pointer
        # right —— right pointer
        # pre_max —— highest wall seen so far from the left
        # suf_max —— highest wall seen so far from the right
        ans = left = pre_max = suf_max = 0
        right = len(height) - 1

        # Scan until the two pointers meet
        while left < right:
            # Update running maxima from both ends
            pre_max = max(pre_max, height[left])
            suf_max = max(suf_max, height[right])

            # The lower side sets the current water level
            if pre_max < suf_max:
                # Left wall is lower → water level == pre_max
                # Water trapped on position `left`
                ans += pre_max - height[left]
                left += 1            # Move left pointer inward
            else:
                # Right wall is lower (or equal) → level == suf_max
                # Water trapped on position `right`
                ans += suf_max - height[right]
                right -= 1           # Move right pointer inward

        # Total trapped rain water
        return ans



################################################################################
# 专题：滑动窗口
https://www.bilibili.com/video/BV1hd4y1r7Gq/?spm_id_from=333.1387.collection.video_card.click&vd_source=d996d505ca380f869ce8821dbe18355d
################################################################################

########################################
# 题号：209. Minimum Size Subarray Sum
# 复杂度：时间 O(N) | 空间 O(1)
########################################
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n   = len(nums)       # total length of the array
        ans = n + 1           # sentinel: “impossible” large length
        s   = 0               # current window sum
        left = 0              # left boundary of the sliding window

        # ── Expand the right boundary of the window ────────────────
        for right, x in enumerate(nums):   # x == nums[right]
            s += x                         # include new element in sum

            # ── Shrink the left boundary while the sum is big enough ──
            while s >= target:
                # update the best (shortest) length seen so far
                ans = min(ans, right - left + 1)

                # remove nums[left] from window and move left pointer
                s -= nums[left]
                left += 1

        # if ans never updated, return 0; otherwise return the length
        return ans if ans <= n else 0

########################################
# 题号：713. Subarray Product Less Than K
# 复杂度：时间 O(N) | 空间 O(1)
########################################
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:                    # k ≤ 1 ⇒ 不可能有乘积 < k
            return 0

        ans = left = 0
        prod = 1

        for right, x in enumerate(nums):  # 右端指针逐格推进
            prod *= x                     # 把 nums[right] 乘进窗口积

            while prod >= k:              # 若超标，缩小左端直到积 < k
                prod //= nums[left]
                left += 1

            ans += right - left + 1       # 以 right 结尾的合法子数组数量

        return ans

########################################
# 题号：3. Longest Substring Without Repeating Characters
# 复杂度：时间 O(N) | 空间 O(∣Σ∣)
########################################
from collections import defaultdict

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        ans  = 0                  # longest length found so far
        left = 0                  # left boundary of the sliding window
        cnt  = defaultdict(int)   # char → current frequency in window

        # move the right boundary one step at a time
        for right, ch in enumerate(s):
            cnt[ch] += 1          # include s[right] into the window

            # if ch appears twice, shrink window from the left
            while cnt[ch] > 1:    # window is no longer “all unique”
                cnt[s[left]] -= 1
                left += 1         # drop s[left] and move left boundary

            # window [left, right] now has all unique chars
            ans = max(ans, right - left + 1)

        return ans

########################################
# 题号：2958. Length of Longest Subarray With at Most K Frequency
# 复杂度：时间 O(N) | 空间 O(N)
########################################
class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        ans = left = 0                 # ans = longest valid window length
        cnt = Counter()                # frequency of each num in window

        for right, x in enumerate(nums):   # expand window by moving right
            cnt[x] += 1                    # add nums[right] to the counter

            # shrink window from the left **until** x appears ≤ k times
            while cnt[x] > k:             
                cnt[nums[left]] -= 1       # remove nums[left] from window
                left += 1                  # move left boundary rightward

            # window [left, right] is now valid; record its length
            ans = max(ans, right - left + 1)

        return ans

########################################
# 题号：2730. Find the Longest Semi-Repetitive Substring
# 复杂度：时间 O(N) | 空间 O(1)
########################################
class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        ans, left, same = 1, 0, 0        # ans=max length; left=window start; same=#adjacent-equal pairs
        for right in range(1, len(s)):   # expand window by moving right pointer
            same += s[right] == s[right-1]  # add a pair if s[right]==prev char

            # keep at most ONE adjacent-equal pair in the window
            if same > 1:                   
                left += 1                  # drop at least one char from left
                while s[left] != s[left-1]:# skip until we pass the first duplicate pair
                    left += 1
                same = 1                   # exactly one pair now remains

            ans = max(ans, right - left + 1)  # update best length
        return ans

########################################
# 题号：2779. Maximum Beauty of an Array After Applying Operation
# 复杂度：时间 O(nlogn) | 空间 O(1)
########################################
class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        nums.sort()               # sort so we can use a length-bounded window
        ans = left = 0            # ans = best length; left = window start

        for right, x in enumerate(nums):        # expand window by moving right
            # shrink left until every pair in window differs ≤ 2*k  (intervals overlap)
            while x - nums[left] > 2 * k:
                left += 1

            # window [left, right] now represents elements whose intervals share a point
            ans = max(ans, right - left + 1)    # record longest such window

        return ans

########################################
# 题号：1004. Max Consecutive Ones III
# 复杂度：时间 O(N) | 空间 O(1)
########################################
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        ans = left = cnt0 = 0 
        for right, x in enumerate(nums):       # expand window by moving right
            cnt0 += 1 - x                      # if x==0, 1-x==1 → increment zero count; if x==1, no change

            while cnt0 > k:
                cnt0 -= 1 - nums[left]        # if nums[left]==0, decrement zero count
                left += 1                     # move left boundary inward
            ans = max(ans, right - left + 1)  # record its length
            
        return ans

########################################
# 题号：2962. Count Subarrays Where Max Element Appears at Least K Times
# 复杂度：时间 O(N) | 空间 O(1)
########################################
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        mx = max(nums)                  # find the global maximum value in nums
        ans = cnt_mx = left = 0         # ans = result count; cnt_mx = count of mx in window; left = window start index

        for x in nums:                  # right pointer implicitly moves over each element x
            if x == mx:
                cnt_mx += 1             # include a new mx into the current window

            # if we have exactly k occurrences of mx, shrink from the left
            while cnt_mx == k:
                if nums[left] == mx:
                    cnt_mx -= 1         # remove one mx from the window
                left += 1              # advance left pointer to restore < k occurrences

            ans += left                  # all subarrays ending at current right with start < left are valid

        return ans

########################################
# 题号：2302. Count Subarrays With Score Less Than K
# 复杂度：时间 O(N) | 空间 O(1)
########################################
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        ans = s = left = 0          # ans = total valid subarrays, s = current window sum, left = window start index

        for right, x in enumerate(nums):   # expand window by moving right pointer
            s += x                         # include nums[right] in the running sum

            # shrink window from the left while score ≥ k
            # score = s * window_length
            while s * (right - left + 1) >= k:
                s -= nums[left]           # remove nums[left] from sum
                left += 1                 # move left pointer rightwards

            # now window [left..right] is valid:
            # all subarrays ending at right with start in [left..right] are valid
            ans += right - left + 1       # add count of such subarrays

        return ans

########################################
# 题号：1658. Minimum Operations to Reduce X to Zero
# 复杂度：时间 O(N) | 空间 O(1)
########################################
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        # Compute the sum we need to keep in the middle subarray
        target = sum(nums) - x
        if target < 0:
            # If total sum is less than x, we can't reach x by removing ends
            return -1
        
        ans = -1         # length of longest subarray with sum == target
        s = 0            # running sum of current window [left..right]
        left = 0         # left boundary of the sliding window

        # Expand the window by moving 'right'
        for right, v in enumerate(nums):
            s += v                             # include nums[right] in sum

            # Shrink from the left until window sum ≤ target
            while s > target:
                s -= nums[left]               # remove nums[left] from sum
                left += 1                     # move left boundary right

            # If we've hit the exact target sum, update the max length
            if s == target:
                length = right - left + 1
                ans = max(ans, length)

        # If no valid subarray found, return -1
        if ans < 0:
            return -1

        # Otherwise, operations = total length − longest kept subarray
        return len(nums) - ans

########################################
# 题号：1234. Replace the Substring for Balanced String
# 复杂度：时间 O(nC)，其中 n 为 s 的长度，C=4 | 空间 O(C)。如果用哈希表实现，可以做到 O(C)
########################################
from collections import Counter
inf = float('inf')

class Solution:
    def balancedString(self, s: str) -> int:
        m = len(s) // 4
        cnt = Counter(s)

        # Initial balance check
        if len(cnt) == 4 and min(cnt.values()) == m:
            return 0

        ans = inf
        left = 0

        # cnt tracks character counts *outside* the current window s[left...right]
        for right, c in enumerate(s):
            cnt[c] -= 1  # s[right] is now in window, update 'outside' counts

            # While 'outside' counts allow for a balanced string
            while max(cnt.values()) <= m:
                ans = min(ans, right - left + 1)
                
                cnt[s[left]] += 1  # s[left] is removed from window, update 'outside' counts
                left += 1
        
        return ans

########################################
# 题号：76. Minimum Window Substring
# 复杂度：时间 O(m+n) 或 O(m+n+∣Σ∣)，其中 m 为 s 的长度，n 为 t 的长度，∣Σ∣=128。注意 left 只会增加不会减少，二重循环的时间复杂度为 O(m)。使用哈希表写法的时间复杂度为 O(m+n)，数组写法的时间复杂度为 O(m+n+∣Σ∣)。 | 空间 O(∣Σ∣)。无论 m 和 n 有多大，额外空间都不会超过 O(∣Σ∣)
########################################
from collections import defaultdict, Counter

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        ans_left, ans_right = -1, len(s)
        cnt = defaultdict(int)  # Faster than Counter
        for c in t:
            cnt[c] += 1
        # 'less' is the number of unique characters from t we still need to find
        less = len(cnt)

        left = 0
        for right, c in enumerate(s):  # Move the window's right endpoint
            cnt[c] -= 1  # The right-end character enters the window
            if cnt[c] == 0:
                # The requirement for character 'c' is now met
                less -= 1
            
            # While the window contains all required characters from t
            while less == 0:
                if right - left < ans_right - ans_left:  # If a shorter valid window is found
                    ans_left, ans_right = left, right  # Record the current endpoints
                
                x = s[left]  # The character at the left endpoint
                if cnt[x] == 0:
                    # Before moving x out, check its count.
                    # If its count was exactly what was needed (0),
                    # then after moving it out, the window will no longer be valid for x.
                    less += 1
                cnt[x] += 1  # The left-end character leaves the window
                left += 1
                
        return "" if ans_left < 0 else s[ans_left: ans_right + 1]



################################################################################
# 专题：二分查找 https://www.bilibili.com/video/BV1AP41137w7/?vd_source=d996d505ca380f869ce8821dbe18355d
################################################################################

########################################
# 题号：34. Find First and Last Position of Element in Sorted Array
# 复杂度：时间 O(logN) | 空间 O(1)
########################################
class Solution:
    def lower_bound(self, nums: List[int], target: int) -> int:
        # Finds the first index where nums[index] >= target
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            # If mid element is greater than or equal to target,
            # it could be a valid position, so search the left half
            if nums[mid] >= target:
                right = mid - 1
            else:
                # Otherwise, discard the left half and search right
                left = mid + 1
        # At the end, left points to the first element >= target
        return left

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # Find the first index where nums[index] >= target
        start = self.lower_bound(nums, target)

        # Check if target is out of bounds or not actually present
        # Important: check start == len(nums) first to avoid index out of bounds
        if start == len(nums) or nums[start] != target:
            return [-1, -1]  # Target not found

        # Find the first index where nums[index] > target,
        # then subtract 1 to get the last position of target
        end = self.lower_bound(nums, target + 1) - 1

        return [start, end]

########################################
# 题号：2529. Maximum Count of Positive Integer and Negative Integer
# 复杂度：时间 O(logN) | 空间 O(1)
# 笔记：bisect 就是 Python 的“二分查找神器”模块，能帮你高效地在有序数组中查找或插入元素的位置，不用自己写繁琐的二分查找代码。
########################################
class Solution:
    def maximumCount(self, nums: List[int]) -> int:
        # Count of negative numbers = number of elements < 0
        neg = bisect_left(nums, 0)
        
        # Count of positive numbers = number of elements > 0
        pos = len(nums) - bisect_right(nums, 0)
        
        # Return the maximum of the two
        return max(neg, pos)

########################################
# 题号：2300. Successful Pairs of Spells and Potions
# 复杂度：时间 O(NLogN) | 空间 O(1)
# 笔记：因为 y 是整数，而 success / x 是小数，想保证乘积满足，就必须找第一个整数 ≥ success / x，也就是 ceil(success / x)，而不是向下取整的 floor。 
#      ceil(x / y)  ==  (x + y - 1) // y
########################################
class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        potions.sort()  # Sort potions for binary search
        m = len(potions)
        success -= 1     # Trick to simulate ceil(success / x) using integer division

        # For each spell, find number of potions such that spell * potion >= success
        # This is equivalent to potion >= ceil(success / spell)
        return [m - bisect_right(potions, success // x) for x in spells]
        

