#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd


# # Merge Strings Alternately

# In[5]:


def merged(word1,word2):
        result = ""
        len1, len2 = len(word1), len(word2)
        i,j=0,0
        while i < len1 and j < len2:
            result += word1[i ] + " " + word2[j ]+ " "
            i += 1
            j += 1
        result += " ".join(word1[i:]) + " ".join(word2[j:])
        print("merged: ", result) 
merged("abc","pqr")
merged("ab","pqrs")
merged("abcd","pq")


# # Find the Difference

# In[11]:


def add_letter(s,t):
    s_count={}
    t_count={}
    # Count characters in string s
    for char in s:
        s_count[char] = s_count.get(char, 0) + 1

    # Count characters in string t
    for char in t:
        t_count[char] = t_count.get(char, 0) + 1
        # Find the character with different frequencies
        # Find the character with different frequencies
    for char, count in t_count.items():
        if s_count.get(char, 0) != count:
            return char
print("added letter: ",add_letter("abcd","abcde"))
print("added letter: ",add_letter("","y"))


# # Find the Index of the First Occurence in a string

# In[25]:


def occurence(haystack, needle):
    if not needle:
        return 0  # Empty needle is always present in the haystack

    len_h, len_n = len(haystack), len(needle)
    
    for i in range(len_h - len_n + 1):
        if haystack[i:i + len_n] == needle:
            return i

    return -1  # Needle not found in haystack
print("Occurence : ",occurence("sadbutsad","sad"))
print("Occurence: ",occurence("leetcode","leeto"))


# # Valid Anagram

# In[19]:


def isanagram(s,t):
    return sorted(s) == sorted(t)
print(isanagram("anagram","nagaram"))
print(isanagram("rat","car"))  


# # Repeated Substring pattern

# In[23]:


def substring(s):
    length = len(s)
    for i in range(1, length // 2 + 1):
        if length % i == 0:
            divisor = length // i
            substring = s[:i]

            if substring * divisor == s:
                return True

    return False
print(substring("abab"))
print(substring("aba"))
print(substring("abcabcabcabc"))


# # Roman to Integer

# In[28]:


def roman(s):
    roman_dict = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        a'D': 500,
        'M': 1000
    }

    result = 0
    prev = 0

    for char in reversed(s):
        current = roman_dict[char]

        if current < prev:
            result -= current
        else:
            result += current

        prev = current

    return result
print(roman("III"))
print(roman("LVIII"))
print(roman("MCMXCIV"))


# # Move Zeroes

# In[32]:


def moveZeroes(nums):
    nonzero_ind = 0

    # Iterating through the array
    for i in range(len(nums)):
        if nums[i] != 0:
            # Swap the current element with the element at the non-zero index
            nums[i], nums[nonzero_ind] = nums[nonzero_ind], nums[i]
            # Move the non-zero index forward
            nonzero_ind += 1
nums1 = [0, 1, 0, 3, 12]
nums2 = [0]

moveZeroes(nums1)
moveZeroes(nums2)

print(nums1)
print(nums2)


# # Plus One

# In[34]:


def increment(digits):
    carry = 1  # Initialize carry to 1 as we want to increment by one

    for i in range(len(digits) - 1, -1, -1):
        total = digits[i] + carry
        digits[i] = total % 10  
        carry = total // 10  

    if carry:
        digits.insert(0, carry) 

    return digits

digits1 = [1, 2, 3]
digits2 = [4, 3, 2, 1]
digits3 = [9]

result1 = increment(digits1)
result2 = increment(digits2)
result3 = increment(digits3)

print(result1)  
print(result2)
print(result3)


# # Sign of the product of an Array

# In[36]:


def arraySign(nums):
    product_sign = 1 
    for num in nums:
        if num == 0:
            return 0  
        if num < 0:
            product_sign *= -1  

    return product_sign

nums1 = [-1, -2, -3, -4, 3, 2, 1]
nums2 = [1, 5, 0, 2, -3]
nums3 = [-1, 1, -1, 1, -1]

output1 = arraySign(nums1)
output2 = arraySign(nums2)
output3 = arraySign(nums3)

print(output1)  
print(output2) 
print(output3) 


# # Can Make Arithmetic Progression From Sequence

# In[37]:


def ArithmeticProgression(arr):
    arr.sort()  

    common_difference = arr[1] - arr[0]

    for i in range(2, len(arr)):
        if arr[i] - arr[i - 1] != common_difference:
            return False

    return True
arr1 = [3, 5, 1]
arr2 = [1, 2, 4]

result1 = ArithmeticProgression(arr1)
result2 = ArithmeticProgression(arr2)

print(result1)  
print(result2) 


# # Monotonic Array

# In[38]:


def monotonic(nums):
    increasing = decreasing = True

    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            decreasing = False
        if nums[i] < nums[i - 1]:
            increasing = False

    return increasing or decreasing

nums1 = [1, 2, 2, 3]
nums2 = [6, 5, 4, 4]
nums3 = [1, 3, 2]

result1 = monotonic(nums1)
result2 = monotonic(nums2)
result3 = monotonic(nums3)

print(result1)  
print(result2) 
print(result3)

