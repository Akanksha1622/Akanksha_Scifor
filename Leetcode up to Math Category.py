#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd


# # Merge Strings Alternately

# In[49]:


class Solution(object):
    def mergeAlternately(self, word1, word2):
        result = ""
        len1, len2 = len(word1), len(word2)
        i, j = 0, 0

        while i < len1 and j < len2:
            result += word1[i] + word2[j]
            i += 1
            j += 1

        # Append the remaining characters from the longer word
        result += word1[i:] + word2[j:]

        return result


solution_instance = Solution()
output1 = solution_instance.mergeAlternately("abc", "pqr")
output2 = solution_instance.mergeAlternately("ab", "pqrs")
output3 = solution_instance.mergeAlternately("abcd", "pq")

print(output1)  
print(output2)  
print(output3)  


# # Find the Difference

# In[50]:


class Solution(object):
    def findTheDifference(self, s, t):
        result = 0

        # XOR all ASCII values of characters in s and t
        for char in s:
            result ^= ord(char)
        for char in t:
            result ^= ord(char)

        # Convert the XOR result back to character
        return chr(result)


solution_instance = Solution()
output1 = solution_instance.findTheDifference("abcd", "abcde")
output2 = solution_instance.findTheDifference("", "y")

print(output1) 
print(output2) 


# # Find the Index of the First Occurence in a string

# In[51]:


class Solution(object):
    def strStr(self, haystack, needle):
        if not needle:
            return 0  # Empty needle matches at the beginning

        len_haystack, len_needle = len(haystack), len(needle)

        for i in range(len_haystack - len_needle + 1):
            if haystack[i:i + len_needle] == needle:
                return i

        return -1  # Needle not found in haystack
solution_instance = Solution()
output1 = solution_instance.strStr("sadbutsad", "sad")
output2 = solution_instance.strStr("leetcode", "leeto")

print(output1)  
print(output2) 


# # Valid Anagram

# In[52]:


class Solution(object):
    def isAnagram(self, s, t):
        return sorted(s) == sorted(t)

solution_instance = Solution()
output1 = solution_instance.isAnagram("anagram", "nagaram")
output2 = solution_instance.isAnagram("rat", "car")

print(output1) 
print(output2)   


# # Repeated Substring pattern

# In[53]:


class Solution(object):
    def repeatedSubstringPattern(self, s):
        len_s = len(s)

        for i in range(1, len_s // 2 + 1):
            if len_s % i == 0:
                substring = s[:i]
                repeats = len_s // i

                if substring * repeats == s:
                    return True

        return False

solution_instance = Solution()
output1 = solution_instance.repeatedSubstringPattern("abab")
output2 = solution_instance.repeatedSubstringPattern("aba")
output3 = solution_instance.repeatedSubstringPattern("abcabcabcabc")

print(output1)  
print(output2)  
print(output3)  


# # Roman to Integer

# In[54]:


class Solution(object):
    def romanToInt(self, s):
        roman_dict = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }

        result = 0
        prev_value = 0

        for char in s:
            current_value = roman_dict[char]

            if current_value > prev_value:
                result += current_value - 2 * prev_value  # Subtract twice the previous value
            else:
                result += current_value

            prev_value = current_value

        return result

solution_instance = Solution()
roman_numeral1 = "III"
roman_numeral2 = "LVIII"
roman_numeral3 = "MCMXCIV"

result1 = solution_instance.romanToInt(roman_numeral1)
result2 = solution_instance.romanToInt(roman_numeral2)
result3 = solution_instance.romanToInt(roman_numeral3)

print(result1)  
print(result2)  
print(result3) 


# # Move Zeroes

# In[55]:


class Solution(object):
    def moveZeroes(self, nums):
        non_zero_index = 0

        for i in range(len(nums)):
            if nums[i] != 0:
                nums[non_zero_index], nums[i] = nums[i], nums[non_zero_index]
                non_zero_index += 1

        for i in range(non_zero_index, len(nums)):
            nums[i] = 0


solution_instance = Solution()
nums1 = [0, 1, 0, 3, 12]
nums2 = [0]

solution_instance.moveZeroes(nums1)
solution_instance.moveZeroes(nums2)

print(nums1) 
print(nums2)
print(nums2)


# # Plus One

# In[56]:


class Solution(object):
    def plusOne(self, digits):
        n = len(digits)

        # Start from the least significant digit
        for i in range(n - 1, -1, -1):
            # Increment the current digit
            digits[i] += 1

            # Check if there's a carry
            if digits[i] < 10:
                return digits  # No carry, we are done

            # If there is a carry, set the current digit to 0 and continue to the next digit
            digits[i] = 0

        # If we reach here, it means there's a carry at the most significant digit
        # Insert a new digit (1) at the beginning
        digits.insert(0, 1)

        return digits


solution_instance = Solution()
digits1 = [1, 2, 3]
digits2 = [4, 3, 2, 1]
digits3 = [9]

result1 = solution_instance.plusOne(digits1)
result2 = solution_instance.plusOne(digits2)
result3 = solution_instance.plusOne(digits3)

print(result1)  
print(result2)
print(result3) 


# # Sign of the product of an Array

# In[57]:


class Solution(object):
    def arraySign(self, nums):
        product = 1    
        for num in nums:
            product *= num  
        if product > 0:
            return 1
        elif product < 0:
            return -1
        else:
            return 0
solution_instance = Solution()
nums1 = [-1, -2, -3, -4, 3, 2, 1]
nums2 = [1, 5, 0, 2, -3]
nums3 = [-1, 1, -1, 1, -1]

result1 = solution_instance.arraySign(nums1)
result2 = solution_instance.arraySign(nums2)
result3 = solution_instance.arraySign(nums3)

print(result1) 
print(result2) 
print(result3) 


# # Can Make Arithmetic Progression From Sequence

# In[58]:


class Solution(object):
    def canMakeArithmeticProgression(self, arr):
        arr.sort()
        common_difference = arr[1] - arr[0]

        for i in range(1, len(arr)):
            if arr[i] - arr[i - 1] != common_difference:
                return False

        return True
solution_instance = Solution()
arr1 = [3, 5, 1]
arr2 = [1, 2, 4]

result1 = solution_instance.canMakeArithmeticProgression(arr1)
result2 = solution_instance.canMakeArithmeticProgression(arr2)

print(result1) 
print(result2)


# # Monotonic Array

# In[59]:


class Solution(object):
    def isMonotonic(self, nums):
        increasing = decreasing = True

        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                decreasing = False
            elif nums[i] < nums[i - 1]:
                increasing = False

        return increasing or decreasing

solution_instance = Solution()
nums1 = [1, 2, 2, 3]
nums2 = [6, 5, 4, 4]
nums3 = [1, 3, 2]

result1 = solution_instance.isMonotonic(nums1)
result2 = solution_instance.isMonotonic(nums2)
result3 = solution_instance.isMonotonic(nums3)

print(result1)  
print(result2) 
print(result3) 


# # Length of Last Word

# In[60]:


class Solution(object):
    def lengthOfLastWord(self, s):
        # Remove leading and trailing spaces
        s = s.strip()
        words = s.split()
        if words:
            return len(words[-1])
        else:
            return 0

solution_instance = Solution()
input_str1 = "Hello World"
input_str2 = "   fly me   to   the moon  "
input_str3 = "luffy is still joyboy"

result1 = solution_instance.lengthOfLastWord(input_str1)
result2 = solution_instance.lengthOfLastWord(input_str2)
result3 = solution_instance.lengthOfLastWord(input_str3)

print(result1) 
print(result2)  
print(result3) 


# # To Lower Case

# In[32]:


class Solution:
    def toLowerCase(self, s):
        return s.lower()

solution_instance = Solution()

input_str1 = "Hello"
input_str2 = "here"
input_str3 = "LOVELY"

output_str1 = solution_instance.toLowerCase(input_str1)
output_str2 = solution_instance.toLowerCase(input_str2)
output_str3 = solution_instance.toLowerCase(input_str3)

print(f"Input: {input_str1}, Output: {output_str1}")
print(f"Input: {input_str2}, Output: {output_str2}")
print(f"Input: {input_str3}, Output: {output_str3}")


# # BaseBall game

# In[33]:


class Solution(object):
    def calPoints(self, operations):
        """
        :type operations: List[str]
        :rtype: int
        """
        stack = []

        for op in operations:
            if op == "C":
                stack.pop()
            elif op == "D":
                stack.append(2 * stack[-1])
            elif op == "+":
                stack.append(stack[-1] + stack[-2])
            else:
                stack.append(int(op))

        total_sum = sum(stack)
        print("Output for {}: {}".format(operations, total_sum))
        return total_sum

# Example usage:
solution_instance = Solution()

ops1 = ["5", "2", "C", "D", "+"]
ops2 = ["5", "-2", "4", "C", "D", "9", "+", "+"]
ops3 = ["1", "C"]

result1 = solution_instance.calPoints(ops1)
result2 = solution_instance.calPoints(ops2)
result3 = solution_instance.calPoints(ops3)


# # Robot Return to origin

# In[34]:


class Solution(object):
    def judgeCircle(self, moves):
        """
        :type moves: str
        :rtype: bool
        """
        horizontal = 0
        vertical = 0

        for move in moves:
            if move == 'U':
                vertical += 1
            elif move == 'D':
                vertical -= 1
            elif move == 'L':
                horizontal -= 1
            elif move == 'R':
                horizontal += 1

        at_origin = horizontal == 0 and vertical == 0
        print("Moves: {}, At Origin: {}".format(moves, at_origin))
        return at_origin

# Example usage:
solution_instance = Solution()

moves1 = "UD"
moves2 = "LL"

result1 = solution_instance.judgeCircle(moves1)
result2 = solution_instance.judgeCircle(moves2)


# # Find winner in tic tac toe game

# In[35]:


class Solution(object):
    def tictactoe(self, moves):
        """
        :type moves: List[List[int]]
        :rtype: str
        """
        n = 3
        grid = [[' ' for _ in range(n)] for _ in range(n)]

        for i, move in enumerate(moves):
            player = 'A' if i % 2 == 0 else 'B'
            row, col = move
            grid[row][col] = player

            winner = self.check_winner(grid)
            if winner:
                print("Moves: {}, Winner: {}".format(moves, winner))
                return winner

        if len(moves) == n * n:
            print("Moves: {}, Result: Draw".format(moves))
            return "Draw"
        else:
            print("Moves: {}, Result: Pending".format(moves))
            return "Pending"

    def check_winner(self, grid):
        for i in range(3):
            if grid[i][0] == grid[i][1] == grid[i][2] != ' ':
                return grid[i][0]  # Winner in the row
            if grid[0][i] == grid[1][i] == grid[2][i] != ' ':
                return grid[0][i]  # Winner in the column

        if grid[0][0] == grid[1][1] == grid[2][2] != ' ':
            return grid[0][0]  # Winner in the diagonal

        if grid[0][2] == grid[1][1] == grid[2][0] != ' ':
            return grid[0][2]  # Winner in the diagonal

        return None

# Example usage:
solution_instance = Solution()

moves1 = [[0, 0], [2, 0], [1, 1], [2, 1], [2, 2]]
moves2 = [[0, 0], [1, 1], [0, 1], [0, 2], [1, 0], [2, 0]]
moves3 = [[0, 0], [1, 1], [2, 0], [1, 0], [1, 2], [2, 1], [0, 1], [0, 2], [2, 2]]

result1 = solution_instance.tictactoe(moves1)
result2 = solution_instance.tictactoe(moves2)
result3 = solution_instance.tictactoe(moves3)


# # Robot bounded in a circle

# In[36]:


class Solution(object):
    def isRobotBounded(self, instructions):
        """
        :type instructions: str
        :rtype: bool
        """
        x, y = 0, 0
        direction = 0  # 0: North, 1: East, 2: South, 3: West

        for instruction in instructions:
            if instruction == "G":
                if direction == 0:
                    y += 1
                elif direction == 1:
                    x += 1
                elif direction == 2:
                    y -= 1
                else:
                    x -= 1
            elif instruction == "L":
                direction = (direction - 1) % 4
            elif instruction == "R":
                direction = (direction + 1) % 4

        # Robot returns to the starting point or is facing a direction other than North
        return (x == 0 and y == 0) or direction != 0

# Example usage:
solution_instance = Solution()

instructions1 = "GGLLGG"
instructions2 = "GG"
instructions3 = "GL"

result1 = solution_instance.isRobotBounded(instructions1)
result2 = solution_instance.isRobotBounded(instructions2)
result3 = solution_instance.isRobotBounded(instructions3)


# # Richest customer wealth

# In[37]:


class Solution(object):
    def maximumWealth(self, accounts):
        """
        :type accounts: List[List[int]]
        :rtype: int
        """
        max_wealth = 0

        for customer in accounts:
            wealth = sum(customer)
            max_wealth = max(max_wealth, wealth)

        return max_wealth

solution_instance = Solution()

accounts1 = [[1, 2, 3], [3, 2, 1]]
accounts2 = [[1, 5], [7, 3], [3, 5]]
accounts3 = [[2, 8, 7], [7, 1, 3], [1, 9, 5]]

result1 = solution_instance.maximumWealth(accounts1)
result2 = solution_instance.maximumWealth(accounts2)
result3 = solution_instance.maximumWealth(accounts3)

print("Maximum Wealth for accounts1: {}".format(result1))
print("Maximum Wealth for accounts2: {}".format(result2))
print("Maximum Wealth for accounts3: {}".format(result3))


# # Matrix diagonal sum

# In[38]:


class Solution(object):
    def diagonalSum(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: int
        """
        n = len(mat)
        diagonal_sum = 0

        for i in range(n):
            diagonal_sum += mat[i][i]  # Primary diagonal
            diagonal_sum += mat[i][n - i - 1]  # Secondary diagonal

        # Adjust for the element at the center (if matrix size is odd)
        if n % 2 == 1:
            diagonal_sum -= mat[n // 2][n // 2]

        return diagonal_sum

# Example usage:
solution_instance = Solution()

mat1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
mat2 = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
mat3 = [[5]]

result1 = solution_instance.diagonalSum(mat1)
result2 = solution_instance.diagonalSum(mat2)
result3 = solution_instance.diagonalSum(mat3)

print("Diagonal Sum for mat1: {}".format(result1))
print("Diagonal Sum for mat2: {}".format(result2))
print("Diagonal Sum for mat3: {}".format(result3))


# # Spiral Matrix

# In[39]:


class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        result = []
        
        while matrix:
            result += matrix.pop(0)
            
            if matrix and matrix[0]:
                for row in matrix:
                    result.append(row.pop())
                    
            if matrix:
                result += matrix.pop()[::-1]
                
            if matrix and matrix[0]:
                for row in matrix[::-1]:
                    result.append(row.pop(0))
        
        return result

# Example usage:
solution_instance = Solution()

matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
matrix2 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

result1 = solution_instance.spiralOrder(matrix1)
result2 = solution_instance.spiralOrder(matrix2)

print("Spiral Order for matrix1: {}".format(result1))
print("Spiral Order for matrix2: {}".format(result2))


# # Set Matrix Zeroes

# In[40]:


class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        rows, cols = len(matrix), len(matrix[0])
        first_row_has_zero = any(matrix[0][j] == 0 for j in range(cols))
        first_col_has_zero = any(matrix[i][0] == 0 for i in range(rows))

        # Mark zeros on the first row and column
        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0

        # Set zeros based on the marks
        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0

        # Set zeros for the first row and column if necessary
        if first_row_has_zero:
            for j in range(cols):
                matrix[0][j] = 0

        if first_col_has_zero:
            for i in range(rows):
                matrix[i][0] = 0

# Example usage:
solution_instance = Solution()

matrix1 = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
matrix2 = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]

solution_instance.setZeroes(matrix1)
solution_instance.setZeroes(matrix2)

print("Modified Matrix for matrix1: {}".format(matrix1))
print("Modified Matrix for matrix2: {}".format(matrix2))


# # Count Odd numbers in an interval range

# In[41]:


class Solution(object):
    def countOdds(self, low, high):
        """
        :type low: int
        :type high: int
        :rtype: int
        """
        # If low is odd, start with low, otherwise start with the next odd number
        start = low if low % 2 == 1 else low + 1

        # If high is odd, include it; otherwise, exclude it
        end = high if high % 2 == 1 else high - 1

        # Calculate the count of odd numbers between low and high (inclusive)
        count = max(0, (end - start) // 2 + 1)

        return count

# Example usage:
solution_instance = Solution()

low1, high1 = 3, 7
low2, high2 = 8, 10

result1 = solution_instance.countOdds(low1, high1)
result2 = solution_instance.countOdds(low2, high2)

print("Count of Odds between {} and {}: {}".format(low1, high1, result1))
print("Count of Odds between {} and {}: {}".format(low2, high2, result2))


# # Average Salary Excluding the Minimum and Maximum Salary

# In[42]:


class Solution(object):
    def average(self, salary):
        """
        :type salary: List[int]
        :rtype: float
        """
        min_salary = min(salary)
        max_salary = max(salary)

        # Calculate the sum excluding the minimum and maximum salary
        total_sum = sum(salary) - min_salary - max_salary

        # Calculate the average
        average = total_sum / (len(salary) - 2)

        return average

# Example usage:
solution_instance = Solution()

salary1 = [4000, 3000, 1000, 2000]
salary2 = [1000, 2000, 3000]

result1 = solution_instance.average(salary1)
result2 = solution_instance.average(salary2)

print("Average for salary1: {:.5f}".format(result1))
print("Average for salary2: {:.5f}".format(result2))


# # Lemonade Change

# In[43]:


class Solution(object):
    def average(self, salary):
        """
        :type salary: List[int]
        :rtype: float
        """
        min_salary = min(salary)
        max_salary = max(salary)

        # Calculate the sum excluding the minimum and maximum salary
        total_sum = sum(salary) - min_salary - max_salary

        # Calculate the average
        average = total_sum / (len(salary) - 2)

        return average

    def lemonadeChange(self, bills):
        """
        :type bills: List[int]
        :rtype: bool
        """
        # Initialize variables to keep track of available change
        five_count = ten_count = 0

        for bill in bills:
            if bill == 5:
                five_count += 1
            elif bill == 10:
                ten_count += 1
                five_count -= 1
            elif ten_count > 0:
                ten_count -= 1
                five_count -= 1
            else:
                five_count -= 3

            # If at any point we have negative change, return False
            if five_count < 0:
                return False

        return True

# Example usage:
solution_instance = Solution()

salary1 = [4000, 3000, 1000, 2000]
salary2 = [1000, 2000, 3000]

result1 = solution_instance.average(salary1)
result2 = solution_instance.average(salary2)

print("Average for salary1: {:.5f}".format(result1))
print("Average for salary2: {:.5f}".format(result2))

bills1 = [5, 5, 5, 10, 20]
bills2 = [5, 5, 10, 10, 20]

result3 = solution_instance.lemonadeChange(bills1)
result4 = solution_instance.lemonadeChange(bills2)

print("Lemonade change for bills1: {}".format(result3))
print("Lemonade change for bills2: {}".format(result4))


# # Lrgest perimeter triangle

# In[44]:


class Solution(object):
    def largestPerimeter(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # Sort the array in descending order
        nums.sort(reverse=True)

        for i in range(len(nums) - 2):
            # Check if the current three lengths can form a triangle
            if nums[i] < nums[i + 1] + nums[i + 2]:
                return nums[i] + nums[i + 1] + nums[i + 2]

        return 0

# Example usage:
solution_instance = Solution()

nums1 = [2, 1, 2]
nums2 = [1, 2, 1, 10]

result1 = solution_instance.largestPerimeter(nums1)
result2 = solution_instance.largestPerimeter(nums2)

print("Largest Perimeter for nums1: {}".format(result1))
print("Largest Perimeter for nums2: {}".format(result2))


# # Check if it is a straight line

# In[45]:


class Solution(object):
    def checkStraightLine(self, coordinates):
        """
        :type coordinates: List[List[int]]
        :rtype: bool
        """
        # Check if the slope between the first two points is the same for all points
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]

        for i in range(2, len(coordinates)):
            x, y = coordinates[i]

            # Check if the slopes are not equal
            if (x2 - x1) * (y - y1) != (x - x1) * (y2 - y1):
                return False

        return True

# Example usage:
solution_instance = Solution()

coordinates1 = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
coordinates2 = [[1, 1], [2, 2], [3, 4], [4, 5], [5, 6], [7, 7]]

result1 = solution_instance.checkStraightLine(coordinates1)
result2 = solution_instance.checkStraightLine(coordinates2)

print("Is it a straight line for coordinates1? {}".format(result1))
print("Is it a straight line for coordinates2? {}".format(result2))


# # Add Binary

# In[46]:


class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        # Initialize variables to store the result and carry
        result = []
        carry = 0

        # Convert input strings to lists for easy manipulation
        a = list(a)
        b = list(b)

        # Pad the shorter string with leading zeros
        len_a, len_b = len(a), len(b)
        max_len = max(len_a, len_b)
        a = ['0'] * (max_len - len_a) + a
        b = ['0'] * (max_len - len_b) + b

        # Iterate through the strings from right to left
        for i in range(max_len - 1, -1, -1):
            bit_sum = int(a[i]) + int(b[i]) + carry
            result.append(str(bit_sum % 2))
            carry = bit_sum // 2

        # If there is a carry after the loop, add it to the result
        if carry:
            result.append(str(carry))

        # Reverse the result list and convert it to a string
        return ''.join(result[::-1])

# Example usage:
solution_instance = Solution()

a1, b1 = "11", "1"
a2, b2 = "1010", "1011"

result1 = solution_instance.addBinary(a1, b1)
result2 = solution_instance.addBinary(a2, b2)

print("Binary sum for a1 and b1: {}".format(result1))
print("Binary sum for a2 and b2: {}".format(result2))


# # Multiply Strings

# In[47]:


class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        # Initialize a list to store the product
        product = [0] * (len(num1) + len(num2))

        # Multiply each digit of num1 with each digit of num2
        for i in range(len(num1) - 1, -1, -1):
            for j in range(len(num2) - 1, -1, -1):
                # Calculate the product and add it to the corresponding position in the result
                digit_product = int(num1[i]) * int(num2[j])
                total_sum = digit_product + product[i + j + 1]
                product[i + j + 1] = total_sum % 10
                product[i + j] += total_sum // 10

        # Convert the product list to a string
        result = ''.join(map(str, product))

        # Remove leading zeros
        result = result.lstrip('0')

        # If the result is empty, return '0'
        return result if result else '0'

# Example usage:
solution_instance = Solution()

num1_1, num2_1 = "2", "3"
num1_2, num2_2 = "123", "456"

result1 = solution_instance.multiply(num1_1, num2_1)
result2 = solution_instance.multiply(num1_2, num2_2)

print("Product for num1_1 and num2_1: {}".format(result1))
print("Product for num1_2 and num2_2: {}".format(result2))


# # Pow(x,n)

# In[48]:


class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        # Base case: If n is 0, return 1
        if n == 0:
            return 1.0

        # Handle negative exponent by taking reciprocal
        if n < 0:
            x = 1 / x
            n = -n

        # Recursive calculation of power using binary exponentiation
        def power(x, n):
            # Base case: If exponent is 0, return 1
            if n == 0:
                return 1.0

            # Divide the problem into subproblems (divide exponent by 2)
            half_power = power(x, n // 2)

            # Combine the subproblems to get the final result
            if n % 2 == 0:
                return half_power * half_power
            else:
                return half_power * half_power * x

        return power(x, n)

# Example usage:
solution_instance = Solution()

x1, n1 = 2.00000, 10
x2, n2 = 2.10000, 3
x3, n3 = 2.00000, -2

result1 = solution_instance.myPow(x1, n1)
result2 = solution_instance.myPow(x2, n2)
result3 = solution_instance.myPow(x3, n3)

print("Result for x1 and n1: {:.5f}".format(result1))
print("Result for x2 and n2: {:.5f}".format(result2))
print("Result for x3 and n3: {:.5f}".format(result3))


# In[ ]:




