# Sum of Natural Numbers (0 to 10)
n = 10
sum = 0
for i in range(n + 1):
    sum += i
print(sum)


# Count Digits
x = 3459
count = 0
while x != 0:
    x //= 10
    count += 1
print(count)


# Check Palindrome
x = 989
x_str = str(x)
if x_str == x_str[::-1]:
    print("yes")
else:
    print("no")


# Factorial of a Number
x = 5
factorial = 1
for i in range(1, x + 1):
    factorial *= i
print(factorial)
