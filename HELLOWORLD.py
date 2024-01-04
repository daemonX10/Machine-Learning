name = "*"  # replace with your name

# define heart shape
heart = [
    " **** **** ",
    "***********",
    "***********",
    " *********",
    "  *******",
    "   ***** ",
    "     *  "
]

# print heart with name
for row in heart:
    line = ""
    for char in row:
        if char == "*":
            line += name
        else:
            line += " "
    print(line)
