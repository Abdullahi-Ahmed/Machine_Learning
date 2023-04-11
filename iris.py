# define variables
total_marbles = 10
red_marbles = 4
blue_marbles = 6

# probability of drawing a blue marble
p_blue = blue_marbles / total_marbles

# probability of drawing a red marble given that we drew a blue marble
p_red_given_blue = red_marbles / (total_marbles - 1)

# calculate the conditional probability using Bayes' theorem
p_blue_given_red = p_red_given_blue * p_blue / (red_marbles / total_marbles)

print("The probability of drawing a blue marble is:", p_blue)
print("The probability of drawing a red marble given that we drew a blue marble is:", p_red_given_blue)
print("The probability of drawing a blue marble given that we drew a red marble is:", p_blue_given_red)