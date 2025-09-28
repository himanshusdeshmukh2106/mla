# Read the requirements.txt file
with open('requirements.txt', 'r') as f:
    lines = f.readlines()

# Remove version specifiers for numpy and pandas-ta
new_lines = []
for line in lines:
    if line.strip().startswith('numpy') and '==' in line:
        new_lines.append('numpy\n')
    elif line.strip().startswith('pandas-ta') and '==' in line:
        new_lines.append('pandas-ta\n')
    else:
        new_lines.append(line)

# Write the modified requirements.txt file
with open('requirements.txt', 'w') as f:
    f.writelines(new_lines)

# Install packages again with the modified requirements.txt
!pip install -r requirements.txtthe