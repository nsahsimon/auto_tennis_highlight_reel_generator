import os
def log_data(data):
    with open("logs.txt",'a') as file:
        file.write(f"{data} \n")

data = ['1', '2', '3', '4']

log_data(data)

