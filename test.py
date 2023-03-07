
for j in range(3):
    for i in range(10):
        global idx
        idx = i
        print("hello: {i}")

print(f"final idx: {idx}")