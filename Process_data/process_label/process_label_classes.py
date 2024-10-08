if __name__ == "__main__":
    classes = [];
    with open("label-name.txt", "r") as file:
        for row in file:
            truncated = row[:4];    
            classes.append(truncated);
    with open("classes.txt", "a") as file:
        for clazz in classes:
            file.write(clazz + "\n");