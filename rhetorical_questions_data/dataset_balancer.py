

def create_balanced(dataset):
    rhet_count = 0
    with open(
            "rhetorical_questions_data/" + dataset + ".txt",
            encoding="utf8", mode="r") as fr:
        lines = fr.readlines()
    for line in lines:
        label = line[0]
        if label == "1":
            rhet_count += 1
    print("RHET COUNT", rhet_count)
    non_rhet_count = rhet_count

    new_lines = []

    for line in lines:
        label = line[0]
        if label == "1":
            new_lines.append(line)
            rhet_count -= 1
        if label == "0" and non_rhet_count > 0:
            new_lines.append(line)
            non_rhet_count -= 1
    with open("rhetorical_questions_data/" + dataset + "_balanced.txt", encoding="utf8", mode="w") as fw:
        fw.writelines(new_lines)

if __name__ == '__main__':
    create_balanced(dataset="train1")
    create_balanced(dataset="validate1")
    create_balanced(dataset="test")