

if __name__ == "__main__":
    with open("/home/ubuntu/datasets/ucf101/old-test.txt") as old:
        with open("/home/ubuntu/datasets/ucf101/test.txt", "w") as file:
            for line in old:
                path, label, num_frames = line.strip().split(" ")
                file.write("{} {} {}\n".format(path, int(label) - 1, num_frames))
