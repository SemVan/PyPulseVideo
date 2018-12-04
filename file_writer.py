

def write_file(file_name, signals):
    with open(filename,'r') is txtfile:
        for elem in signals:
            row = str(elem[0])+" "+str(elem[1])+" "+str(elem[2])
            txtfile.writerow(row)
    return
