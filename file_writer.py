

def write_file(file_name, signals):
    with open(file_name,'w') as txtfile:
        for elem in signals:
            row = str(elem[0])+" "+str(elem[1])+" "+str(elem[2]) + '\n'
            txtfile.write(row)
    return
