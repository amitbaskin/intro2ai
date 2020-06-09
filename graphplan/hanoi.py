import sys


def create_domain_file(domain_file_name, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    domain_file = open(domain_file_name, 'w')  # use domain_file.write(str) to write to domain_file
    domain_file.write("Propositions:\n")
    for i in range(n_):
        domain_file.write(disks[i] + "Clear ")

        for j in range(m_):
            domain_file.write(disks[i] + "ON" + pegs[j] + " ")

        for j in range(i + 1, n_):
            domain_file.write(disks[i] + "ON" + disks[j] + " ")

    for i in range(m_):
        domain_file.write(pegs[i] + "Clear ")
    domain_file.write("\n")
    domain_file.write("Actions:\n")

    for i in range(n_):  # peg to peg
        for j in range(m_):
            for k in range(m_):
                if j == k:
                    continue
                domain_file.write("Name: MV{}F{}T{} \n".format(disks[i], pegs[j], pegs[k]))
                domain_file.write("pre: {}Clear {}Clear {}ON{} \n".format(disks[i], pegs[k], disks[i], pegs[j]))
                domain_file.write("add: {}ON{} {}Clear \n".format(disks[i], pegs[k], pegs[j]))
                domain_file.write("delete: {}Clear {}ON{} \n".format(pegs[k], disks[i], pegs[j]))

    for i in range(n_):  # disk to disk
        for j in range(i + 1, n_):
            for k in range(i + 1, n_):
                if j == k:
                    continue
                domain_file.write("Name: MV{}F{}T{} \n".format(disks[i], disks[j], disks[k]))
                domain_file.write("pre: {}Clear {}Clear {}ON{} \n".format(disks[i], disks[k], disks[i], disks[j]))
                domain_file.write("add: {}ON{} {}Clear \n".format(disks[i], disks[k], disks[j]))
                domain_file.write("delete: {}Clear {}ON{} \n".format(disks[k], disks[i], disks[j]))

    for i in range(n_):  # peg to disk
        for j in range(m_):
            for k in range(i + 1, n_):
                domain_file.write("Name: MV{}F{}T{} \n".format(disks[i], pegs[j], disks[k]))
                domain_file.write("pre: {}Clear {}Clear {}ON{} \n".format(disks[i], disks[k], disks[i], pegs[j]))
                domain_file.write("add: {}ON{} {}Clear \n".format(disks[i], disks[k], pegs[j]))
                domain_file.write("delete: {}Clear {}ON{} \n".format(disks[k], disks[i], pegs[j]))

    for i in range(n_):  # disk to peg
        for j in range(i + 1, n_):
            for k in range(m_):
                domain_file.write("Name: MV{}F{}T{} \n".format(disks[i], disks[j], pegs[k]))
                domain_file.write("pre: {}Clear {}Clear {}ON{} \n".format(disks[i], pegs[k], disks[i], disks[j]))
                domain_file.write("add: {}ON{} {}Clear \n".format(disks[i], pegs[k], disks[j]))
                domain_file.write("delete: {}Clear {}ON{} \n".format(pegs[k], disks[i], disks[j]))

    domain_file.close()


def create_problem_file(problem_file_name_, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    problem_file = open(problem_file_name_, 'w')  # use problem_file.write(str) to write to problem_file
    problem_file.write("Initial state: ")
    for i in range(n_ - 1):
        problem_file.write(disks[i] + "ON" + disks[i + 1] + " ")
    problem_file.write(disks[0] + "Clear ")
    for i in range(1, m_):
        problem_file.write(pegs[i] + "Clear ")

    problem_file.write(disks[-1] + "ON" + pegs[0] + " \n")

    problem_file.write("Goal: ")
    for i in range(n_ - 1):
        problem_file.write(disks[i] + "ON" + disks[i + 1] + " ")
    problem_file.write(disks[-1] + "ON" + pegs[-1] + " \n")

    problem_file.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: hanoi.py n m')
        sys.exit(2)

    n = int(float(sys.argv[1]))  # number of disks
    m = int(float(sys.argv[2]))  # number of pegs

    domain_file_name = 'hanoi_%s_%s_domain.txt' % (n, m)
    problem_file_name = 'hanoi_%s_%s_problem.txt' % (n, m)

    create_domain_file(domain_file_name, n, m)
    create_problem_file(problem_file_name, n, m)
