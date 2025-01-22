import os
import re
import csv
import math
import torch
import pickle

import numpy as np
from tqdm import tqdm


def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def match_feature(x, all_for_assign):
    x_p = np.zeros((len(x), 7))

    for j in range(len(x)):
        if x[j] == 'ALA':
            x_p[j] = all_for_assign[0, :]
        elif x[j] == 'CYS':
            x_p[j] = all_for_assign[1, :]
        elif x[j] == 'ASP':
            x_p[j] = all_for_assign[2, :]
        elif x[j] == 'GLU':
            x_p[j] = all_for_assign[3, :]
        elif x[j] == 'PHE':
            x_p[j] = all_for_assign[4, :]
        elif x[j] == 'GLY':
            x_p[j] = all_for_assign[5, :]
        elif x[j] == 'HIS':
            x_p[j] = all_for_assign[6, :]
        elif x[j] == 'ILE':
            x_p[j] = all_for_assign[7, :]
        elif x[j] == 'LYS':
            x_p[j] = all_for_assign[8, :]
        elif x[j] == 'LEU':
            x_p[j] = all_for_assign[9, :]
        elif x[j] == 'MET':
            x_p[j] = all_for_assign[10, :]
        elif x[j] == 'ASN':
            x_p[j] = all_for_assign[11, :]
        elif x[j] == 'PRO':
            x_p[j] = all_for_assign[12, :]
        elif x[j] == 'GLN':
            x_p[j] = all_for_assign[13, :]
        elif x[j] == 'ARG':
            x_p[j] = all_for_assign[14, :]
        elif x[j] == 'SER':
            x_p[j] = all_for_assign[15, :]
        elif x[j] == 'THR':
            x_p[j] = all_for_assign[16, :]
        elif x[j] == 'VAL':
            x_p[j] = all_for_assign[17, :]
        elif x[j] == 'TRP':
            x_p[j] = all_for_assign[18, :]
        elif x[j] == 'TYR':
            x_p[j] = all_for_assign[19, :]

    return x_p


def read_atoms(file, chain="."):
    pattern = re.compile(chain)

    atoms = []
    ajs = []

    for line in file:
        line = line.strip()
        if line.startswith("ATOM"):
            type = line[12:16].strip()
            chain = line[21:22]
            if type == "CA" and re.match(pattern, chain):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                ajs_id = line[17:20]
                atoms.append((x, y, z))
                ajs.append(ajs_id)

    return atoms, ajs


def compute_contacts(atoms, threshold):
    contacts = []
    for i in range(len(atoms) - 2):
        for j in range(i + 2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i, j))
                contacts.append((j, i))
    return contacts


def knn(atoms, k=5):
    x = np.zeros((len(atoms), len(atoms)))
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            x[i, j] = dist(atoms[i], atoms[j])
    index = np.argsort(x, axis=-1)

    contacts = []
    for i in range(len(atoms)):
        num = 0
        for j in range(len(atoms)):
            if index[i, j] != i and index[i, j] != i - 1 and index[i, j] != i + 1:
                contacts.append((i, index[i, j]))
                num += 1
            if num == k:
                break

    return contacts


def pdb_to_cm(file, threshold):
    atoms, x = read_atoms(file)
    r_contacts = compute_contacts(atoms, threshold)
    k_contacts = knn(atoms)
    return r_contacts, k_contacts, x


node_list = []
r_edge_list = []
k_edge_list = []


def data_processing(dictionary, ppi):
    # Generate Adjacency Matrix
    distance = 10
    all_for_assign = np.loadtxt("./all_assign.txt")

    pdb_file_dir = "./raw_data/results/" + dictionary[int(ppi[0])] + "_" + dictionary[int(ppi[1])] + '.pdb'
    if os.path.exists(pdb_file_dir):
        r_contacts, k_contacts, x = pdb_to_cm(open(pdb_file_dir, "r"), distance)
        x = match_feature(x, all_for_assign)

        node_list.append(x)
        r_edge_list.append(r_contacts)
        k_edge_list.append(k_contacts)

    else:
        print(pdb_file_dir, "not found")


dictionary_path = "/root/autodl-fs/protein.SHS27k.sequences_3di.dictionary.csv"
ppi_path = "/root/autodl-fs/SHS27k_ppi.pkl"

dictionary = []  # 蛋白id
ppi_list = []

with open(dictionary_path) as f:
    reader = csv.reader(f)
    for row in reader:
        dictionary.append(row[0])

with open(ppi_path, "rb") as tf:
    ppi_list = pickle.load(tf)

for ppi in tqdm(ppi_list):
    data_processing(dictionary, ppi)

print(len(node_list))
print(len(r_edge_list))
print(len(k_edge_list))

np.save("./dock.protein.rball.edges.dock.npy", np.asarray(r_edge_list, dtype=object))
np.save("./dock.protein.knn.edges.dock.npy", np.array(k_edge_list))
torch.save(node_list, "./dock.protein.nodes.dock.pt")


