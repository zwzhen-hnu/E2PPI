from datetime import datetime
import os
from subprocess import Popen, PIPE
import shutil
from tqdm import tqdm
import csv
from joblib import Parallel, delayed


raw_dir = "./raw_data/"


def get_date():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def rename_chains(pdb_dir, pdb_out_dir, reversed=True):
    chains_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    all_chains = chains_choices[:]
    if reversed:
        all_chains.reverse()

    new_chains = []
    chains_seen = []
    with open(pdb_out_dir, 'w') as out:
        with open(pdb_dir, 'r') as f:
            for line in f.readlines():
                if line[:6] == 'HEADER':
                    continue
                if line[:4] == 'ATOM' or line[:6] == 'HETATM':
                    line = [char for char in line]
                    if line[21] not in chains_seen:
                        new_chains.append(all_chains.pop())
                        chains_seen.append(line[21])
                    line[21] = new_chains[-1]
                    line = ''.join(line)
                out.write(line)
    return pdb_out_dir, ''.join(new_chains)


def execute_hdock(pid1, pid2, PDB_TARGET, PDB_LIGAND):
    # pdb.set_trace()
    args = ['hdock', PDB_TARGET, PDB_LIGAND, '-out', 'Hdock.out']
    print(' '.join(args))
    process = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stderr)
    return 1


def run_hdock_one(out_dock, pid1, pid2):
    print("[ {} ] Start docking {} with {}...".format(get_date(), pid1, pid2))

    # change dir to docking path
    curr_dir = os.getcwd()
    os.chdir(out_dock)
    print("renaming chains")
    # renaming chains
    out_pdb1 = './{}_renamed.pdb'.format(pid1)
    out_pdb2 = './{}_renamed.pdb'.format(pid2)
    PDB_TARGET, new_ch1 = rename_chains(raw_dir + '/STRING_AF2DB/' + pid1 + '.pdb', out_pdb1, True)
    PDB_LIGAND, new_ch2 = rename_chains(raw_dir + '/STRING_AF2DB/' + pid2 + '.pdb', out_pdb2, False)

    if not os.path.exists(out_dock + 'Hdock.out'):
        print("hdocking.....")
        execute_hdock(pid1, pid2, out_pdb1, out_pdb2)
    else:
        print("[ {} ] Hdock already exists.".format(out_dock + '/Hdock.out'))

    if not os.path.exists(out_dock + 'model_1.pdb'):
        args_pr = ['createpl', './Hdock.out', 'top100.pdb', '-complex', '-nmax', '100', '-models']
        process = Popen(args_pr, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
    else:
        print("[ {} ] model already exists.".format(out_dock))

    return new_ch1, new_ch2


def read_affinity(file):
    data = [x.strip('\n') for x in open(file, 'r').readlines()]
    affinity = data[9].split('\t')[5]

    return affinity


def run_dock_one(pid1, pid2):
    if os.path.exists(raw_dir + '/results/' + pid1 + '_' + pid2 + '.pdb'):
        print("exist result", pid1 + '_' + pid2 + '.pdb')
        return "exist result"
    else:
        print("docking {} and {}".format(pid1, pid2))
    out_dock = raw_dir + "/dock/" + pid1 + '_' + pid2 + '/'
    print("outdir {}".format(out_dock))

    if not os.path.exists(out_dock):
        os.mkdir(out_dock)
    # run hdock generate 100 model
    print("run_hdock_one")
    new_ch1, new_ch2 = run_hdock_one(out_dock, pid1, pid2)
    affinitys = []
    print("run foldx ")
    for i in tqdm(range(1, 101)):
        # run foldx
        args_pr = ['foldx', '--command=AnalyseComplex', '--pdb=model_{}.pdb'.format(i),
                   '--analyseComplexChains={},{}'.format(new_ch1, new_ch2)]
        process = Popen(args_pr, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()

        # read affinity & save data
        affinity = read_affinity(out_dock + 'Interaction_model_{}_AC.fxout'.format(i))
        affinitys.append(affinity)

        # delete foldx out
        os.remove(out_dock + 'Indiv_energies_model_{}_AC.fxout'.format(i))
        os.remove(out_dock + 'Interaction_model_{}_AC.fxout'.format(i))
        os.remove(out_dock + 'Interface_Residues_model_{}_AC.fxout'.format(i))
        os.remove(out_dock + 'Summary_model_{}_AC.fxout'.format(i))
        shutil.rmtree(out_dock + 'molecules')

    # choose max model

    max_model = affinitys.index(max(affinitys))
    print("max model is {}".format(max_model + 1))

    # save affinity
    print("saving affinity")
    with open(out_dock + 'affinity.txt', 'w') as out:
        for affinity in affinitys:
            out.write('{}\n'.format(affinity))

    print("saved model to results")
    # copy this model to results
    shutil.copyfile(out_dock + '/model_{}.pdb'.format(max_model + 1),
                    raw_dir + '/results/' + pid1 + '_' + pid2 + '.pdb')
    # delete model file
    for i in range(1, 101):
        os.remove("./model_{}.pdb".format(i))

    return f"{pid1}_{pid2}"


prot_seq_path = "/root/autodl-tmp/processed_data/protein.SHS148k.sequences.dictionary.csv"
protein_name = []
with open(prot_seq_path) as f:
    reader = csv.reader(f)
    for row in reader:
        protein_name.append(row[0])

import pickle

with open("/root/autodl-tmp/processed_data/SHS27k_ppi.pkl", "rb") as tf:
    ppi_list = pickle.load(tf)

# for ppi in ppi_list:
#     pid1 = protein_name[ppi[0]]
#     pid2 = protein_name[ppi[1]]
#     run_dock_one(pid1, pid2)
#     break

results = Parallel(n_jobs=32)(delayed(run_dock_one)(ppi[0], ppi[1]) for ppi in tqdm(ppi_list))
